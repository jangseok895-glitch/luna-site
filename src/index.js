const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
};

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: JSON_HEADERS,
  });
}

function getExtension(mimeType) {
  const types = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
  };

  return types[mimeType] || "jpg";
}

function cleanBase64(value) {
  return String(value || "")
    .replace(/^data:image\/[a-zA-Z0-9.+-]+;base64,/, "")
    .replace(/\s/g, "");
}

function extractOutputText(result) {
  if (typeof result?.output_text === "string" && result.output_text.trim()) {
    return result.output_text.trim();
  }

  const output = Array.isArray(result?.output) ? result.output : [];

  for (const item of output) {
    const contents = Array.isArray(item?.content) ? item.content : [];

    for (const content of contents) {
      if (
        (content?.type === "output_text" || content?.type === "text") &&
        typeof content?.text === "string"
      ) {
        return content.text.trim();
      }
    }
  }

  return "";
}

function parseJsonText(text) {
  if (!text) return null;

  const cleaned = String(text)
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    const firstBrace = cleaned.indexOf("{");
    const lastBrace = cleaned.lastIndexOf("}");

    if (firstBrace >= 0 && lastBrace > firstBrace) {
      try {
        return JSON.parse(cleaned.slice(firstBrace, lastBrace + 1));
      } catch {
        return null;
      }
    }

    return null;
  }
}

function normalizeRecords(data) {
  const records = Array.isArray(data?.records)
    ? data.records
    : Array.isArray(data)
      ? data
      : [];

  return records
    .map((record) => ({
      mapName: String(
        record?.mapName ??
          record?.map_name ??
          record?.map ??
          record?.맵이름 ??
          "",
      ).trim(),

      record: String(
        record?.record ??
          record?.time ??
          record?.기록 ??
          "",
      ).trim(),

      confidence: Number(record?.confidence ?? 0),

      sourceImage: Number(record?.sourceImage ?? record?.imageIndex ?? 0),
    }))
    .filter((record) => record.mapName && record.record);
}

async function analyzeImages(request, env) {
  if (!env.OPENAI_API_KEY) {
    return jsonResponse(
      {
        ok: false,
        error:
          "Cloudflare에 OPENAI_API_KEY 비밀 변수가 등록되지 않았습니다.",
      },
      500,
    );
  }

  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      {
        ok: false,
        error: "요청 내용을 읽을 수 없습니다.",
      },
      400,
    );
  }

  const images = Array.isArray(body?.images) ? body.images : [];

  if (!images.length) {
    return jsonResponse(
      {
        ok: false,
        error: "분석할 스크린샷을 첨부해주세요.",
      },
      400,
    );
  }

  if (images.length > 10) {
    return jsonResponse(
      {
        ok: false,
        error: "스크린샷은 최대 10장까지 첨부할 수 있습니다.",
      },
      400,
    );
  }

  const inputContent = [
    {
      type: "input_text",
      text: `
당신은 카트라이더 러쉬플러스 경기기록 스크린샷 판독기입니다.

첨부된 이미지에서 사용자의 본인 기록을 찾아주세요.

반드시 다음 규칙을 따르세요.

1. 이미지에 실제로 보이는 맵 이름과 기록만 추출합니다.
2. 기록은 "분.초.센티초" 형식으로 통일합니다.
   예시: 1.32.57, 0.58.99, 2.01.30
3. 같은 맵이 여러 번 보이면 가장 빠른 기록만 남깁니다.
4. 맵 이름이나 기록이 불분명하면 억지로 추측하지 않습니다.
5. 다른 사람의 기록, 랭킹표의 다른 선수 기록은 제외합니다.
6. 맵 이름은 이미지에 표시된 한국어 명칭을 최대한 그대로 사용합니다.
7. 설명 문장은 쓰지 말고 JSON만 반환합니다.

반환 형식:

{
  "records": [
    {
      "mapName": "맵 이름",
      "record": "1.32.57",
      "confidence": 0.95,
      "sourceImage": 1
    }
  ],
  "warnings": [
    "판독이 어려운 내용이 있을 경우 작성"
  ]
}

기록을 찾지 못한 경우:

{
  "records": [],
  "warnings": ["확인 가능한 본인 기록을 찾지 못했습니다."]
}
      `.trim(),
    },
  ];

  images.forEach((image, index) => {
    const mimeType = String(image?.mimeType || "image/jpeg").toLowerCase();
    const base64 = cleanBase64(image?.base64 || image?.data || "");

    if (!base64) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(mimeType)) {
      return;
    }

    const extension = getExtension(mimeType);

    inputContent.push({
      type: "input_text",
      text: `스크린샷 ${index + 1}번 파일: ${
        image?.name || `image-${index + 1}.${extension}`
      }`,
    });

    inputContent.push({
      type: "input_image",
      image_url: `data:${mimeType};base64,${base64}`,
      detail: "high",
    });
  });

  if (inputContent.length <= 1) {
    return jsonResponse(
      {
        ok: false,
        error: "사용할 수 있는 이미지가 없습니다. JPG, PNG 또는 WEBP를 올려주세요.",
      },
      400,
    );
  }

  const model = env.OPENAI_MODEL || "gpt-4.1-mini";

  let openAIResponse;

  try {
    openAIResponse = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        authorization: `Bearer ${env.OPENAI_API_KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        input: [
          {
            role: "user",
            content: inputContent,
          },
        ],
        temperature: 0,
        max_output_tokens: 4000,
      }),
    });
  } catch (error) {
    return jsonResponse(
      {
        ok: false,
        error: "OpenAI 서버에 연결하지 못했습니다.",
        detail: String(error?.message || error),
      },
      502,
    );
  }

  const result = await openAIResponse.json().catch(() => null);

  if (!openAIResponse.ok) {
    console.error("OpenAI API error:", result);

    return jsonResponse(
      {
        ok: false,
        error:
          result?.error?.message ||
          `AI 분석 요청에 실패했습니다. 상태코드: ${openAIResponse.status}`,
      },
      openAIResponse.status,
    );
  }

  const outputText = extractOutputText(result);
  const parsed = parseJsonText(outputText);

  if (!parsed) {
    return jsonResponse(
      {
        ok: false,
        error: "AI 분석 결과를 JSON 형식으로 읽지 못했습니다.",
        rawText: outputText,
      },
      502,
    );
  }

  const records = normalizeRecords(parsed);
  const warnings = Array.isArray(parsed?.warnings)
    ? parsed.warnings.map((item) => String(item))
    : [];

  return jsonResponse({
    ok: true,
    records,
    warnings,
    model,
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers": "content-type",
        },
      });
    }

    if (url.pathname === "/analyze") {
      if (request.method !== "POST") {
        return jsonResponse(
          {
            ok: false,
            error: "POST 요청만 사용할 수 있습니다.",
          },
          405,
        );
      }

      return analyzeImages(request, env);
    }

    if (url.pathname === "/api/health") {
      return jsonResponse({
        ok: true,
        service: "luna-site",
        model: env.OPENAI_MODEL || "gpt-4.1-mini",
        hasOpenAIKey: Boolean(env.OPENAI_API_KEY),
      });
    }

    return env.ASSETS.fetch(request);
  },
};
