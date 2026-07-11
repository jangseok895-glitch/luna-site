const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
};

const ALLOWED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

const MAX_IMAGES = 10;
const MAX_BASE64_CHARS_PER_IMAGE = 12_000_000;

function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      ...JSON_HEADERS,
      ...extraHeaders,
    },
  });
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
  }

  return null;
}

function normalizeRecordTime(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";

  const match = raw.match(/(\d{1,2})\D+(\d{1,2})\D+(\d{1,3})/);
  if (!match) return raw;

  const minutes = Number(match[1]);
  const seconds = Number(match[2]);
  let fraction = String(match[3]);

  if (!Number.isFinite(minutes) || !Number.isFinite(seconds)) return raw;

  if (fraction.length === 1) fraction += "0";
  if (fraction.length > 2) fraction = fraction.slice(0, 2);

  return `${minutes}.${String(seconds).padStart(2, "0")}.${fraction.padStart(2, "0")}`;
}

function recordToMilliseconds(value) {
  const normalized = normalizeRecordTime(value);
  const match = normalized.match(/^(\d+)\.(\d{2})\.(\d{2})$/);
  if (!match) return Number.POSITIVE_INFINITY;

  return (
    Number(match[1]) * 60_000 +
    Number(match[2]) * 1_000 +
    Number(match[3]) * 10
  );
}

function normalizeRecords(data) {
  const source = Array.isArray(data?.records)
    ? data.records
    : Array.isArray(data)
      ? data
      : [];

  const bestByMap = new Map();

  source.forEach((record) => {
    const mapName = String(
      record?.mapName ??
        record?.map_name ??
        record?.map ??
        record?.맵이름 ??
        "",
    ).trim();

    const time = normalizeRecordTime(
      record?.record ??
        record?.time ??
        record?.기록 ??
        "",
    );

    if (!mapName || !time) return;

    const normalized = {
      mapName,
      record: time,
      confidence: Math.max(
        0,
        Math.min(1, Number(record?.confidence ?? 0)),
      ),
      sourceImage: Math.max(
        1,
        Number(record?.sourceImage ?? record?.imageIndex ?? 1),
      ),
      sourceType: String(
        record?.sourceType ??
          record?.source_type ??
          "unknown",
      ).trim(),
      evidence: String(record?.evidence ?? "").trim(),
    };

    const key = mapName
      .toLowerCase()
      .replace(/\s+/g, "")
      .replace(/[^\p{Letter}\p{Number}]/gu, "");

    const previous = bestByMap.get(key);

    if (
      !previous ||
      recordToMilliseconds(normalized.record) <
        recordToMilliseconds(previous.record)
    ) {
      bestByMap.set(key, normalized);
    }
  });

  return Array.from(bestByMap.values());
}

function buildVisionPrompt() {
  return `
당신은 카트라이더 러쉬플러스 기록 스크린샷 판독기입니다.
첨부된 각 이미지에서 맵 이름과 사용자의 본인 기록을 판독하세요.

지원 화면 유형:

A. 타임어택/개인 기록 화면
- 화면에 표시된 맵 이름과 사용자의 기록을 추출합니다.
- 최고 기록, 개인 기록, 내 기록처럼 사용자의 기록임이 명확한 값만 사용합니다.

B. 랭킹전 또는 경기 종료 결과표
- 한 화면에 최대 8명의 선수와 기록이 표시될 수 있습니다.
- 기록과 평균 속도(내 속도)가 모두 노란색 또는 금색으로 강조된 행을 사용자의 행으로 판단합니다.
- 노란색 강조가 명확하지 않으면 억지로 다른 선수 기록을 선택하지 않습니다.
- 상단에 표시된 맵 이름과 노란색 본인 행의 완주 기록을 한 쌍으로 추출합니다.
- 평균 속도 숫자는 본인 판별에만 사용하며 record 값으로 반환하지 않습니다.

필수 규칙:

1. 이미지에서 실제로 확인되는 맵 이름과 기록만 반환합니다.
2. 기록 형식은 반드시 "분.초.센티초"로 통일합니다.
   예: 1.32.57, 0.58.99, 2.01.30
3. 같은 맵이 여러 이미지에 있으면 가장 빠른 기록만 남깁니다.
4. 맵 이름이나 본인 기록이 불명확하면 추측하지 않습니다.
5. 다른 선수의 기록을 절대 본인 기록으로 반환하지 않습니다.
6. sourceImage는 첨부 이미지 순번이며 첫 번째 이미지는 1입니다.
7. sourceType은 타임어택이면 "time_attack", 랭전/결과표이면 "ranked_result"로 작성합니다.
8. evidence에는 본인 기록이라고 판단한 짧은 근거를 작성합니다.
   예: "개인 최고기록 표시", "기록과 평균속도가 노란색인 행"
9. 설명문이나 마크다운 없이 JSON 객체만 반환합니다.

반환 형식:

{
  "records": [
    {
      "mapName": "맵 이름",
      "record": "1.32.57",
      "confidence": 0.95,
      "sourceImage": 1,
      "sourceType": "time_attack",
      "evidence": "개인 최고기록 표시"
    }
  ],
  "warnings": []
}

확인 가능한 본인 기록이 없으면:

{
  "records": [],
  "warnings": ["확인 가능한 본인 기록을 찾지 못했습니다."]
}
  `.trim();
}

async function analyzeImages(request, env) {
  if (!env.OPENAI_API_KEY) {
    return jsonResponse(
      {
        ok: false,
        error: "AI 서비스 키가 등록되지 않았습니다.",
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

  if (images.length > MAX_IMAGES) {
    return jsonResponse(
      {
        ok: false,
        error: `스크린샷은 최대 ${MAX_IMAGES}장까지 첨부할 수 있습니다.`,
      },
      400,
    );
  }

  const inputContent = [
    {
      type: "input_text",
      text: buildVisionPrompt(),
    },
  ];

  let acceptedImages = 0;

  images.forEach((image, index) => {
    const mimeType = String(
      image?.mimeType || image?.type || "image/jpeg",
    ).toLowerCase();

    const base64 = cleanBase64(image?.base64 || image?.data || "");

    if (!ALLOWED_IMAGE_TYPES.has(mimeType)) return;
    if (!base64 || base64.length > MAX_BASE64_CHARS_PER_IMAGE) return;

    acceptedImages += 1;

    inputContent.push({
      type: "input_text",
      text: `첨부 이미지 ${index + 1}번: ${String(
        image?.name || `screenshot-${index + 1}`,
      )}`,
    });

    inputContent.push({
      type: "input_image",
      image_url: `data:${mimeType};base64,${base64}`,
      detail: "high",
    });
  });

  if (!acceptedImages) {
    return jsonResponse(
      {
        ok: false,
        error:
          "사용할 수 있는 이미지가 없습니다. JPG, PNG 또는 WEBP 파일을 올려주세요.",
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
        max_output_tokens: 3000,
      }),
    });
  } catch (error) {
    console.error("AI connection error:", error);

    return jsonResponse(
      {
        ok: false,
        error: "AI 분석 서버에 연결하지 못했습니다.",
      },
      502,
    );
  }

  // 오류 응답이 JSON이 아닐 수도 있으므로 먼저 원문 텍스트로 읽습니다.
  const responseText = await openAIResponse.text().catch(() => "");
  let result = null;

  if (responseText) {
    try {
      result = JSON.parse(responseText);
    } catch {
      result = null;
    }
  }

  if (!openAIResponse.ok) {
    const upstreamMessage = String(
      result?.error?.message ||
      result?.message ||
      responseText ||
      "",
    )
      .replace(/<[^>]*>/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 1200);

    const upstreamCode = String(
      result?.error?.code ||
      result?.code ||
      "",
    ).trim();

    const upstreamType = String(
      result?.error?.type ||
      result?.type ||
      "",
    ).trim();

    console.error("AI API error:", {
      status: openAIResponse.status,
      message: upstreamMessage,
      code: upstreamCode,
      type: upstreamType,
    });

    const detailParts = [
      upstreamMessage,
      upstreamCode ? `오류 코드: ${upstreamCode}` : "",
      upstreamType ? `오류 유형: ${upstreamType}` : "",
      `상태코드: ${openAIResponse.status}`,
    ].filter(Boolean);

    return jsonResponse(
      {
        ok: false,
        error: detailParts.join("\n"),
        status: openAIResponse.status,
        code: upstreamCode || null,
        type: upstreamType || null,
      },
      openAIResponse.status >= 400 && openAIResponse.status < 600
        ? openAIResponse.status
        : 502,
    );
  }

  const outputText = extractOutputText(result);
  const parsed = parseJsonText(outputText);

  if (!parsed) {
    console.error("Invalid AI JSON:", outputText);

    return jsonResponse(
      {
        ok: false,
        error: "AI 분석 결과를 읽지 못했습니다. 다시 시도해주세요.",
      },
      502,
    );
  }

  const records = normalizeRecords(parsed);
  const warnings = Array.isArray(parsed?.warnings)
    ? parsed.warnings.map((item) => String(item || "").trim()).filter(Boolean)
    : [];

  return jsonResponse({
    ok: true,
    records,
    warnings,
    model,
    analyzedImages: acceptedImages,
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
          "access-control-max-age": "86400",
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
          { allow: "POST" },
        );
      }

      return analyzeImages(request, env);
    }

    if (url.pathname === "/api/test") {
      if (!env.OPENAI_API_KEY) {
        return jsonResponse(
          {
            ok: false,
            error: "OPENAI_API_KEY가 등록되지 않았습니다.",
          },
          500,
        );
      }

      let testResponse;

      try {
        testResponse = await fetch("https://api.openai.com/v1/responses", {
          method: "POST",
          headers: {
            authorization: `Bearer ${env.OPENAI_API_KEY}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: env.OPENAI_MODEL || "gpt-4.1-mini",
            input: "Reply with exactly: TEST_OK",
            max_output_tokens: 20,
          }),
        });
      } catch (error) {
        return jsonResponse(
          {
            ok: false,
            stage: "connection",
            error: String(error?.message || error || "연결 실패"),
          },
          502,
        );
      }

      const responseText = await testResponse.text().catch(() => "");
      let parsed = null;

      if (responseText) {
        try {
          parsed = JSON.parse(responseText);
        } catch {
          parsed = null;
        }
      }

      const message = String(
        parsed?.error?.message ||
        parsed?.message ||
        responseText ||
        "",
      )
        .replace(/<[^>]*>/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 1200);

      const outputText = parsed ? extractOutputText(parsed) : "";

      return jsonResponse(
        {
          ok: testResponse.ok,
          status: testResponse.status,
          model: env.OPENAI_MODEL || "gpt-4.1-mini",
          output: outputText || null,
          error: testResponse.ok ? null : message || "알 수 없는 오류",
          code: parsed?.error?.code || null,
          type: parsed?.error?.type || null,
          cf: {
            country: request.cf?.country || null,
            city: request.cf?.city || null,
            region: request.cf?.region || null,
            colo: request.cf?.colo || null,
          },
        },
        testResponse.status,
      );
    }

    if (url.pathname === "/api/health") {
      return jsonResponse({
        ok: true,
        service: "luna-site",
        model: env.OPENAI_MODEL || "gpt-4.1-mini",
        hasOpenAIKey: Boolean(env.OPENAI_API_KEY),
      });
    }

    if (!env.ASSETS || typeof env.ASSETS.fetch !== "function") {
      return jsonResponse(
        {
          ok: false,
          error: "정적 파일 바인딩을 찾지 못했습니다.",
        },
        500,
      );
    }

    return env.ASSETS.fetch(request);
  },
};
