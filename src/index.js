const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
  "access-control-allow-origin": "*",
};

const ALLOWED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

const MAX_IMAGES = 10;

// Base64 문자열 기준 약 12MB입니다.
// 일반적으로 원본 이미지 약 9MB 정도에 해당합니다.
const MAX_BASE64_CHARS_PER_IMAGE = 12_000_000;

// 여러 장을 한 번에 올릴 때 Worker/OpenAI 요청이 지나치게 커지는 것을 방지합니다.
const MAX_TOTAL_BASE64_CHARS = 45_000_000;

const RECORD_RESPONSE_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    records: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          mapName: {
            type: "string",
          },
          record: {
            type: "string",
          },
          confidence: {
            type: "number",
            minimum: 0,
            maximum: 1,
          },
          sourceImage: {
            type: "integer",
            minimum: 1,
          },
          sourceType: {
            type: "string",
            enum: ["time_attack", "ranked_result", "unknown"],
          },
          evidence: {
            type: "string",
          },
        },
        required: [
          "mapName",
          "record",
          "confidence",
          "sourceImage",
          "sourceType",
          "evidence",
        ],
      },
    },
    warnings: {
      type: "array",
      items: {
        type: "string",
      },
    },
  },
  required: ["records", "warnings"],
};

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

function sanitizeMimeType(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .split(";")[0];
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
        typeof content?.text === "string" &&
        content.text.trim()
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

  // 1:23:45 / 1.23.45 / 1'23"45처럼 구분자가 달라도 처리합니다.
  const match = raw.match(/(\d{1,2})\D+(\d{1,2})\D+(\d{1,3})/);
  if (!match) return "";

  const minutes = Number(match[1]);
  const seconds = Number(match[2]);
  let fraction = String(match[3]);

  if (
    !Number.isFinite(minutes) ||
    !Number.isFinite(seconds) ||
    seconds < 0 ||
    seconds > 59
  ) {
    return "";
  }

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

function normalizeMapKey(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/[^\p{Letter}\p{Number}]/gu, "");
}

function normalizeRecords(data) {
  const source = Array.isArray(data?.records)
    ? data.records
    : Array.isArray(data)
      ? data
      : [];

  const bestByMap = new Map();

  for (const record of source) {
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

    if (!mapName || !time) continue;

    const confidenceValue = Number(record?.confidence ?? 0);

    const normalized = {
      mapName,
      record: time,
      confidence: Number.isFinite(confidenceValue)
        ? Math.max(0, Math.min(1, confidenceValue))
        : 0,
      sourceImage: Math.max(
        1,
        Math.trunc(
          Number(record?.sourceImage ?? record?.imageIndex ?? 1) || 1,
        ),
      ),
      sourceType: ["time_attack", "ranked_result", "unknown"].includes(
        String(record?.sourceType || ""),
      )
        ? String(record.sourceType)
        : "unknown",
      evidence: String(record?.evidence || "").trim(),
    };

    const key = normalizeMapKey(mapName);
    if (!key) continue;

    const previous = bestByMap.get(key);

    if (
      !previous ||
      recordToMilliseconds(normalized.record) <
        recordToMilliseconds(previous.record)
    ) {
      bestByMap.set(key, normalized);
    }
  }

  return Array.from(bestByMap.values());
}

function buildVisionPrompt(imageNumber, retryMode = false) {
  return `
당신은 카트라이더 러쉬플러스 기록 스크린샷을 판독하는 전문 분석기입니다.
현재 첨부된 이미지는 ${imageNumber}번 이미지입니다.
이 이미지 한 장에서 맵 이름과 사용자의 본인 기록만 정확하게 추출하세요.

중요: 랭킹전/경기 종료 순위표에서는 "노란색 평균 속도"만 보고 행을 선택하면 안 됩니다.
반드시 "완주 기록 숫자 자체"가 노란색 또는 금색인지 먼저 확인해야 합니다.

지원 화면 유형

A. 타임어택 또는 개인 기록 화면
- 화면에 표시된 맵 이름을 읽습니다.
- 개인 최고 기록, 내 기록, 본인 기록처럼 사용자의 기록임이 명확한 값만 추출합니다.
- 다른 사람의 기록이나 비교 기록은 제외합니다.

B. 랭킹전 또는 경기 종료 순위표
- 표를 위에서 아래까지 행 단위로 확인합니다.
- 각 행에서 닉네임, 완주 기록, 점수, 카트, 평균 속도가 같은 가로줄에 있는지 확인합니다.
- 사용자의 행은 "완주 기록 숫자"가 노란색/금색으로 표시된 행입니다.
- 가능하면 같은 행의 평균 속도도 노란색/금색인지 추가로 확인합니다.
- 평균 속도만 노란색이고 완주 기록 숫자가 흰색이면 그 행은 본인 행으로 선택하지 않습니다.
- 여러 행의 평균 속도가 노란색처럼 보여도, 완주 기록 숫자까지 노란색인 행 하나를 우선 선택합니다.
- 예: 첫 번째 행 기록이 01:58:68이고 노란색이며, 두 번째 행 기록 02:00:68은 흰색이라면 반드시 01:58:68을 선택해야 합니다.
- 상단 왼쪽 또는 상단 영역의 맵 이름과 선택한 동일 행의 완주 기록을 한 쌍으로 추출합니다.
- 평균 속도는 본인 행을 찾는 근거로만 사용하며 record 값에는 넣지 않습니다.
- '미완료', 리타이어, 완주 기록 없음은 반환하지 않습니다.

판독 절차

1. 맵 이름을 찾습니다.
2. 순위표 화면이면 각 행의 완주 기록 숫자 색상을 먼저 비교합니다.
3. 노란색/금색 완주 기록 숫자가 있는 행을 찾습니다.
4. 그 행의 기록을 정확히 문자 단위로 읽습니다.
5. 기록은 화면의 01:58:68을 1.58.68로 변환하는 식으로 반환합니다.
6. 다른 행의 기록과 섞지 않습니다.
7. 기록 숫자가 작거나 흐려도 확대해서 다시 확인하듯 신중하게 읽습니다.
8. 후보가 정확히 하나이고 색상 근거가 충분하면 confidence 0.70 이상으로 반환합니다.
9. 완주 기록 숫자의 색상을 전혀 구분할 수 없을 때만 제외합니다.

필수 규칙

1. 이미지에서 실제로 확인되는 정보만 사용합니다.
2. 기록 형식은 반드시 "분.초.센티초"로 반환합니다.
3. 다른 선수의 기록을 본인 기록으로 반환하지 않습니다.
4. sourceImage는 ${imageNumber}입니다.
5. sourceType:
   - 타임어택/개인 기록 화면: time_attack
   - 랭킹전/경기 결과 순위표: ranked_result
   - 구분 불가: unknown
6. evidence에는 판독 근거를 구체적으로 적습니다.
   예: "완주 기록 01:58:68이 노란색이고 같은 행 평균속도도 노란색"
7. 확인 가능한 본인 기록이 없으면 records를 빈 배열로 반환하고 warnings에 이유를 적습니다.
8. 반드시 지정된 JSON 형식으로만 응답합니다.

${retryMode ? `
재검토 모드:
- 첫 판독에서 기록을 찾지 못했으므로, 이번에는 완주 기록 숫자의 노란색/금색 여부를 가장 우선해서 다시 확인하세요.
- 평균 속도 색상만으로 포기하지 말고, 각 행의 기록 숫자를 한 줄씩 비교하세요.
- 노란색 기록 후보가 하나라면 낮은 confidence라도 반환하세요.
` : ""}
  `.trim();
}

function getCloudflareLocation(request) {
  return {
    country: request.cf?.country || null,
    city: request.cf?.city || null,
    region: request.cf?.region || null,
    colo: request.cf?.colo || null,
  };
}

async function readOpenAIResponse(openAIResponse) {
  const responseText = await openAIResponse.text().catch(() => "");
  let parsed = null;

  if (responseText) {
    try {
      parsed = JSON.parse(responseText);
    } catch {
      parsed = null;
    }
  }

  return {
    responseText,
    parsed,
    requestId:
      openAIResponse.headers.get("x-request-id") ||
      openAIResponse.headers.get("request-id") ||
      null,
  };
}

function buildUpstreamError(openAIResponse, responseText, parsed, requestId) {
  const message = String(
    parsed?.error?.message ||
      parsed?.message ||
      responseText ||
      "AI 분석 요청에 실패했습니다.",
  )
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 1500);

  return {
    ok: false,
    error: message,
    status: openAIResponse.status,
    code: parsed?.error?.code || parsed?.code || null,
    type: parsed?.error?.type || parsed?.type || null,
    requestId,
  };
}

async function callVisionForSingleImage({
  env,
  image,
  imageNumber,
  retryMode = false,
}) {
  const model = env.OPENAI_MODEL || "gpt-4.1-mini";

  const inputContent = [
    {
      type: "input_text",
      text: buildVisionPrompt(imageNumber, retryMode),
    },
    {
      type: "input_text",
      text: `첨부 이미지 ${imageNumber}번 파일명: ${image.name}`,
    },
    {
      type: "input_image",
      image_url: `data:${image.mimeType};base64,${image.base64}`,
      detail: "high",
    },
  ];

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
        store: false,
        input: [
          {
            role: "user",
            content: inputContent,
          },
        ],
        text: {
          format: {
            type: "json_schema",
            name: "kart_record_analysis",
            description:
              "카트라이더 러쉬플러스 스크린샷에서 본인 맵 기록을 추출한 결과",
            strict: true,
            schema: RECORD_RESPONSE_SCHEMA,
          },
        },
        max_output_tokens: 1800,
      }),
    });
  } catch (error) {
    return {
      ok: false,
      status: 502,
      error: "AI 분석 서버에 연결하지 못했습니다.",
      detail: String(error?.message || error || ""),
      requestId: null,
    };
  }

  const {
    responseText,
    parsed: openAIResult,
    requestId,
  } = await readOpenAIResponse(openAIResponse);

  if (!openAIResponse.ok) {
    return {
      ...buildUpstreamError(
        openAIResponse,
        responseText,
        openAIResult,
        requestId,
      ),
      model,
    };
  }

  const outputText = extractOutputText(openAIResult);
  const parsedResult = parseJsonText(outputText);

  if (!parsedResult) {
    return {
      ok: false,
      status: 502,
      error: "AI 분석 결과를 읽지 못했습니다.",
      requestId,
    };
  }

  return {
    ok: true,
    status: 200,
    records: normalizeRecords(parsedResult),
    warnings: Array.isArray(parsedResult?.warnings)
      ? parsedResult.warnings
          .map((item) => String(item || "").trim())
          .filter(Boolean)
      : [],
    requestId,
  };
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

  const acceptedImages = [];
  const rejectedImages = [];
  let totalBase64Chars = 0;

  images.forEach((image, index) => {
    const mimeType = sanitizeMimeType(
      image?.mimeType || image?.type || "image/jpeg",
    );

    const base64 = cleanBase64(image?.base64 || image?.data || "");
    const name = String(image?.name || `screenshot-${index + 1}`);

    if (!ALLOWED_IMAGE_TYPES.has(mimeType)) {
      rejectedImages.push(`${name}: 지원하지 않는 파일 형식`);
      return;
    }

    if (!base64) {
      rejectedImages.push(`${name}: 이미지 데이터 없음`);
      return;
    }

    if (base64.length > MAX_BASE64_CHARS_PER_IMAGE) {
      rejectedImages.push(`${name}: 파일 용량이 너무 큼`);
      return;
    }

    if (totalBase64Chars + base64.length > MAX_TOTAL_BASE64_CHARS) {
      rejectedImages.push(`${name}: 전체 첨부 용량 제한 초과`);
      return;
    }

    totalBase64Chars += base64.length;
    acceptedImages.push({
      name,
      mimeType,
      base64,
      sourceImage: index + 1,
    });
  });

  if (!acceptedImages.length) {
    return jsonResponse(
      {
        ok: false,
        error:
          "사용할 수 있는 이미지가 없습니다. JPG, PNG 또는 WEBP 파일을 올려주세요.",
        rejectedImages,
      },
      400,
    );
  }

  const allRecords = [];
  const warnings = [];
  const requestIds = [];

  // 여러 이미지를 한 요청에 섞지 않고 한 장씩 분석하여 행/기록 혼동을 줄입니다.
  for (const image of acceptedImages) {
    let result = await callVisionForSingleImage({
      env,
      image,
      imageNumber: image.sourceImage,
      retryMode: false,
    });

    // 첫 판독에서 기록을 못 찾은 경우 랭전 노란색 기록 중심으로 한 번 더 재검토합니다.
    if (result.ok && !result.records.length) {
      const retryResult = await callVisionForSingleImage({
        env,
        image,
        imageNumber: image.sourceImage,
        retryMode: true,
      });

      if (retryResult.ok && retryResult.records.length) {
        result = retryResult;
      } else if (retryResult.ok) {
        warnings.push(...retryResult.warnings);
      }
    }

    if (!result.ok) {
      return jsonResponse(
        {
          ok: false,
          error: result.error || "AI 분석 요청에 실패했습니다.",
          status: result.status || 502,
          code: result.code || null,
          type: result.type || null,
          requestId: result.requestId || null,
          cf: getCloudflareLocation(request),
        },
        result.status || 502,
      );
    }

    allRecords.push(...result.records);
    warnings.push(...result.warnings);

    if (result.requestId) {
      requestIds.push(result.requestId);
    }
  }

  // 여러 이미지에서 같은 맵이 나온 경우 가장 빠른 기록만 유지합니다.
  const records = normalizeRecords({ records: allRecords });

  return jsonResponse({
    ok: true,
    records,
    warnings: Array.from(new Set(warnings)),
    model: env.OPENAI_MODEL || "gpt-4.1-mini",
    analyzedImages: acceptedImages.length,
    rejectedImages,
    requestIds,
  });
}

async function testOpenAIConnection(request, env) {
  if (!env.OPENAI_API_KEY) {
    return jsonResponse(
      {
        ok: false,
        error: "OPENAI_API_KEY가 등록되지 않았습니다.",
      },
      500,
    );
  }

  const model = env.OPENAI_MODEL || "gpt-4.1-mini";
  let testResponse;

  try {
    testResponse = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        authorization: `Bearer ${env.OPENAI_API_KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        store: false,
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
        cf: getCloudflareLocation(request),
      },
      502,
    );
  }

  const {
    responseText,
    parsed,
    requestId,
  } = await readOpenAIResponse(testResponse);

  if (!testResponse.ok) {
    return jsonResponse(
      {
        ...buildUpstreamError(
          testResponse,
          responseText,
          parsed,
          requestId,
        ),
        model,
        cf: getCloudflareLocation(request),
      },
      testResponse.status,
    );
  }

  return jsonResponse({
    ok: true,
    status: testResponse.status,
    model,
    output: extractOutputText(parsed) || null,
    requestId,
    cf: getCloudflareLocation(request),
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
          {
            allow: "POST",
          },
        );
      }

      return analyzeImages(request, env);
    }

    if (url.pathname === "/api/test") {
      if (request.method !== "GET") {
        return jsonResponse(
          {
            ok: false,
            error: "GET 요청만 사용할 수 있습니다.",
          },
          405,
          {
            allow: "GET",
          },
        );
      }

      return testOpenAIConnection(request, env);
    }

    if (url.pathname === "/api/health") {
      return jsonResponse({
        ok: true,
        service: "luna-site",
        model: env.OPENAI_MODEL || "gpt-4.1-mini",
        hasOpenAIKey: Boolean(env.OPENAI_API_KEY),
        cf: getCloudflareLocation(request),
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
