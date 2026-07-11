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

function buildVisionPrompt(imageNumber, retryMode = false, hasAquaCrop = false) {
  return `
당신은 카트라이더 러쉬플러스 기록 스크린샷 판독기입니다.
현재 이미지는 ${imageNumber}번입니다.
한 이미지에 보이는 모든 유효한 맵 이름과 기록을 빠짐없이 추출하세요.

절대 규칙
- "사진 1장 = 기록 1개"라고 가정하지 마세요.
- 한 화면에 기록 카드가 2개, 6개, 10개 이상 보이면 보이는 유효 기록을 모두 records 배열에 넣으세요.
- 첫 번째 기록 하나를 찾았다고 분석을 끝내지 마세요. 화면 전체를 왼쪽 위부터 오른쪽 아래까지 끝까지 훑으세요.
- 같은 맵이 중복되면 가장 빠른 기록만 남기세요.
- '미완료', 기록 없음, 잠금 상태는 반환하지 마세요.
- 노란색 글씨는 1등 표시일 수 있으므로 본인 판별 근거로 사용하지 마세요.

먼저 화면 유형을 구분하세요.

A. 프로필 → 기록 / 마스터 기록 / 트랙 기록 카드 목록 화면
- 화면에 여러 개의 맵 카드가 격자 형태로 배치됩니다.
- 각 카드에는 맵 이름과 '기록 01:14:37' 같은 값이 함께 표시됩니다.
- 이 화면에서는 아쿠아 선택 행을 찾지 않습니다.
- 각 카드의 맵 이름과 바로 그 카드 안의 기록을 정확히 한 쌍으로 묶습니다.
- 왼쪽 열과 오른쪽 열을 모두 읽고, 다음 행까지 계속 내려가며 화면에 보이는 모든 카드를 추출합니다.
- 예: 6개 카드가 보이면 records도 원칙적으로 6개여야 합니다.
- 카드 일부가 잘렸거나 기록이 불명확한 카드만 제외하고, 나머지는 모두 반환합니다.
- sourceType은 time_attack으로 반환합니다.

B. 타임어택 / 개인 기록 상세 화면
- 개인 최고기록, 내 기록 등 본인 기록임이 명확한 값을 추출합니다.
- 한 화면에 여러 맵과 기록이 있으면 역시 모두 추출합니다.
- sourceType은 time_attack으로 반환합니다.

C. 랭킹전 / 경기 종료 순위표
- 이 화면에서만 본인 행 선택 표시를 사용합니다.
- 1등의 기록·닉네임·평균속도가 노란색이어도 본인이라는 뜻이 아닙니다. 노란색은 절대 사용하지 마세요.
- 본인 표식은 표 왼쪽 끝에 행 높이만큼 보이는 조금 두껍고 밝은 아쿠아색 세로 강조선입니다.
- 이 세로 강조선이 붙어 있는 정확한 가로 행만 본인 행입니다. 바로 위나 바로 아래의 더 밝은 행을 선택하지 마세요.
- 원본에서 세로 강조선이 3위 행에 붙어 있으면, 1위 기록이 노란색이어도 반드시 3위 행 기록을 선택합니다.
- 선택된 동일 행의 완주 기록만 읽고, '미완료'면 반환하지 않습니다.
- 상단 맵 이름과 본인 행 기록을 한 쌍으로 반환합니다.
- sourceType은 ranked_result로 반환합니다.

${hasAquaCrop ? `
첨부 이미지 안내
- 첫 번째 이미지는 전체 원본입니다. 화면 유형 판별, 맵 이름, 여러 기록 카드 전체 판독은 반드시 원본을 기준으로 합니다.
- 두 번째 이미지는 브라우저가 왼쪽의 굵은 아쿠아 세로 강조선을 직접 탐지해 잘라낸 본인 행 후보입니다.
- 랭킹전 화면으로 확인되면 두 번째 Crop의 행이 본인 행이며, Crop 안의 기록을 최우선·사실상 확정값으로 사용하세요.
- 랭킹전에서 원본의 1등 노란 기록과 Crop 기록이 다르면 반드시 Crop 기록을 선택하세요.
- Crop 바로 위·아래 행의 기록을 원본에서 가져오지 마세요.
- 원본이 여러 기록 카드 화면이라면 Crop 이미지는 완전히 무시하고 원본의 모든 카드를 추출하세요.
- Crop 때문에 원본의 다른 기록 카드들을 누락해서는 안 됩니다.
` : `
보조 Crop이 없습니다. 랭킹전이라면 원본의 왼쪽 아쿠아 선택 테두리를 찾고, 기록 카드 화면이라면 모든 카드를 읽으세요.
`}

기록 형식
- 01:53:77 → 1.53.77
- 00:58:68 → 0.58.68
- 03:20:75 → 3.20.75
- 맵 이름과 기록이 서로 다른 카드나 행에서 섞이지 않도록 하세요.

반환 전 자체 점검
1. 화면에 보이는 유효 기록 개수를 다시 셉니다.
2. records 배열 개수가 그 수와 대체로 일치하는지 확인합니다.
3. 6개 카드 화면인데 1개만 반환했다면 화면을 다시 훑어 나머지 5개를 찾습니다.
4. 왼쪽 열뿐 아니라 오른쪽 열도 읽었는지 확인합니다.
5. 화면 아래쪽 카드까지 빠짐없이 확인합니다.

필수 반환 규칙
- sourceImage는 ${imageNumber}입니다.
- sourceType은 time_attack, ranked_result, unknown 중 하나입니다.
- evidence에는 판독 근거를 간단히 적습니다.
  예: "마스터 기록 카드의 맵 이름과 기록을 같은 카드에서 판독"
  예: "왼쪽의 굵은 아쿠아 세로 강조선이 3위 행에 붙어 있고, 보조 Crop에서도 3위 기록을 확인함"
- 확인할 수 없는 항목은 억지로 추측하지 말고 warning에 적습니다.
- 지정된 JSON 스키마 외 텍스트는 출력하지 마세요.

${retryMode ? `
재검토 모드
- 첫 판독에서 기록을 하나도 찾지 못했습니다.
- 프로필 기록 카드 화면인지 먼저 다시 확인하고, 카드가 여러 개면 모두 읽으세요.
- 랭킹전 화면일 때만 왼쪽 아쿠아 테두리를 다시 추적하세요.
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
      text: buildVisionPrompt(imageNumber, retryMode, Boolean(image.aquaCropBase64)),
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

  if (image.aquaCropBase64) {
    inputContent.push(
      {
        type: "input_text",
        text: `아래 이미지는 브라우저가 왼쪽의 굵은 아쿠아 선택 세로선을 탐지해 자른 본인 행 후보입니다. 탐지 신뢰도: ${Number(image.aquaCropConfidence || 0).toFixed(3)}. 랭킹전이면 이 Crop 행의 기록을 원본의 노란 1등 기록보다 우선하세요.`,
      },
      {
        type: "input_image",
        image_url: `data:${image.aquaCropMimeType || "image/jpeg"};base64,${image.aquaCropBase64}`,
        detail: "high",
      },
    );
  }

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
        max_output_tokens: 7000,
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

    let aquaCropBase64 = cleanBase64(image?.aquaCropBase64 || "");
    let aquaCropMimeType = sanitizeMimeType(image?.aquaCropMimeType || "image/jpeg");

    if (!ALLOWED_IMAGE_TYPES.has(aquaCropMimeType)) {
      aquaCropBase64 = "";
      aquaCropMimeType = "";
    }

    if (aquaCropBase64 && aquaCropBase64.length > MAX_BASE64_CHARS_PER_IMAGE) {
      aquaCropBase64 = "";
      aquaCropMimeType = "";
    }

    if (aquaCropBase64 && totalBase64Chars + base64.length + aquaCropBase64.length > MAX_TOTAL_BASE64_CHARS) {
      aquaCropBase64 = "";
      aquaCropMimeType = "";
    }

    totalBase64Chars += base64.length + aquaCropBase64.length;
    acceptedImages.push({
      name,
      mimeType,
      base64,
      aquaCropBase64,
      aquaCropMimeType,
      aquaCropFound: Boolean(image?.aquaCropFound && aquaCropBase64),
      aquaCropConfidence: Number(image?.aquaCropConfidence || 0),
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

    // 첫 판독에서 기록을 못 찾은 경우 화면 유형을 다시 판별하여 한 번 더 재검토합니다.
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
    aquaCropsUsed: acceptedImages.filter((image) => image.aquaCropBase64).length,
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
