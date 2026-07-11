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

const RANK_SELECTION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    screenType: {
      type: "string",
      enum: ["ranked_result", "profile_records", "time_attack", "unknown"],
    },
    selectedRank: {
      anyOf: [
        { type: "integer", minimum: 1, maximum: 8 },
        { type: "null" },
      ],
    },
    confidence: {
      type: "number",
      minimum: 0,
      maximum: 1,
    },
    evidence: {
      type: "string",
    },
  },
  required: ["screenType", "selectedRank", "confidence", "evidence"],
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

function buildVisionPrompt({
  imageNumber,
  retryMode = false,
  hasAquaCrop = false,
  selectedRank = null,
  screenType = "unknown",
  allowedMapNames = [],
}) {
  const allowedList = Array.isArray(allowedMapNames) && allowedMapNames.length
    ? allowedMapNames.map((name) => `- ${name}`).join("\n")
    : "- 허용된 맵 목록이 제공되지 않음";

  return `
당신은 카트라이더 러쉬플러스 기록 스크린샷 판독기입니다.
현재 이미지는 ${imageNumber}번입니다.
화면에 보이는 유효한 맵 이름과 기록을 정확히 추출하세요.

가장 중요한 금지사항
- 카트/차량 이름을 맵 이름으로 반환하지 마세요.
- "아쿠아 비틀", "에어리 코튼", "이그니스 비틀", "세이버 오라클" 같은 값은 차량 이름이므로 절대 mapName으로 반환하면 안 됩니다.
- 랭전에서 1등 노란색 기록은 본인 표시가 아닙니다.
- 허용 맵 목록에 없는 이름은 반환하지 마세요.

허용 맵 목록
${allowedList}

영어 맵 처리
- 화면의 맵 이름이 영어면 의미를 번역하고, 위 허용 맵 목록에서 정확히 대응하는 한글 맵 이름으로 반환하세요.
- 예: "Skull Castle (Tomb)"처럼 영어로 표시되면 허용 목록의 대응 한글 이름을 선택하세요.
- 대응되는 한글 맵을 확실히 찾지 못하면 records에 넣지 말고 warning에 적으세요.

화면 유형별 규칙

A. 프로필 기록 / 마스터 기록 카드 목록
- 한 장에 여러 기록 카드가 보이면 모두 읽으세요.
- 카드의 맵 이름과 같은 카드 안의 기록만 한 쌍으로 묶으세요.
- 첫 번째 카드 하나만 읽고 끝내지 마세요.
- sourceType은 time_attack입니다.

B. 타임어택 / 개인 기록 상세
- 본인 기록임이 명확한 값만 추출하세요.
- sourceType은 time_attack입니다.

C. 랭킹전 / 경기 종료 순위표
- 본인 행은 등수 숫자 바로 왼쪽에 붙은 굵고 진한 아쿠아색 세로 강조선으로 판별합니다.
- 행 전체가 밝은지, 닉네임이나 기록이 노란색인지로 판별하지 마세요.
- 선택 세로선은 1~8 등수 숫자 중 정확히 하나의 왼쪽에 붙습니다.
- 왼쪽의 승리/패배 글자 배경색은 보조 단서일 뿐이며, 최종 기준은 등수 숫자 바로 왼쪽 세로 강조선입니다.
- 사전 판별 단계에서 선택된 등수는 ${selectedRank === null ? "확정되지 않음" : `${selectedRank}등`}입니다.
- 화면 유형 판별 결과는 ${screenType}입니다.
${selectedRank !== null ? `- 반드시 ${selectedRank}등 행의 완주 기록만 읽으세요. 다른 행 기록은 반환하지 마세요.` : "- 선택 등수가 확정되지 않았다면 임의로 1등을 선택하지 말고 기록을 반환하지 마세요."}
- 맵 이름은 화면 상단 왼쪽 제목 영역에서만 읽으세요.
- 카트 열, 차량 열, 평균 속도 열의 글자는 mapName 후보가 아닙니다.
- sourceType은 ranked_result입니다.

첨부 이미지 안내
- 첫 번째 이미지는 전체 원본입니다.
- 두 번째 이미지는 자동 탐지한 본인 행 후보 Crop일 수 있습니다.
- 세 번째 이미지는 등수 숫자와 그 왼쪽 세로선이 보이도록 왼쪽 영역을 확대한 안내 이미지일 수 있습니다.
- 보조 이미지보다 사전 판별된 selectedRank를 우선하세요.

기록 형식
- 01:53:77 → 1.53.77
- 00:58:68 → 0.58.68
- 03:20:75 → 3.20.75
- 미완료는 반환하지 마세요.

반환 전 자체 점검
1. mapName이 허용 맵 목록에 정확히 존재하는지 확인하세요.
2. mapName이 카트/차량 이름이 아닌지 확인하세요.
3. 랭전이면 selectedRank와 같은 행의 기록인지 확인하세요.
4. 프로필 기록 화면이면 화면에 보이는 모든 카드를 빠짐없이 읽었는지 확인하세요.

필수 반환 규칙
- sourceImage는 ${imageNumber}입니다.
- sourceType은 time_attack, ranked_result, unknown 중 하나입니다.
- evidence에는 선택 근거를 구체적으로 적으세요.
- 확실하지 않은 항목은 반환하지 말고 warning에 적으세요.
- 지정된 JSON 스키마 외 텍스트는 출력하지 마세요.

${retryMode ? `
재검토 모드
- 첫 판독에서 유효한 결과가 없었습니다.
- 차량 이름을 맵으로 잘못 읽지 않았는지 확인하세요.
- 랭전이면 selectedRank가 확정된 경우 그 행만 다시 확대해서 읽으세요.
- 프로필 기록 화면이면 카드 전체를 다시 훑으세요.
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

async function detectScreenAndSelectedRank({ env, image }) {
  const model = env.OPENAI_MODEL || "gpt-4.1-mini";
  const content = [
    {
      type: "input_text",
      text: `
이 이미지는 카트라이더 러쉬플러스 화면입니다.
먼저 화면 유형을 분류하고, 랭킹전 결과표이면 본인 등수를 판별하세요.

본인 등수 판별의 유일한 핵심 기준
- 1~8 등수 숫자 바로 왼쪽에 붙은 굵고 진한 아쿠아색 세로 강조선입니다.
- 행 전체 밝기, 노란 기록, 노란 닉네임, 노란 평균속도는 모두 무시하세요.
- 세로선이 숫자 3의 바로 왼쪽에 붙으면 selectedRank는 3입니다.
- 세로선이 숫자 7의 바로 왼쪽에 붙으면 selectedRank는 7입니다.
- 승리/패배 글자 뒤쪽의 아쿠아 배경은 보조 단서이며, 최종 기준은 등수 숫자 바로 왼쪽 세로선입니다.
- 확실하지 않으면 selectedRank는 null로 반환하세요. 임의로 1등을 선택하지 마세요.

screenType 분류
- 순위 1~8과 여러 선수 행이 보이면 ranked_result
- 여러 맵 카드와 기록이 격자로 보이면 profile_records
- 개인 단일 기록 화면이면 time_attack
- 그 외 unknown
      `.trim(),
    },
    {
      type: "input_image",
      image_url: `data:${image.mimeType};base64,${image.base64}`,
      detail: "high",
    },
  ];

  if (image.aquaGuideBase64) {
    content.push(
      {
        type: "input_text",
        text: "아래 확대 이미지는 등수 숫자와 그 바로 왼쪽의 아쿠아 세로 강조선을 비교하기 위한 것입니다. 숫자 왼쪽 세로선이 가장 굵고 진한 행의 등수를 선택하세요.",
      },
      {
        type: "input_image",
        image_url: `data:${image.aquaGuideMimeType || "image/jpeg"};base64,${image.aquaGuideBase64}`,
        detail: "high",
      },
    );
  }

  let response;
  try {
    response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        authorization: `Bearer ${env.OPENAI_API_KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        store: false,
        input: [{ role: "user", content }],
        text: {
          format: {
            type: "json_schema",
            name: "rank_selection",
            strict: true,
            schema: RANK_SELECTION_SCHEMA,
          },
        },
        max_output_tokens: 500,
      }),
    });
  } catch (error) {
    return {
      ok: false,
      screenType: "unknown",
      selectedRank: null,
      confidence: 0,
      evidence: String(error?.message || error || "선택 행 판별 실패"),
    };
  }

  const { responseText, parsed, requestId } = await readOpenAIResponse(response);
  if (!response.ok) {
    return {
      ok: false,
      screenType: "unknown",
      selectedRank: null,
      confidence: 0,
      evidence: parsed?.error?.message || responseText || "선택 행 판별 실패",
      requestId,
    };
  }

  const output = parseJsonText(extractOutputText(parsed));
  if (!output) {
    return {
      ok: false,
      screenType: "unknown",
      selectedRank: null,
      confidence: 0,
      evidence: "선택 행 판별 결과를 읽지 못함",
      requestId,
    };
  }

  return {
    ok: true,
    screenType: String(output.screenType || "unknown"),
    selectedRank: Number.isInteger(output.selectedRank) ? output.selectedRank : null,
    confidence: Number(output.confidence || 0),
    evidence: String(output.evidence || ""),
    requestId,
  };
}

async function callVisionForSingleImage({
  env,
  image,
  imageNumber,
  retryMode = false,
  selectedRank = null,
  screenType = "unknown",
  allowedMapNames = [],
}) {
  const model = env.OPENAI_MODEL || "gpt-4.1-mini";

  const inputContent = [
    {
      type: "input_text",
      text: buildVisionPrompt({
        imageNumber,
        retryMode,
        hasAquaCrop: Boolean(image.aquaCropBase64 || image.aquaGuideBase64),
        selectedRank,
        screenType,
        allowedMapNames,
      }),
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
        text: `아래 이미지는 브라우저 자동 탐지 본인 행 후보 Crop입니다. 탐지 신뢰도: ${Number(image.aquaCropConfidence || 0).toFixed(3)}. 사진 기울기에 따라 한 행 위·아래로 빗나갈 수 있으므로 왼쪽 확대 안내 이미지와 교차 확인하세요.`,
      },
      {
        type: "input_image",
        image_url: `data:${image.aquaCropMimeType || "image/jpeg"};base64,${image.aquaCropBase64}`,
        detail: "high",
      },
    );
  }

  if (image.aquaGuideBase64) {
    inputContent.push(
      {
        type: "input_text",
        text: "아래 이미지는 등수 숫자와 그 바로 왼쪽의 세로 강조선을 확대한 안내 이미지입니다. 행 배경이나 노란 글씨는 무시하고, 등수 숫자 바로 왼쪽에 붙은 굵고 진한 아쿠아 세로선만 확인하세요.",
      },
      {
        type: "input_image",
        image_url: `data:${image.aquaGuideMimeType || "image/jpeg"};base64,${image.aquaGuideBase64}`,
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
  const allowedMapNames = Array.isArray(body?.allowedMapNames)
    ? Array.from(new Set(body.allowedMapNames.map((name) => String(name || "").trim()).filter(Boolean))).slice(0, 300)
    : [];

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
    let aquaGuideBase64 = cleanBase64(image?.aquaGuideBase64 || "");
    let aquaGuideMimeType = sanitizeMimeType(image?.aquaGuideMimeType || "image/jpeg");

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

    if (!ALLOWED_IMAGE_TYPES.has(aquaGuideMimeType)) {
      aquaGuideBase64 = "";
      aquaGuideMimeType = "";
    }

    if (aquaGuideBase64 && aquaGuideBase64.length > MAX_BASE64_CHARS_PER_IMAGE) {
      aquaGuideBase64 = "";
      aquaGuideMimeType = "";
    }

    if (
      aquaGuideBase64 &&
      totalBase64Chars + base64.length + aquaCropBase64.length + aquaGuideBase64.length > MAX_TOTAL_BASE64_CHARS
    ) {
      aquaGuideBase64 = "";
      aquaGuideMimeType = "";
    }

    totalBase64Chars += base64.length + aquaCropBase64.length + aquaGuideBase64.length;
    acceptedImages.push({
      name,
      mimeType,
      base64,
      aquaCropBase64,
      aquaCropMimeType,
      aquaCropFound: Boolean(image?.aquaCropFound && aquaCropBase64),
      aquaCropConfidence: Number(image?.aquaCropConfidence || 0),
      aquaGuideBase64,
      aquaGuideMimeType,
      aquaGuideBox: image?.aquaGuideBox || null,
      aquaGuideVersion: String(image?.aquaGuideVersion || ""),
      aquaCenterYRatio: Number.isFinite(Number(image?.aquaCenterYRatio)) ? Number(image.aquaCenterYRatio) : null,
      aquaXRatio: Number.isFinite(Number(image?.aquaXRatio)) ? Number(image.aquaXRatio) : null,
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
    const selection = await detectScreenAndSelectedRank({ env, image });

    if (selection.requestId) {
      requestIds.push(selection.requestId);
    }

    if (
      selection.screenType === "ranked_result" &&
      selection.selectedRank === null
    ) {
      warnings.push(`${image.name}: 등수 숫자 바로 왼쪽의 아쿠아 세로 강조선을 확실히 찾지 못해 랭전 기록을 제외했습니다.`);
      continue;
    }

    let result = await callVisionForSingleImage({
      env,
      image,
      imageNumber: image.sourceImage,
      retryMode: false,
      selectedRank: selection.selectedRank,
      screenType: selection.screenType,
      allowedMapNames,
    });

    // 첫 판독에서 기록을 못 찾은 경우 화면 유형을 다시 판별하여 한 번 더 재검토합니다.
    if (result.ok && !result.records.length) {
      const retryResult = await callVisionForSingleImage({
        env,
        image,
        imageNumber: image.sourceImage,
        retryMode: true,
        selectedRank: selection.selectedRank,
        screenType: selection.screenType,
        allowedMapNames,
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
    aquaGuidesUsed: acceptedImages.filter((image) => image.aquaGuideBase64).length,
    allowedMapCount: allowedMapNames.length,
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
