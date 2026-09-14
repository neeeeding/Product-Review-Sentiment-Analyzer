"""에이전트가 호출하는 도구(Tool) 모음.

기획서 3장에 정의된 도구 3개를 구현한다.

    (1) classify_review_text        - 리뷰를 장점/단점으로 분류   (LLM 호출)
    (2) score_sentiment             - 리뷰 내용만으로 별점 추정   (LLM 호출)
    (3) detect_review_exaggeration  - 별점 괴리 계산              (순수 계산)

(3)번은 LLM을 쓰지 않는 순수 파이썬 계산이다. 무료 등급 호출 횟수를
아끼기 위한 의도적인 설계이며, 그래서 API 키 없이도 테스트할 수 있다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from google.genai import types

from src.config import MAX_REVIEW_CHARS, MIN_REVIEW_CHARS
from src.gemini_client import generate

# 실제 별점이 감성 점수보다 이만큼 이상 높으면 "과장 의심"으로 판단한다.
# 기획서에서 제시한 예시 값 1.5점을 기본값으로 쓴다.
DEFAULT_THRESHOLD = 1.5

# 별점/감성 점수가 가질 수 있는 범위
MIN_SCORE = 1.0
MAX_SCORE = 5.0


# ---------------------------------------------------------------------------
# 도구 ① 리뷰 텍스트 분류
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """\
너는 쇼핑몰 리뷰를 분석하는 도우미다.
주어진 리뷰에서 장점과 단점을 찾아 각각 짧은 문장으로 정리해라.

규칙:
- 리뷰에 실제로 적힌 내용만 쓴다. 없는 내용을 지어내지 않는다.
- 한 항목은 한 가지 내용만 담고, 30자 이내로 짧게 쓴다.
- 장점만 있거나 단점만 있는 리뷰라면 해당 목록은 빈 배열로 둔다.
- 배송/포장처럼 상품 자체와 무관한 내용도 리뷰어가 언급했다면 포함한다.
- 반드시 한국어로 답한다.
"""

_CLASSIFY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "pros": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="리뷰에서 찾은 장점 목록. 없으면 빈 배열.",
        ),
        "cons": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="리뷰에서 찾은 단점 목록. 없으면 빈 배열.",
        ),
    },
    required=["pros", "cons"],
)


def classify_review_text(review_text: str) -> Dict[str, Any]:
    """리뷰 원문을 장점/단점 문장으로 분류한다. (Gemini 호출)

    Args:
        review_text: 분석할 리뷰 원문.

    Returns:
        {"pros": [str, ...], "cons": [str, ...]}

    Raises:
        ValueError: 리뷰가 비었거나 길이 제한을 벗어날 때.
    """
    review_text = _validate_review_text(review_text)

    response = generate(
        contents=f"다음 리뷰를 분석해라.\n\n---\n{review_text}\n---",
        config=types.GenerateContentConfig(
            system_instruction=_CLASSIFY_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_CLASSIFY_SCHEMA,
            # 분류는 창의성이 필요 없다. 낮출수록 같은 입력에 같은 결과가 나온다.
            temperature=0.1,
        ),
    )

    data = _parse_json_response(response.text, tool_name="classify_review_text")

    return {
        "pros": _clean_string_list(data.get("pros")),
        "cons": _clean_string_list(data.get("cons")),
    }


# ---------------------------------------------------------------------------
# 도구 ② 감성 점수화
# ---------------------------------------------------------------------------

_SCORE_SYSTEM_PROMPT = """\
너는 쇼핑몰 리뷰를 읽고 만족도를 점수로 환산하는 도우미다.
리뷰에 적힌 "내용만" 보고, 이 리뷰어가 별점을 매긴다면 몇 점일지 추정해라.

규칙:
- 점수는 1.0 ~ 5.0 사이의 숫자다. 0.5 단위로 매겨도 된다.
  1점 = 매우 불만족, 3점 = 보통, 5점 = 매우 만족
- 리뷰에 이미 별점이 몇 점이라고 적혀 있어도 무시하고, 내용만으로 판단해라.
- 근거는 리뷰에서 근거가 된 표현을 인용하며 두 문장 이내로 한국어로 쓴다.
- 칭찬과 불만이 섞여 있으면 어느 쪽이 더 강한지로 판단한다.
"""

_SCORE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "score": types.Schema(
            type=types.Type.NUMBER,
            description="리뷰 내용만으로 추정한 별점 (1.0~5.0).",
        ),
        "reason": types.Schema(
            type=types.Type.STRING,
            description="그 점수를 준 근거를 두 문장 이내로 설명.",
        ),
    },
    required=["score", "reason"],
)


def score_sentiment(review_text: str) -> Dict[str, Any]:
    """리뷰 내용만 보고 감성 점수(1~5)를 추정한다. (Gemini 호출)

    Args:
        review_text: 분석할 리뷰 원문.

    Returns:
        {"score": float, "reason": str}

    Raises:
        ValueError: 리뷰가 비었거나 길이 제한을 벗어날 때.
    """
    review_text = _validate_review_text(review_text)

    response = generate(
        contents=f"다음 리뷰의 만족도를 점수로 매겨라.\n\n---\n{review_text}\n---",
        config=types.GenerateContentConfig(
            system_instruction=_SCORE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_SCORE_SCHEMA,
            temperature=0.1,
        ),
    )

    data = _parse_json_response(response.text, tool_name="score_sentiment")

    # 모델이 범위를 벗어난 점수를 주더라도 1~5 안으로 잘라서 돌려준다.
    # 도구 ③이 1~5 범위를 요구하므로 여기서 막아두지 않으면 뒤에서 터진다.
    raw_score = data.get("score")
    score = _clamp_score(raw_score)

    reason = str(data.get("reason") or "").strip()

    return {
        "score": score,
        "reason": reason or "근거 설명을 받지 못했습니다.",
    }


# ---------------------------------------------------------------------------
# 도구 ③ 과장 리뷰 탐지 (LLM 호출 없음)
# ---------------------------------------------------------------------------


def detect_review_exaggeration(
    actual_rating: float,
    sentiment_score: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """사용자가 준 실제 별점과 리뷰 내용 기반 감성 점수의 괴리를 계산한다.

    LLM을 호출하지 않는 순수 계산 함수다.

    Args:
        actual_rating: 사용자가 실제로 매긴 별점 (1~5).
        sentiment_score: 도구 2번이 리뷰 내용만 보고 추정한 점수 (1~5).
        threshold: 이 값 이상 벌어지면 괴리가 있다고 판단할 기준 점수.

    Returns:
        아래 키를 가진 dict.
            gap             (float) 실제별점 - 감성점수. 양수면 별점이 더 후하다.
            abs_gap         (float) 괴리의 크기 (부호 없음).
            is_exaggerated  (bool)  별점이 내용보다 threshold 이상 후한가.
            is_understated  (bool)  별점이 내용보다 threshold 이상 박한가.
            verdict         (str)   "과장 의심" / "과소평가 의심" / "대체로 일치".
            explanation     (str)   사람이 읽을 수 있는 설명 문장.
            threshold       (float) 판단에 사용한 기준값.

    Raises:
        ValueError: 점수가 1~5 범위를 벗어나거나 threshold가 0 이하일 때.
    """
    actual_rating = _validate_score(actual_rating, "actual_rating")
    sentiment_score = _validate_score(sentiment_score, "sentiment_score")

    if threshold <= 0:
        raise ValueError(f"threshold는 0보다 커야 합니다. 받은 값: {threshold}")

    # 양수 = 별점이 리뷰 내용보다 후하다 (과장 방향)
    # 음수 = 별점이 리뷰 내용보다 박하다 (과소평가 방향)
    gap = round(actual_rating - sentiment_score, 2)
    abs_gap = abs(gap)

    is_exaggerated = gap >= threshold
    is_understated = gap <= -threshold

    if is_exaggerated:
        verdict = "과장 의심"
        explanation = (
            f"실제 별점 {actual_rating:g}점이 리뷰 내용에서 읽히는 만족도"
            f"({sentiment_score:g}점)보다 {abs_gap:g}점 높습니다. "
            f"기준치 {threshold:g}점 이상 벌어졌으므로 내용에 비해 별점을 후하게 준 "
            f"과장 리뷰일 가능성이 있습니다."
        )
    elif is_understated:
        verdict = "과소평가 의심"
        explanation = (
            f"실제 별점 {actual_rating:g}점이 리뷰 내용에서 읽히는 만족도"
            f"({sentiment_score:g}점)보다 {abs_gap:g}점 낮습니다. "
            f"내용은 긍정적인데 별점만 박하게 준 경우로 보입니다."
        )
    else:
        verdict = "대체로 일치"
        explanation = (
            f"실제 별점 {actual_rating:g}점과 리뷰 내용 기반 감성 점수"
            f"({sentiment_score:g}점)의 차이가 {abs_gap:g}점으로, "
            f"기준치 {threshold:g}점 미만입니다. 별점과 내용이 대체로 일치합니다."
        )

    return {
        "gap": gap,
        "abs_gap": abs_gap,
        "is_exaggerated": is_exaggerated,
        "is_understated": is_understated,
        "verdict": verdict,
        "explanation": explanation,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------------------------

# ```json ... ``` 형태의 마크다운 코드 블록을 벗겨내기 위한 패턴.
# response_schema를 쓰면 보통 순수 JSON이 오지만, 모델이 감싸서 줄 때가 있다.
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _validate_review_text(review_text: Any) -> str:
    """리뷰 원문이 분석 가능한 상태인지 확인하고 앞뒤 공백을 정리한다."""
    if not isinstance(review_text, str):
        raise ValueError(
            f"리뷰는 문자열이어야 합니다. 받은 타입: {type(review_text).__name__}"
        )

    text = review_text.strip()

    if len(text) < MIN_REVIEW_CHARS:
        raise ValueError(
            f"리뷰가 너무 짧습니다. 최소 {MIN_REVIEW_CHARS}자 이상 입력해 주세요. "
            f"(현재 {len(text)}자)"
        )

    if len(text) > MAX_REVIEW_CHARS:
        raise ValueError(
            f"리뷰가 너무 깁니다. {MAX_REVIEW_CHARS}자 이내로 줄여 주세요. "
            f"(현재 {len(text)}자) 무료 등급 토큰 제한 때문입니다."
        )

    return text


def _parse_json_response(raw_text: Any, tool_name: str) -> Dict[str, Any]:
    """Gemini 응답 텍스트를 dict로 파싱한다.

    response_schema를 지정했으므로 대부분 순수 JSON이 오지만,
    안전망으로 코드 블록 표기를 벗겨내고 파싱 실패를 친절한 오류로 바꾼다.
    """
    if not raw_text:
        raise ValueError(
            f"{tool_name}: Gemini가 빈 응답을 반환했습니다. "
            "리뷰 내용이 안전 필터에 걸렸을 수 있습니다. 다시 시도해 주세요."
        )

    text = str(raw_text).strip()

    fenced = _CODE_FENCE.match(text)
    if fenced:
        text = fenced.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{tool_name}: Gemini 응답을 JSON으로 읽지 못했습니다.\n"
            f"응답 앞부분: {text[:200]}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"{tool_name}: JSON 객체를 기대했지만 {type(data).__name__}을 받았습니다."
        )

    return data


def _clean_string_list(value: Any) -> List[str]:
    """모델이 준 목록을 문자열 리스트로 정리한다.

    None, 빈 문자열, 중복을 걸러내고 순서는 유지한다.
    """
    if value is None:
        return []

    if isinstance(value, str):
        # 모델이 리스트 대신 문자열 하나를 줬을 때의 안전망
        value = [value]

    if not isinstance(value, (list, tuple)):
        return []

    cleaned: List[str] = []
    seen = set()

    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)

    return cleaned


def _clamp_score(value: Any) -> float:
    """모델이 준 점수를 1.0~5.0 범위로 자른다."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"score_sentiment: 점수를 숫자로 읽지 못했습니다. 받은 값: {value!r}"
        ) from None

    if score != score:  # NaN
        raise ValueError("score_sentiment: 점수가 NaN입니다.")

    clamped = max(MIN_SCORE, min(MAX_SCORE, score))
    return round(clamped, 2)


def _validate_score(value: Any, field_name: str) -> float:
    """점수 입력값을 float으로 바꾸고 1~5 범위인지 확인한다."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field_name}는 숫자여야 합니다. 받은 값: {value!r}"
        ) from None

    # NaN은 어떤 비교에도 False를 반환해서 범위 검사를 통과해버리므로 따로 막는다.
    if score != score:
        raise ValueError(f"{field_name}에 NaN은 쓸 수 없습니다.")

    if not (MIN_SCORE <= score <= MAX_SCORE):
        raise ValueError(
            f"{field_name}는 {MIN_SCORE:g}~{MAX_SCORE:g} 사이여야 합니다. "
            f"받은 값: {score:g}"
        )

    return score
