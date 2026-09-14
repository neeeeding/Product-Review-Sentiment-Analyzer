"""에이전트가 호출하는 도구(Tool) 모음.

기획서 3장에 정의된 도구 3개를 구현한다.

    (1) classify_review_text        - 리뷰를 장점/단점으로 분류   (LLM 호출)
    (2) score_sentiment             - 리뷰 내용만으로 별점 추정   (LLM 호출)
    (3) detect_review_exaggeration  - 별점 괴리 계산              (순수 계산)

(3)번은 LLM을 쓰지 않는 순수 파이썬 계산이다. 무료 등급 호출 횟수를
아끼기 위한 의도적인 설계이며, 그래서 API 키 없이도 테스트할 수 있다.
"""

from __future__ import annotations

from typing import Any, Dict

# 실제 별점이 감성 점수보다 이만큼 이상 높으면 "과장 의심"으로 판단한다.
# 기획서에서 제시한 예시 값 1.5점을 기본값으로 쓴다.
DEFAULT_THRESHOLD = 1.5

# 별점/감성 점수가 가질 수 있는 범위
MIN_SCORE = 1.0
MAX_SCORE = 5.0


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
