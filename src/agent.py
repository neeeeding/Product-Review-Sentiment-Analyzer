"""리뷰 분석 에이전트 - Gemini Function Calling 오케스트레이션.

기획서 3장의 실행 흐름을 구현한다.

    사용자 입력(리뷰 + 선택적 실제 별점)
        -> Gemini가 필요한 도구를 스스로 골라 호출
        -> 도구 ①② 실행 (각각 내부적으로 Gemini 호출)
        -> 실제 별점이 있으면 도구 ③ 실행 (계산만)
        -> Gemini가 결과를 종합해 최종 리포트 작성

SDK의 자동 함수 호출(automatic function calling) 기능을 일부러 끄고
직접 루프를 돌린다. 그래야 "에이전트가 어떤 도구를 어떤 인자로 불렀는지"를
기록해서 화면에 보여줄 수 있다.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from google.genai import types

from src.config import MAX_REVIEW_CHARS, MIN_REVIEW_CHARS
from src.gemini_client import generate
from src.tools import (
    DEFAULT_THRESHOLD,
    classify_review_text,
    detect_review_exaggeration,
    score_sentiment,
)

# 도구 호출 루프의 최대 반복 횟수.
# 모델이 같은 도구를 계속 부르며 무한 루프에 빠지는 것을 막는다.
MAX_TURNS = 6


class AgentError(RuntimeError):
    """에이전트 실행이 정상적으로 끝나지 못한 경우."""


# ---------------------------------------------------------------------------
# 도구 스키마 등록 (기획서 3장)
# ---------------------------------------------------------------------------

_TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="classify_review_text",
        description=(
            "리뷰 원문을 읽고 장점 문장과 단점 문장으로 분류한다. "
            "리뷰 분석을 시작할 때 가장 먼저 호출해야 한다."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "review_text": types.Schema(
                    type=types.Type.STRING,
                    description="분석할 리뷰 원문 전체.",
                )
            },
            required=["review_text"],
        ),
    ),
    types.FunctionDeclaration(
        name="score_sentiment",
        description=(
            "리뷰 내용만 보고 이 리뷰어의 만족도를 1~5점 별점으로 환산한다. "
            "사용자가 알려준 실제 별점은 절대 참고하지 않는다."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "review_text": types.Schema(
                    type=types.Type.STRING,
                    description="분석할 리뷰 원문 전체.",
                )
            },
            required=["review_text"],
        ),
    ),
    types.FunctionDeclaration(
        name="detect_review_exaggeration",
        description=(
            "사용자가 입력한 실제 별점과 score_sentiment가 추정한 감성 점수를 "
            "비교해 과장 리뷰 여부를 판단한다. "
            "사용자가 실제 별점을 알려준 경우에만 호출한다. "
            "반드시 score_sentiment를 먼저 실행한 뒤 그 결과 점수를 넣어야 한다."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "actual_rating": types.Schema(
                    type=types.Type.NUMBER,
                    description="사용자가 입력한 실제 별점 (1~5).",
                ),
                "sentiment_score": types.Schema(
                    type=types.Type.NUMBER,
                    description="score_sentiment 도구가 반환한 감성 점수 (1~5).",
                ),
            },
            required=["actual_rating", "sentiment_score"],
        ),
    ),
]

# 이름 -> 실제 파이썬 함수 연결표
_TOOL_FUNCTIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "classify_review_text": classify_review_text,
    "score_sentiment": score_sentiment,
    "detect_review_exaggeration": detect_review_exaggeration,
}

_SYSTEM_PROMPT = """\
너는 쇼핑몰 리뷰를 분석하는 에이전트다. 등록된 도구를 사용해 분석을 수행한다.

작업 순서:
1. classify_review_text 로 리뷰를 장점/단점으로 나눈다.
2. score_sentiment 로 리뷰 내용만 보고 별점을 추정한다.
3. 사용자가 실제 별점을 알려준 경우에만, detect_review_exaggeration 에
   그 별점과 2번에서 받은 감성 점수를 넣어 괴리를 계산한다.
   사용자가 별점을 알려주지 않았다면 3번은 건너뛴다.

도구 실행이 끝나면 결과를 종합해 한국어로 최종 리포트를 작성해라.
리포트는 3~5문장으로 짧게 쓰고, 다음을 담는다.
- 이 리뷰가 전체적으로 어떤 평가인지
- 감성 점수가 몇 점이고 왜 그런지
- (별점 비교를 했다면) 과장 여부와 그 이유

도구가 준 숫자와 문장을 그대로 근거로 삼고, 없는 내용을 지어내지 마라.
"""


def analyze_review(
    review_text: str,
    actual_rating: Optional[float] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """리뷰를 분석한다. Gemini가 도구를 골라 호출하는 전체 흐름을 실행한다.

    Args:
        review_text: 분석할 리뷰 원문.
        actual_rating: 사용자가 입력한 실제 별점 (1~5). 없으면 None.
        threshold: 과장 판단 기준 점수.

    Returns:
        아래 키를 가진 dict.
            pros              (list[str])  장점 목록
            cons              (list[str])  단점 목록
            sentiment_score   (float|None) 감성 점수
            sentiment_reason  (str|None)   감성 점수 근거
            exaggeration      (dict|None)  도구 ③ 결과
            summary           (str)        에이전트의 최종 리포트
            tool_trace        (list[dict]) 어떤 도구를 어떻게 불렀는지 기록
            elapsed_sec       (float)      총 소요 시간

    Raises:
        ValueError: 입력이 유효하지 않을 때.
        AgentError: 도구 호출 루프가 정상적으로 끝나지 않았을 때.
    """
    review_text = _validate_review(review_text)
    actual_rating = _validate_rating(actual_rating)

    started_at = time.monotonic()

    contents: List[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=_build_user_prompt(review_text, actual_rating))],
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=_TOOL_DECLARATIONS)],
        # SDK가 알아서 함수를 호출해버리면 호출 기록을 남길 수 없다. 직접 처리한다.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )

    trace: List[Dict[str, Any]] = []
    summary = ""

    for _ in range(MAX_TURNS):
        response = generate(contents=contents, config=config)

        calls = response.function_calls
        if not calls:
            # 도구를 더 부르지 않는다 = 최종 답변을 내놓은 것이다.
            summary = (response.text or "").strip()
            break

        # 모델이 만든 함수 호출 요청을 대화 기록에 그대로 남긴다.
        contents.append(response.candidates[0].content)

        response_parts: List[types.Part] = []
        for call in calls:
            record = _execute_tool(call, threshold=threshold)
            trace.append(record)
            response_parts.append(
                types.Part.from_function_response(
                    name=record["name"],
                    response=record["response"],
                )
            )

        contents.append(types.Content(role="user", parts=response_parts))

    else:
        # for 루프가 break 없이 끝났다 = MAX_TURNS를 다 썼다.
        raise AgentError(
            f"에이전트가 {MAX_TURNS}번 안에 분석을 끝내지 못했습니다. "
            "리뷰를 더 짧게 줄여서 다시 시도해 주세요."
        )

    result = _collect_results(trace)
    result["summary"] = summary
    result["tool_trace"] = trace
    result["elapsed_sec"] = round(time.monotonic() - started_at, 2)

    return result


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _build_user_prompt(review_text: str, actual_rating: Optional[float]) -> str:
    """에이전트에게 보낼 첫 메시지를 만든다."""
    lines = ["다음 쇼핑몰 리뷰를 분석해줘.", "", "[리뷰 원문]", review_text, ""]

    if actual_rating is None:
        lines.append(
            "사용자가 실제 별점을 알려주지 않았다. "
            "detect_review_exaggeration 는 호출하지 마라."
        )
    else:
        lines.append(f"사용자가 이 상품에 실제로 매긴 별점은 {actual_rating:g}점이다.")

    return "\n".join(lines)


def _execute_tool(call: types.FunctionCall, threshold: float) -> Dict[str, Any]:
    """모델이 요청한 도구 하나를 실제로 실행하고 기록을 남긴다.

    도구가 실패해도 예외를 밖으로 던지지 않는다. 대신 오류 내용을 모델에게
    돌려줘서 모델이 상황을 설명하거나 다시 시도할 수 있게 한다.
    """
    name = call.name or ""
    args: Dict[str, Any] = dict(call.args or {})

    record: Dict[str, Any] = {
        "name": name,
        "args": args,
        "ok": False,
        "result": None,
        "error": None,
        "response": {},
    }

    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        record["error"] = f"등록되지 않은 도구입니다: {name}"
        record["response"] = {"error": record["error"]}
        return record

    # 임계값은 모델이 정하는 값이 아니라 앱 설정값이므로 여기서 주입한다.
    if name == "detect_review_exaggeration":
        args["threshold"] = threshold

    try:
        result = function(**args)
    except TypeError as error:
        # 모델이 인자 이름을 잘못 줬을 때
        record["error"] = f"{name} 호출 인자가 잘못되었습니다: {error}"
    except ValueError as error:
        record["error"] = str(error)
    else:
        record["ok"] = True
        record["result"] = result
        record["response"] = result
        return record

    record["response"] = {"error": record["error"]}
    return record


def _collect_results(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """도구 호출 기록에서 화면에 보여줄 값들을 뽑아낸다.

    같은 도구가 여러 번 호출됐다면 마지막으로 성공한 결과를 쓴다.
    """
    collected: Dict[str, Any] = {
        "pros": [],
        "cons": [],
        "sentiment_score": None,
        "sentiment_reason": None,
        "exaggeration": None,
    }

    for record in trace:
        if not record["ok"]:
            continue

        name = record["name"]
        result = record["result"]

        if name == "classify_review_text":
            collected["pros"] = result.get("pros", [])
            collected["cons"] = result.get("cons", [])
        elif name == "score_sentiment":
            collected["sentiment_score"] = result.get("score")
            collected["sentiment_reason"] = result.get("reason")
        elif name == "detect_review_exaggeration":
            collected["exaggeration"] = result

    return collected


def _validate_review(review_text: Any) -> str:
    """리뷰 원문 검증. 도구와 같은 기준을 앱 입구에서 먼저 적용한다."""
    if not isinstance(review_text, str):
        raise ValueError("리뷰는 문자열이어야 합니다.")

    text = review_text.strip()

    if len(text) < MIN_REVIEW_CHARS:
        raise ValueError(
            f"리뷰가 너무 짧습니다. 최소 {MIN_REVIEW_CHARS}자 이상 입력해 주세요. "
            f"(현재 {len(text)}자)"
        )

    if len(text) > MAX_REVIEW_CHARS:
        raise ValueError(
            f"리뷰가 너무 깁니다. {MAX_REVIEW_CHARS}자 이내로 줄여 주세요. "
            f"(현재 {len(text)}자)"
        )

    return text


def _validate_rating(actual_rating: Any) -> Optional[float]:
    """실제 별점 검증. None은 '입력하지 않음'을 뜻하므로 그대로 통과시킨다."""
    if actual_rating is None:
        return None

    try:
        rating = float(actual_rating)
    except (TypeError, ValueError):
        raise ValueError(
            f"실제 별점은 숫자여야 합니다. 받은 값: {actual_rating!r}"
        ) from None

    if rating != rating:  # NaN
        raise ValueError("실제 별점에 NaN은 쓸 수 없습니다.")

    if not (1.0 <= rating <= 5.0):
        raise ValueError(f"실제 별점은 1~5 사이여야 합니다. 받은 값: {rating:g}")

    return rating
