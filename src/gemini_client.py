"""Gemini API 호출 래퍼.

무료 등급에서는 분당/일일 요청 한도에 걸리는 일이 잦다. 그래서 모든 호출을
한 군데로 모으고, 한도 초과(429)나 일시적 서버 오류(5xx)일 때는
지수 백오프(exponential backoff)로 재시도한다.
기획서 6장의 "rate limit 대응" 요구사항에 해당한다.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, List, Optional, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import get_api_key, get_model_name

T = TypeVar("T")

# 재시도 설정
MAX_RETRIES = 4          # 최초 1회 + 재시도 3회
BASE_DELAY_SEC = 2.0     # 첫 재시도 대기 시간
MAX_DELAY_SEC = 30.0     # 대기 시간 상한

# 재시도해볼 만한 HTTP 상태 코드
#   429 = 요청 한도 초과 (무료 등급에서 가장 흔함)
#   500 = 서버 내부 오류, 503 = 일시적 과부하, 504 = 타임아웃
RETRYABLE_CODES = frozenset({429, 500, 503, 504})


class RateLimitExceededError(RuntimeError):
    """재시도를 다 쓰고도 요청 한도에 계속 걸린 경우."""


class GeminiCallError(RuntimeError):
    """재시도로 해결되지 않는 API 오류 (잘못된 키, 잘못된 모델 이름 등)."""


_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    """Gemini 클라이언트를 만들어 재사용한다.

    매 호출마다 새로 만들 필요가 없으므로 모듈 수준에 한 번만 캐시한다.
    """
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_api_key())
    return _client


def reset_client() -> None:
    """캐시된 클라이언트를 버린다. (.env 를 고치고 다시 시도할 때 사용)"""
    global _client
    _client = None


def generate(
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None,
    model: Optional[str] = None,
) -> types.GenerateContentResponse:
    """Gemini에 1회 요청한다. 한도 초과/일시 오류면 백오프 후 재시도한다.

    Args:
        contents: 프롬프트 문자열, 또는 대화 기록(Content 리스트).
        config: 생성 옵션 (도구 목록, JSON 스키마, 온도 등).
        model: 모델 이름. 생략하면 `.env` 설정값 또는 기본값.

    Returns:
        Gemini 응답 객체.

    Raises:
        RateLimitExceededError: 재시도를 다 써도 한도에 걸릴 때.
        GeminiCallError: 재시도로 해결되지 않는 오류일 때.
    """
    model_name = model or get_model_name()
    client = get_client()

    return _call_with_retry(
        lambda: client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
    )


def list_available_models() -> List[str]:
    """이 API 키로 쓸 수 있는 모델 이름 목록. (연결 확인 및 디버깅용)"""
    client = get_client()
    names: List[str] = []

    for model in _call_with_retry(lambda: list(client.models.list())):
        name = getattr(model, "name", None)
        if not name:
            continue
        # API는 "models/gemini-2.0-flash" 형태로 주므로 접두사를 떼어낸다.
        names.append(name.split("/", 1)[-1])

    return names


def _call_with_retry(fn: Callable[[], T]) -> T:
    """`fn` 을 실행하고, 재시도 가능한 오류면 지수 백오프로 다시 시도한다."""
    last_error: Optional[genai_errors.APIError] = None

    for attempt in range(MAX_RETRIES):
        try:
            return fn()

        except genai_errors.APIError as error:
            code = getattr(error, "code", None)

            # 키가 잘못됐거나 요청 자체가 잘못된 경우는 재시도해도 소용없다.
            if code not in RETRYABLE_CODES:
                raise GeminiCallError(_explain(error)) from error

            last_error = error
            is_last_attempt = attempt == MAX_RETRIES - 1
            if is_last_attempt:
                break

            time.sleep(_backoff_delay(attempt))

    # 여기까지 왔다면 재시도를 모두 소진했다.
    assert last_error is not None
    if getattr(last_error, "code", None) == 429:
        raise RateLimitExceededError(
            "Gemini 무료 등급 요청 한도에 걸렸습니다.\n"
            f"{MAX_RETRIES}번 시도했지만 계속 거부되었습니다.\n"
            "잠시(1분 정도) 기다렸다가 다시 시도해 주세요.\n"
            "분당 한도는 Google AI Studio에서 확인할 수 있습니다."
        ) from last_error

    raise GeminiCallError(_explain(last_error)) from last_error


def _backoff_delay(attempt: int) -> float:
    """지수 백오프 + 지터(jitter). 2초, 4초, 8초... 에 약간의 무작위성을 더한다.

    지터를 넣는 이유: 여러 요청이 동시에 실패했을 때 똑같은 시점에 몰려서
    재시도하면 또 같이 한도에 걸리기 때문이다.
    """
    delay = min(BASE_DELAY_SEC * (2 ** attempt), MAX_DELAY_SEC)
    return delay + random.uniform(0, delay * 0.25)


def _explain(error: genai_errors.APIError) -> str:
    """API 오류를 초보자가 읽고 조치할 수 있는 한국어 메시지로 바꾼다."""
    code = getattr(error, "code", None)
    message = getattr(error, "message", str(error))

    hints = {
        400: "요청 형식이 잘못되었습니다. 모델 이름이나 입력 내용을 확인해 주세요.",
        401: "API 키가 유효하지 않습니다. .env 의 GOOGLE_API_KEY 를 확인해 주세요.",
        403: (
            "API 키에 권한이 없습니다. 키가 올바른지, Google AI Studio에서 "
            "해당 키가 비활성화되지 않았는지 확인해 주세요."
        ),
        404: (
            "모델을 찾을 수 없습니다. .env 의 GEMINI_MODEL 이름을 확인해 주세요. "
            "사용 가능한 모델은 list_available_models() 로 볼 수 있습니다."
        ),
        429: "요청 한도를 초과했습니다. 잠시 기다렸다가 다시 시도해 주세요.",
    }

    hint = hints.get(code, "")
    return f"Gemini API 오류 (코드 {code}): {message}" + (f"\n→ {hint}" if hint else "")
