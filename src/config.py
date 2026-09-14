"""설정값과 API 키 로딩.

API 키는 코드에 절대 하드코딩하지 않고 `.env` 파일이나 환경변수에서 읽는다.
`.env` 는 `.gitignore` 에 등록되어 있어 깃에 올라가지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 프로젝트 루트 (이 파일의 부모의 부모)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# 기본 모델. 기획서에서 제시한 무료 등급 모델을 쓴다.
DEFAULT_MODEL = "gemini-2.0-flash"

# 무료 등급 토큰 제한을 고려한 입력 길이 가이드 (기획서 2장)
MAX_REVIEW_CHARS = 2000

# 분석하기에 너무 짧은 입력을 막는 최소 길이
MIN_REVIEW_CHARS = 5

# .env 파일에서 환경변수를 읽어들인다.
# override=False 이므로 이미 시스템 환경변수가 설정돼 있으면 그쪽이 우선한다.
load_dotenv(dotenv_path=ENV_PATH, override=False)


class MissingApiKeyError(RuntimeError):
    """API 키를 찾을 수 없을 때 발생. 사용자에게 해결 방법을 알려준다."""


def get_api_key() -> str:
    """환경변수에서 Gemini API 키를 읽어 온다.

    `GOOGLE_API_KEY` 를 먼저 찾고, 없으면 `GEMINI_API_KEY` 도 확인한다.

    Raises:
        MissingApiKeyError: 키가 없거나 템플릿 기본값 그대로일 때.
    """
    key = _read_env("GOOGLE_API_KEY") or _read_env("GEMINI_API_KEY")

    if not key:
        raise MissingApiKeyError(
            "Gemini API 키를 찾을 수 없습니다.\n"
            "해결 방법:\n"
            "  1. .env.example 파일을 복사해서 이름을 .env 로 바꾸세요.\n"
            "     PowerShell:  Copy-Item .env.example .env\n"
            "  2. .env 파일을 열고 GOOGLE_API_KEY= 뒤에 실제 키를 붙여넣으세요.\n"
            "  3. 키는 https://aistudio.google.com/ 에서 무료로 발급받을 수 있습니다.\n"
            f"  (확인한 .env 경로: {ENV_PATH})"
        )

    # .env.example 의 안내 문구를 그대로 둔 채 실행하는 실수를 잡아준다.
    if key.startswith("여기에") or key.strip().lower() in {"your_api_key", "changeme"}:
        raise MissingApiKeyError(
            ".env 파일의 GOOGLE_API_KEY 가 아직 안내 문구 그대로입니다.\n"
            "실제 발급받은 API 키로 바꿔주세요.\n"
            f"  (파일 위치: {ENV_PATH})"
        )

    return key


def get_model_name() -> str:
    """사용할 모델 이름. `.env` 의 GEMINI_MODEL 로 바꿀 수 있다."""
    return _read_env("GEMINI_MODEL") or DEFAULT_MODEL


def has_api_key() -> bool:
    """키가 준비됐는지 확인만 한다 (예외를 던지지 않음)."""
    try:
        get_api_key()
    except MissingApiKeyError:
        return False
    return True


def _read_env(name: str) -> Optional[str]:
    """환경변수를 읽고 앞뒤 공백과 따옴표를 제거한다."""
    value = os.environ.get(name)
    if value is None:
        return None

    # .env 에 GOOGLE_API_KEY="AIza..." 처럼 따옴표를 붙여 쓰는 경우가 흔하다.
    value = value.strip().strip('"').strip("'")
    return value or None
