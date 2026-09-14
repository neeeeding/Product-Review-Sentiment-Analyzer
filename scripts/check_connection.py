"""Gemini API 연결 점검 스크립트.

앱을 실행하기 전에 API 키가 제대로 설정됐는지, 어떤 모델을 쓸 수 있는지
확인한다. 실제 분석 요청은 보내지 않으므로 무료 등급 한도를 거의 쓰지 않는다.

실행:
    venv\\Scripts\\python.exe scripts/check_connection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 이 스크립트를 프로젝트 루트 밖에서 실행해도 src 를 찾을 수 있게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    ENV_PATH,
    MissingApiKeyError,
    get_api_key,
    get_model_name,
)
from src.gemini_client import (  # noqa: E402
    GeminiCallError,
    RateLimitExceededError,
    list_available_models,
)


def main() -> int:
    print("=" * 60)
    print(" Gemini API 연결 점검")
    print("=" * 60)

    # 1. API 키 확인
    print("\n[1/3] API 키 확인")
    try:
        key = get_api_key()
    except MissingApiKeyError as error:
        print("  실패\n")
        print(error)
        return 1

    # 키 전체를 출력하면 화면 공유나 스크린샷으로 유출된다. 앞뒤만 보여준다.
    print(f"  성공 - 키를 찾았습니다: {_mask(key)}")
    print(f"  (읽은 파일: {ENV_PATH})")

    # 2. 사용 가능한 모델 조회 (실제 네트워크 요청)
    print("\n[2/3] 사용 가능한 모델 조회")
    try:
        models = list_available_models()
    except RateLimitExceededError as error:
        print(f"  실패 - {error}")
        return 1
    except GeminiCallError as error:
        print(f"  실패 - {error}")
        return 1

    generate_models = sorted(m for m in models if m.startswith("gemini"))
    print(f"  성공 - 총 {len(models)}개 모델 응답, gemini 계열 {len(generate_models)}개")
    for name in generate_models[:15]:
        print(f"    - {name}")
    if len(generate_models) > 15:
        print(f"    ... 외 {len(generate_models) - 15}개")

    # 3. 설정된 모델이 실제로 목록에 있는지 확인
    print("\n[3/3] 설정된 모델 사용 가능 여부")
    configured = get_model_name()
    if configured in models:
        print(f"  성공 - '{configured}' 사용 가능")
    else:
        print(f"  경고 - '{configured}' 을(를) 목록에서 찾지 못했습니다.")
        print("  .env 의 GEMINI_MODEL 을 위 목록의 이름 중 하나로 바꿔 주세요.")
        return 1

    print("\n" + "=" * 60)
    print(" 모든 점검을 통과했습니다. 앱을 실행할 수 있습니다.")
    print("   venv\\Scripts\\streamlit.exe run app.py")
    print("=" * 60)
    return 0


def _mask(key: str) -> str:
    """키를 화면에 안전하게 표시한다. 앞 6자와 뒤 4자만 남긴다."""
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:6]}{'*' * 10}{key[-4:]}"


if __name__ == "__main__":
    sys.exit(main())
