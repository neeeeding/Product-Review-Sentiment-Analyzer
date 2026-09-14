"""상품 리뷰 감성 분석기 - Streamlit 화면.

실행:
    venv\\Scripts\\streamlit.exe run app.py

리뷰 텍스트(+ 선택적으로 실제 별점)를 입력하면 Gemini 에이전트가
도구 3개를 호출해 장단점 / 감성 점수 / 과장 리뷰 리포트를 만들어 준다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from src.agent import AgentError, analyze_review
from src.config import (
    MAX_REVIEW_CHARS,
    MIN_REVIEW_CHARS,
    MissingApiKeyError,
    get_model_name,
    has_api_key,
)
from src.gemini_client import GeminiCallError, RateLimitExceededError
from src.tools import DEFAULT_THRESHOLD

# 처음 써보는 사람이 바로 눌러볼 수 있도록 예시 리뷰를 넣어둔다.
SAMPLE_REVIEWS = {
    "직접 입력": "",
    "예시 1 · 장단점이 섞인 리뷰": (
        "배송은 주문한 다음 날 바로 와서 정말 빨랐어요. "
        "디자인도 사진이랑 똑같아서 마음에 듭니다. "
        "다만 포장이 너무 허술해서 상자 모서리가 찌그러져 있었고, "
        "생각보다 재질이 얇아서 오래 쓸 수 있을지는 모르겠네요. "
        "가격을 생각하면 그럭저럭 만족합니다."
    ),
    "예시 2 · 별점만 후한 리뷰": (
        "음... 솔직히 기대했던 것보다는 별로였어요. "
        "색상이 화면이랑 좀 다르고 마감도 거칠어요. "
        "그래도 판매자분이 친절하셔서 별 다섯 개 드립니다."
    ),
    "예시 3 · 만족 리뷰": (
        "세 번째 재구매입니다. 품질이 한결같이 좋아요. "
        "특히 세탁을 여러 번 해도 늘어나지 않는 게 마음에 듭니다. "
        "배송도 항상 빠르고 포장도 꼼꼼해요. 주변에 추천하고 있어요."
    ),
}


def main() -> None:
    st.set_page_config(
        page_title="상품 리뷰 감성 분석기",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 상품 리뷰 감성 분석기")
    st.caption(
        "쇼핑몰 리뷰를 붙여넣으면 Gemini 에이전트가 장단점을 정리하고, "
        "내용만으로 별점을 매긴 뒤, 실제 별점과 비교해 과장 여부를 알려줍니다."
    )

    threshold = _render_sidebar()

    # API 키가 없으면 입력 폼을 보여줘도 반드시 실패한다. 먼저 안내하고 멈춘다.
    if not has_api_key():
        _render_api_key_guide()
        return

    review_text, actual_rating = _render_input_form()

    if st.button("분석하기", type="primary", use_container_width=True):
        _run_analysis(review_text, actual_rating, threshold)

    # 분석 결과는 세션에 저장해 두었다가, 버튼을 다시 누르기 전까지 계속 보여준다.
    if "result" in st.session_state:
        _render_result(st.session_state["result"])


# ---------------------------------------------------------------------------
# 화면 조각들
# ---------------------------------------------------------------------------


def _render_sidebar() -> float:
    """왼쪽 설정 패널. 과장 판단 기준값을 돌려준다."""
    with st.sidebar:
        st.header("설정")

        threshold = st.slider(
            "과장 판단 기준 (점수 차이)",
            min_value=0.5,
            max_value=3.0,
            value=DEFAULT_THRESHOLD,
            step=0.5,
            help=(
                "실제 별점이 감성 점수보다 이 값 이상 높으면 '과장 의심'으로 "
                "판단합니다. 기획서 기본값은 1.5점입니다."
            ),
        )

        st.divider()
        st.subheader("상태")

        if has_api_key():
            st.success("API 키 연결됨")
        else:
            st.error("API 키 없음")

        st.caption(f"모델: `{get_model_name()}`")

        st.divider()
        st.caption(
            "무료 등급에는 분당/일일 요청 한도가 있습니다. "
            "연속으로 여러 번 분석하면 잠시 기다려야 할 수 있습니다."
        )

    return threshold


def _render_api_key_guide() -> None:
    """키가 없을 때 보여줄 설정 안내."""
    st.error("Gemini API 키가 설정되지 않아 분석을 시작할 수 없습니다.")

    st.markdown(
        """
        ### 설정 방법

        1. https://aistudio.google.com/ 에 접속해 로그인합니다.
        2. **Get API key → Create API key** 로 무료 키를 발급받습니다.
        3. 프로젝트 폴더의 `.env.example` 을 복사해 `.env` 로 이름을 바꿉니다.

           ```powershell
           Copy-Item .env.example .env
           ```

        4. `.env` 파일을 열어 실제 키를 붙여넣습니다.

           ```
           GOOGLE_API_KEY=여기에_발급받은_키
           ```

        5. 이 페이지를 새로고침합니다.

        `.env` 는 `.gitignore` 에 등록되어 있어 깃허브에 올라가지 않습니다.
        """
    )

    st.info(
        "설정을 마쳤다면 터미널에서 `venv\\Scripts\\python.exe "
        "scripts/check_connection.py` 로 연결을 먼저 확인해 볼 수 있습니다."
    )


def _render_input_form() -> tuple:
    """리뷰 입력 영역. (리뷰 원문, 실제 별점 또는 None) 을 돌려준다."""
    st.subheader("1. 리뷰 입력")

    sample_name = st.selectbox(
        "예시 리뷰로 먼저 시험해 볼 수 있습니다",
        options=list(SAMPLE_REVIEWS.keys()),
    )

    review_text = st.text_area(
        "리뷰 원문",
        value=SAMPLE_REVIEWS[sample_name],
        height=180,
        max_chars=MAX_REVIEW_CHARS,
        placeholder="쇼핑몰에서 복사한 리뷰를 여기에 붙여넣으세요.",
        label_visibility="collapsed",
    )

    char_count = len(review_text.strip())
    st.caption(
        f"{char_count} / {MAX_REVIEW_CHARS}자 "
        f"(최소 {MIN_REVIEW_CHARS}자 이상 입력해 주세요)"
    )

    st.subheader("2. 실제 별점 (선택)")

    use_rating = st.checkbox(
        "이 리뷰에 실제로 매겨진 별점도 입력할게요",
        value=True,
        help="입력하면 리뷰 내용과 별점의 괴리를 계산해 과장 여부를 알려줍니다.",
    )

    actual_rating: Optional[int] = None
    if use_rating:
        actual_rating = st.radio(
            "실제 별점",
            options=[1, 2, 3, 4, 5],
            index=4,
            horizontal=True,
            format_func=lambda n: "★" * n + "☆" * (5 - n) + f"  {n}점",
        )

    return review_text, actual_rating


def _run_analysis(
    review_text: str, actual_rating: Optional[int], threshold: float
) -> None:
    """분석을 실행하고 결과나 오류를 세션에 저장한다."""
    st.session_state.pop("result", None)

    try:
        with st.spinner("에이전트가 도구를 호출하며 분석 중입니다..."):
            result = analyze_review(
                review_text=review_text,
                actual_rating=actual_rating,
                threshold=threshold,
            )

    except ValueError as error:
        # 입력이 잘못된 경우 (너무 짧다/길다 등). 사용자가 고칠 수 있다.
        st.warning(str(error))

    except RateLimitExceededError as error:
        st.error(str(error))
        st.info("무료 등급 한도는 보통 1분 뒤에 다시 열립니다.")

    except MissingApiKeyError as error:
        st.error(str(error))

    except (GeminiCallError, AgentError) as error:
        st.error(str(error))

    else:
        st.session_state["result"] = result


def _render_result(result: Dict[str, Any]) -> None:
    """분석 결과를 카드/표 형태로 보여준다."""
    st.divider()
    st.subheader("분석 결과")

    score = result.get("sentiment_score")
    exaggeration = result.get("exaggeration")

    # --- 요약 지표 3개 ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "감성 점수 (내용 기준)",
            f"{score:g} / 5" if score is not None else "계산 안 됨",
        )

    with col2:
        if exaggeration:
            gap = exaggeration["gap"]
            st.metric(
                "실제 별점과의 차이",
                f"{gap:+g}점",
                delta=exaggeration["verdict"],
                delta_color="inverse" if exaggeration["is_exaggerated"] else "normal",
            )
        else:
            st.metric("실제 별점과의 차이", "비교 안 함")

    with col3:
        st.metric("소요 시간", f"{result.get('elapsed_sec', 0):g}초")

    # --- 과장 여부 리포트 ---
    if exaggeration:
        if exaggeration["is_exaggerated"]:
            st.error(f"**{exaggeration['verdict']}** — {exaggeration['explanation']}")
        elif exaggeration["is_understated"]:
            st.warning(f"**{exaggeration['verdict']}** — {exaggeration['explanation']}")
        else:
            st.success(f"**{exaggeration['verdict']}** — {exaggeration['explanation']}")

    # --- 장점 / 단점 ---
    st.markdown("#### 장점과 단점")
    pros_col, cons_col = st.columns(2)

    with pros_col:
        st.markdown("**👍 장점**")
        _render_bullet_list(result.get("pros"), empty_text="언급된 장점이 없습니다.")

    with cons_col:
        st.markdown("**👎 단점**")
        _render_bullet_list(result.get("cons"), empty_text="언급된 단점이 없습니다.")

    # --- 감성 점수 근거 ---
    reason = result.get("sentiment_reason")
    if reason:
        st.markdown("#### 감성 점수 근거")
        st.info(reason)

    # --- 에이전트 최종 리포트 ---
    summary = result.get("summary")
    if summary:
        st.markdown("#### 에이전트 리포트")
        st.write(summary)

    # --- 도구 호출 기록 ---
    _render_tool_trace(result.get("tool_trace") or [])


def _render_bullet_list(items: Optional[list], empty_text: str) -> None:
    """문자열 목록을 불릿으로 표시한다. 비어 있으면 안내 문구를 보여준다."""
    if not items:
        st.caption(empty_text)
        return

    for item in items:
        st.markdown(f"- {item}")


def _render_tool_trace(trace: list) -> None:
    """에이전트가 어떤 도구를 어떻게 호출했는지 펼쳐볼 수 있게 보여준다.

    "에이전트가 도구를 골라 호출했다"는 것을 눈으로 확인할 수 있는 부분이라
    과제 제출용 스크린샷에도 쓸모가 있다.
    """
    with st.expander(f"에이전트가 호출한 도구 ({len(trace)}건)"):
        if not trace:
            st.caption("호출된 도구가 없습니다.")
            return

        for index, record in enumerate(trace, start=1):
            status = "성공" if record["ok"] else "실패"
            st.markdown(f"**{index}. `{record['name']}` — {status}**")

            # 리뷰 원문이 통째로 인자로 들어가면 화면이 길어지므로 줄여서 보여준다.
            display_args = {
                key: (value[:60] + "..." if isinstance(value, str) and len(value) > 60 else value)
                for key, value in (record.get("args") or {}).items()
            }
            st.json(display_args, expanded=False)

            if record["ok"]:
                st.json(record["result"], expanded=False)
            else:
                st.error(record["error"])


if __name__ == "__main__":
    main()
