"""Streamlit 화면 테스트.

streamlit 공식 테스트 도구(AppTest)로 app.py 를 실제로 실행해서
렌더링 중 예외가 나지 않는지, 상황별로 맞는 화면이 뜨는지 확인한다.
브라우저 없이 돌아가고 실제 Gemini도 호출하지 않는다.
"""

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = "app.py"
RUN_TIMEOUT = 30


def patch_analyze(monkeypatch, replacement):
    """app.py 가 쓰는 analyze_review 를 바꿔치기한다.

    AppTest 는 app.py 를 별도 네임스페이스로 매번 다시 실행하므로,
    이미 임포트된 `app` 모듈을 패치해도 소용이 없다.
    app.py 가 `from src.agent import analyze_review` 로 가져가는
    원본 쪽을 패치해야 다시 실행될 때 바뀐 함수가 잡힌다.
    """
    monkeypatch.setattr("src.agent.analyze_review", replacement)


def run_app(monkeypatch, *, with_key: bool):
    """app.py 를 실행한다. with_key=False 면 API 키가 없는 상태를 흉내낸다."""
    if with_key:
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-테스트용-가짜키")
    else:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    app = AppTest.from_file(APP_PATH, default_timeout=RUN_TIMEOUT)
    app.run()
    return app


class TestNoApiKey:
    """키가 없을 때는 분석 폼 대신 설정 안내가 떠야 한다."""

    def test_예외_없이_렌더링된다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=False)

        assert not app.exception

    def test_키_설정_안내를_보여준다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=False)

        errors = " ".join(e.value for e in app.error)
        assert "API 키" in errors

    def test_발급_주소를_알려준다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=False)

        page_text = " ".join(m.value for m in app.markdown)
        assert "aistudio.google.com" in page_text

    def test_분석_버튼을_보여주지_않는다(self, monkeypatch):
        # 눌러봐야 실패할 버튼을 띄우지 않는다.
        app = run_app(monkeypatch, with_key=False)

        assert len(app.button) == 0


class TestWithApiKey:
    """키가 있을 때는 입력 폼이 떠야 한다."""

    def test_예외_없이_렌더링된다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        assert not app.exception

    def test_리뷰_입력창이_있다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        assert len(app.text_area) == 1

    def test_분석_버튼이_있다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        assert any("분석" in b.label for b in app.button)

    def test_별점_선택이_기본으로_켜져있다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        assert app.checkbox[0].value is True
        assert len(app.radio) == 1  # 별점 라디오가 보인다

    def test_별점_체크를_끄면_별점_선택이_사라진다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        app.checkbox[0].set_value(False).run()

        assert not app.exception
        assert len(app.radio) == 0

    def test_예시_리뷰를_고를_수_있다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        assert len(app.selectbox) == 1
        assert len(app.selectbox[0].options) >= 2

    def test_예시_리뷰를_고르면_입력창에_채워진다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        app.selectbox[0].set_value("예시 1 · 장단점이 섞인 리뷰").run()

        assert not app.exception
        assert "배송" in app.text_area[0].value


class TestSidebar:
    def test_임계값_슬라이더_기본값은_1점5다(self, monkeypatch):
        from src.tools import DEFAULT_THRESHOLD

        app = run_app(monkeypatch, with_key=True)

        assert app.slider[0].value == DEFAULT_THRESHOLD

    def test_임계값을_바꿔도_예외가_없다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        app.slider[0].set_value(2.5).run()

        assert not app.exception
        assert app.slider[0].value == 2.5


class TestAnalysisErrors:
    """분석이 실패했을 때 사용자가 읽을 수 있는 메시지가 떠야 한다."""

    def test_너무_짧은_리뷰는_경고를_띄운다(self, monkeypatch):
        app = run_app(monkeypatch, with_key=True)

        app.text_area[0].set_value("짧음")
        app.button[0].click().run()

        assert not app.exception
        warnings = " ".join(w.value for w in app.warning)
        assert "짧습니다" in warnings

    def test_한도_초과는_오류로_안내한다(self, monkeypatch):
        from src.gemini_client import RateLimitExceededError

        def raise_rate_limit(**kwargs):
            raise RateLimitExceededError("요청 한도에 걸렸습니다.")

        patch_analyze(monkeypatch, raise_rate_limit)

        app = run_app(monkeypatch, with_key=True)
        app.text_area[0].set_value("배송이 빠르고 품질도 좋았습니다.")
        app.button[0].click().run()

        assert not app.exception
        errors = " ".join(e.value for e in app.error)
        assert "한도" in errors


class TestResultRendering:
    """분석에 성공했을 때 결과 화면이 제대로 그려지는지 확인한다."""

    FAKE_RESULT = {
        "pros": ["배송이 빠름", "디자인이 예쁨"],
        "cons": ["포장이 허술함"],
        "sentiment_score": 2.5,
        "sentiment_reason": "불만 표현이 더 강하게 나타납니다.",
        "exaggeration": {
            "gap": 2.5,
            "abs_gap": 2.5,
            "is_exaggerated": True,
            "is_understated": False,
            "verdict": "과장 의심",
            "explanation": "별점이 내용보다 2.5점 높습니다.",
            "threshold": 1.5,
        },
        "summary": "별점에 비해 내용이 부정적인 리뷰입니다.",
        "tool_trace": [
            {
                "name": "classify_review_text",
                "args": {"review_text": "배송은 빨랐지만 포장이 엉망이었어요."},
                "ok": True,
                "result": {"pros": ["배송이 빠름"], "cons": ["포장이 허술함"]},
                "error": None,
                "response": {},
            },
            {
                "name": "score_sentiment",
                "args": {"review_text": "배송은 빨랐지만 포장이 엉망이었어요."},
                "ok": False,
                "result": None,
                "error": "테스트용 실패",
                "response": {},
            },
        ],
        "elapsed_sec": 3.4,
    }

    @pytest.fixture
    def analyzed_app(self, monkeypatch):
        patch_analyze(monkeypatch, lambda **kwargs: self.FAKE_RESULT)

        app = run_app(monkeypatch, with_key=True)
        app.text_area[0].set_value("배송은 빨랐지만 포장이 엉망이었어요.")
        app.button[0].click().run()
        return app

    def test_예외_없이_결과를_그린다(self, analyzed_app):
        assert not analyzed_app.exception

    def test_감성_점수를_지표로_보여준다(self, analyzed_app):
        metrics = [m.value for m in analyzed_app.metric]

        assert any("2.5" in str(v) for v in metrics)

    def test_장점과_단점을_모두_보여준다(self, analyzed_app):
        page_text = " ".join(m.value for m in analyzed_app.markdown)

        assert "배송이 빠름" in page_text
        assert "포장이 허술함" in page_text

    def test_과장_의심은_빨간_오류_박스로_강조한다(self, analyzed_app):
        errors = " ".join(e.value for e in analyzed_app.error)

        assert "과장 의심" in errors

    def test_에이전트_리포트를_보여준다(self, analyzed_app):
        page_text = " ".join(str(m.value) for m in analyzed_app.markdown)

        assert "별점에 비해 내용이 부정적인 리뷰입니다." in page_text

    def test_실패한_도구_호출도_기록에_남는다(self, analyzed_app):
        errors = " ".join(e.value for e in analyzed_app.error)

        assert "테스트용 실패" in errors
