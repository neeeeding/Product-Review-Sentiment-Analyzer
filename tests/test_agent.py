"""에이전트 함수 호출 루프 테스트.

실제 Gemini를 호출하지 않는다. `generate` 를 대본(script)대로 응답하는
가짜로 바꿔서, 도구 호출 루프가 제대로 도는지 검증한다.
"""

import pytest
from google.genai import types

from src import agent


def function_call_response(*calls):
    """도구를 호출하겠다는 Gemini 응답을 흉내낸다.

    calls: (도구이름, 인자dict) 튜플들.
    """
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args=args))
        for name, args in calls
    ]
    content = types.Content(role="model", parts=parts)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content)]
    )


def text_response(text):
    """도구를 더 부르지 않고 최종 답변을 내놓는 응답을 흉내낸다."""
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content)]
    )


@pytest.fixture
def scripted_gemini(monkeypatch):
    """미리 정해둔 순서대로 응답을 돌려주는 가짜 generate."""

    class Script:
        def __init__(self):
            self.responses = []
            self.contents_history = []
            self.config_history = []

        def __call__(self, contents=None, config=None, model=None):
            self.contents_history.append(contents)
            self.config_history.append(config)
            if not self.responses:
                raise AssertionError("대본에 없는 추가 호출이 발생했습니다.")
            return self.responses.pop(0)

    script = Script()
    monkeypatch.setattr(agent, "generate", script)
    return script


@pytest.fixture
def fake_tools(monkeypatch):
    """도구 3개를 고정 결과를 주는 가짜로 바꾼다. 호출 인자를 기록한다."""
    recorded = {"calls": []}

    def fake_classify(review_text):
        recorded["calls"].append(("classify_review_text", review_text))
        return {"pros": ["배송 빠름"], "cons": ["포장 허술"]}

    def fake_score(review_text):
        recorded["calls"].append(("score_sentiment", review_text))
        return {"score": 2.5, "reason": "불만이 더 강합니다."}

    monkeypatch.setitem(agent._TOOL_FUNCTIONS, "classify_review_text", fake_classify)
    monkeypatch.setitem(agent._TOOL_FUNCTIONS, "score_sentiment", fake_score)
    # detect_review_exaggeration 은 순수 계산이라 진짜를 그대로 쓴다.
    return recorded


REVIEW = "배송은 빨랐지만 포장이 엉망이라 상자가 찌그러져 왔어요."


class TestFullFlow:
    def test_별점이_있으면_도구_3개를_모두_거친다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [
            function_call_response(
                ("classify_review_text", {"review_text": REVIEW}),
                ("score_sentiment", {"review_text": REVIEW}),
            ),
            function_call_response(
                (
                    "detect_review_exaggeration",
                    {"actual_rating": 5, "sentiment_score": 2.5},
                )
            ),
            text_response("별점에 비해 내용이 부정적입니다."),
        ]

        result = agent.analyze_review(REVIEW, actual_rating=5)

        assert result["pros"] == ["배송 빠름"]
        assert result["cons"] == ["포장 허술"]
        assert result["sentiment_score"] == 2.5
        assert result["exaggeration"]["is_exaggerated"] is True
        assert result["summary"] == "별점에 비해 내용이 부정적입니다."
        assert [t["name"] for t in result["tool_trace"]] == [
            "classify_review_text",
            "score_sentiment",
            "detect_review_exaggeration",
        ]

    def test_별점이_없으면_도구3없이_끝난다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [
            function_call_response(
                ("classify_review_text", {"review_text": REVIEW}),
                ("score_sentiment", {"review_text": REVIEW}),
            ),
            text_response("전반적으로 아쉬운 평가입니다."),
        ]

        result = agent.analyze_review(REVIEW, actual_rating=None)

        assert result["sentiment_score"] == 2.5
        assert result["exaggeration"] is None

    def test_도구를_한번도_안_불러도_요약은_받는다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [text_response("분석 결과입니다.")]

        result = agent.analyze_review(REVIEW)

        assert result["summary"] == "분석 결과입니다."
        assert result["tool_trace"] == []
        assert result["sentiment_score"] is None

    def test_소요_시간을_기록한다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [text_response("끝")]

        result = agent.analyze_review(REVIEW)

        assert result["elapsed_sec"] >= 0


class TestPromptConstruction:
    def test_별점이_없으면_도구3을_부르지_말라고_지시한다(
        self, scripted_gemini, fake_tools
    ):
        scripted_gemini.responses = [text_response("끝")]

        agent.analyze_review(REVIEW, actual_rating=None)

        prompt = scripted_gemini.contents_history[0][0].parts[0].text
        assert "detect_review_exaggeration 는 호출하지 마라" in prompt

    def test_별점이_있으면_프롬프트에_별점을_알려준다(
        self, scripted_gemini, fake_tools
    ):
        scripted_gemini.responses = [text_response("끝")]

        agent.analyze_review(REVIEW, actual_rating=4)

        prompt = scripted_gemini.contents_history[0][0].parts[0].text
        assert "4점" in prompt

    def test_리뷰_원문이_프롬프트에_들어간다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [text_response("끝")]

        agent.analyze_review(REVIEW)

        prompt = scripted_gemini.contents_history[0][0].parts[0].text
        assert REVIEW in prompt


class TestToolRegistration:
    def test_기획서대로_도구_3개가_등록되어_있다(self):
        names = {d.name for d in agent._TOOL_DECLARATIONS}

        assert names == {
            "classify_review_text",
            "score_sentiment",
            "detect_review_exaggeration",
        }

    def test_선언된_도구마다_실제_함수가_연결되어_있다(self):
        for declaration in agent._TOOL_DECLARATIONS:
            assert declaration.name in agent._TOOL_FUNCTIONS

    def test_자동_함수호출은_꺼져있다(self, scripted_gemini, fake_tools):
        # 꺼져 있어야 도구 호출 기록을 직접 남길 수 있다.
        scripted_gemini.responses = [text_response("끝")]

        agent.analyze_review(REVIEW)

        config = scripted_gemini.config_history[0]
        assert config.automatic_function_calling.disable is True

    def test_도구_3개가_모두_config로_전달된다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [text_response("끝")]

        agent.analyze_review(REVIEW)

        config = scripted_gemini.config_history[0]
        declared = {d.name for d in config.tools[0].function_declarations}
        assert declared == {
            "classify_review_text",
            "score_sentiment",
            "detect_review_exaggeration",
        }


class TestToolErrorHandling:
    def test_도구가_실패해도_모델에게_오류를_돌려주고_계속한다(
        self, scripted_gemini, fake_tools, monkeypatch
    ):
        def broken_classify(review_text):
            raise ValueError("리뷰가 너무 짧습니다.")

        monkeypatch.setitem(
            agent._TOOL_FUNCTIONS, "classify_review_text", broken_classify
        )
        scripted_gemini.responses = [
            function_call_response(("classify_review_text", {"review_text": REVIEW})),
            text_response("분류에 실패했습니다."),
        ]

        result = agent.analyze_review(REVIEW)

        record = result["tool_trace"][0]
        assert record["ok"] is False
        assert "너무 짧습니다" in record["error"]
        assert result["summary"] == "분류에 실패했습니다."

    def test_모르는_도구를_부르면_오류로_기록한다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [
            function_call_response(("존재하지_않는_도구", {})),
            text_response("끝"),
        ]

        result = agent.analyze_review(REVIEW)

        assert result["tool_trace"][0]["ok"] is False
        assert "등록되지 않은 도구" in result["tool_trace"][0]["error"]

    def test_인자_이름이_틀리면_오류로_기록한다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [
            function_call_response(("score_sentiment", {"wrong_arg": "값"})),
            text_response("끝"),
        ]

        result = agent.analyze_review(REVIEW)

        assert result["tool_trace"][0]["ok"] is False
        assert "인자가 잘못되었습니다" in result["tool_trace"][0]["error"]


class TestThresholdInjection:
    def test_임계값은_앱_설정값이_주입된다(self, scripted_gemini, fake_tools):
        # 모델이 threshold를 멋대로 정하지 못하게 앱이 덮어쓴다.
        scripted_gemini.responses = [
            function_call_response(
                (
                    "detect_review_exaggeration",
                    {"actual_rating": 5, "sentiment_score": 4.0, "threshold": 0.1},
                )
            ),
            text_response("끝"),
        ]

        result = agent.analyze_review(REVIEW, actual_rating=5, threshold=2.0)

        assert result["exaggeration"]["threshold"] == 2.0
        assert result["exaggeration"]["is_exaggerated"] is False


class TestLoopSafety:
    def test_도구를_계속_부르면_MAX_TURNS에서_멈춘다(
        self, scripted_gemini, fake_tools
    ):
        # 모델이 같은 도구를 무한히 부르는 상황
        scripted_gemini.responses = [
            function_call_response(("score_sentiment", {"review_text": REVIEW}))
            for _ in range(agent.MAX_TURNS)
        ]

        with pytest.raises(agent.AgentError, match=str(agent.MAX_TURNS)):
            agent.analyze_review(REVIEW)

    def test_같은_도구가_여러번_성공하면_마지막_결과를_쓴다(
        self, scripted_gemini, fake_tools, monkeypatch
    ):
        scores = iter([2.0, 4.0])

        def changing_score(review_text):
            return {"score": next(scores), "reason": "테스트"}

        monkeypatch.setitem(agent._TOOL_FUNCTIONS, "score_sentiment", changing_score)
        scripted_gemini.responses = [
            function_call_response(("score_sentiment", {"review_text": REVIEW})),
            function_call_response(("score_sentiment", {"review_text": REVIEW})),
            text_response("끝"),
        ]

        result = agent.analyze_review(REVIEW)

        assert result["sentiment_score"] == 4.0


class TestInputValidation:
    def test_짧은_리뷰는_API를_쓰기_전에_거부한다(self, scripted_gemini):
        with pytest.raises(ValueError, match="짧습니다"):
            agent.analyze_review("짧음")

        assert scripted_gemini.contents_history == []

    def test_긴_리뷰는_API를_쓰기_전에_거부한다(self, scripted_gemini):
        with pytest.raises(ValueError, match="깁니다"):
            agent.analyze_review("좋" * 2001)

        assert scripted_gemini.contents_history == []

    @pytest.mark.parametrize("bad_rating", [0, 6, -1])
    def test_범위를_벗어난_별점은_거부한다(self, scripted_gemini, bad_rating):
        with pytest.raises(ValueError, match="1~5"):
            agent.analyze_review(REVIEW, actual_rating=bad_rating)

    def test_별점_None은_입력하지_않음을_뜻한다(self, scripted_gemini, fake_tools):
        scripted_gemini.responses = [text_response("끝")]

        result = agent.analyze_review(REVIEW, actual_rating=None)

        assert result["exaggeration"] is None

    def test_별점을_문자열로_줘도_받아준다(self, scripted_gemini, fake_tools):
        # Streamlit 위젯이 문자열로 넘길 가능성에 대비
        scripted_gemini.responses = [text_response("끝")]

        agent.analyze_review(REVIEW, actual_rating="4")

        prompt = scripted_gemini.contents_history[0][0].parts[0].text
        assert "4점" in prompt
