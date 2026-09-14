"""도구 ①②(classify_review_text, score_sentiment) 테스트.

실제 Gemini를 호출하지 않는다. `generate` 함수를 가짜 응답으로 바꿔치기해서
입력 검증 / JSON 파싱 / 결과 정리 로직만 검증한다.
(실제 API 연동은 scripts/check_connection.py 로 따로 확인한다.)
"""

import pytest

from src import tools
from src.config import MAX_REVIEW_CHARS, MIN_REVIEW_CHARS


class FakeResponse:
    """Gemini 응답 객체 흉내. .text 만 있으면 충분하다."""

    def __init__(self, text):
        self.text = text


@pytest.fixture
def fake_gemini(monkeypatch):
    """`generate` 를 가로채서 원하는 텍스트를 돌려주게 만든다.

    반환된 객체의 .calls 에 호출 인자가 쌓이므로 프롬프트도 확인할 수 있다.
    """

    class Recorder:
        def __init__(self):
            self.calls = []
            self.reply = "{}"

        def __call__(self, contents=None, config=None, model=None):
            self.calls.append({"contents": contents, "config": config})
            return FakeResponse(self.reply)

    recorder = Recorder()
    monkeypatch.setattr(tools, "generate", recorder)
    return recorder


# ---------------------------------------------------------------------------
# 도구 ① 분류
# ---------------------------------------------------------------------------


class TestClassifyReviewText:
    def test_장점과_단점을_그대로_돌려준다(self, fake_gemini):
        fake_gemini.reply = (
            '{"pros": ["배송이 빠름", "가격이 저렴함"], "cons": ["포장이 허술함"]}'
        )

        result = tools.classify_review_text("배송은 빨랐는데 포장이 엉망이었어요.")

        assert result == {
            "pros": ["배송이 빠름", "가격이 저렴함"],
            "cons": ["포장이 허술함"],
        }

    def test_한쪽이_비어도_정상_동작한다(self, fake_gemini):
        fake_gemini.reply = '{"pros": ["다 좋아요"], "cons": []}'

        result = tools.classify_review_text("정말 만족스러운 상품입니다.")

        assert result["pros"] == ["다 좋아요"]
        assert result["cons"] == []

    def test_키가_아예_없어도_빈_리스트로_채운다(self, fake_gemini):
        fake_gemini.reply = "{}"

        result = tools.classify_review_text("괜찮은 상품이네요.")

        assert result == {"pros": [], "cons": []}

    def test_중복_항목을_제거한다(self, fake_gemini):
        fake_gemini.reply = '{"pros": ["빠른 배송", "빠른 배송", "  빠른 배송  "], "cons": []}'

        result = tools.classify_review_text("배송이 정말 빨라요.")

        assert result["pros"] == ["빠른 배송"]

    def test_빈_문자열과_null을_걸러낸다(self, fake_gemini):
        fake_gemini.reply = '{"pros": ["좋음", "", "   ", null], "cons": []}'

        result = tools.classify_review_text("좋은 상품입니다.")

        assert result["pros"] == ["좋음"]

    def test_마크다운_코드블록으로_감싸져_와도_파싱한다(self, fake_gemini):
        fake_gemini.reply = '```json\n{"pros": ["좋음"], "cons": []}\n```'

        result = tools.classify_review_text("좋은 상품입니다.")

        assert result["pros"] == ["좋음"]

    def test_리스트_대신_문자열이_와도_감싸준다(self, fake_gemini):
        fake_gemini.reply = '{"pros": "배송이 빠름", "cons": []}'

        result = tools.classify_review_text("배송이 빨라요.")

        assert result["pros"] == ["배송이 빠름"]

    def test_리뷰_원문이_프롬프트에_포함된다(self, fake_gemini):
        fake_gemini.reply = '{"pros": [], "cons": []}'

        tools.classify_review_text("이 상품 정말 좋아요")

        assert "이 상품 정말 좋아요" in fake_gemini.calls[0]["contents"]

    def test_JSON_응답_모드로_요청한다(self, fake_gemini):
        fake_gemini.reply = '{"pros": [], "cons": []}'

        tools.classify_review_text("좋은 상품입니다.")

        config = fake_gemini.calls[0]["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is not None


# ---------------------------------------------------------------------------
# 도구 ② 감성 점수
# ---------------------------------------------------------------------------


class TestScoreSentiment:
    def test_점수와_근거를_돌려준다(self, fake_gemini):
        fake_gemini.reply = '{"score": 4.5, "reason": "칭찬이 대부분입니다."}'

        result = tools.score_sentiment("정말 마음에 들어요. 재구매 의사 있습니다.")

        assert result["score"] == 4.5
        assert result["reason"] == "칭찬이 대부분입니다."

    def test_정수로_와도_float으로_바꾼다(self, fake_gemini):
        fake_gemini.reply = '{"score": 3, "reason": "보통입니다."}'

        result = tools.score_sentiment("그럭저럭 쓸만합니다.")

        assert result["score"] == 3.0
        assert isinstance(result["score"], float)

    @pytest.mark.parametrize(
        "raw_score,expected", [(0, 1.0), (-3, 1.0), (7, 5.0), (5.9, 5.0)]
    )
    def test_범위를_벗어난_점수는_1에서_5로_자른다(
        self, fake_gemini, raw_score, expected
    ):
        # 이걸 막지 않으면 도구 ③의 범위 검사에서 예외가 터진다.
        fake_gemini.reply = f'{{"score": {raw_score}, "reason": "테스트"}}'

        result = tools.score_sentiment("테스트 리뷰입니다.")

        assert result["score"] == expected

    def test_근거가_비면_안내_문구로_채운다(self, fake_gemini):
        fake_gemini.reply = '{"score": 3.0, "reason": ""}'

        result = tools.score_sentiment("그냥 그래요.")

        assert result["reason"]

    def test_점수를_숫자로_못_읽으면_오류를_낸다(self, fake_gemini):
        fake_gemini.reply = '{"score": "매우 좋음", "reason": "테스트"}'

        with pytest.raises(ValueError, match="숫자"):
            tools.score_sentiment("좋은 상품입니다.")

    def test_결과가_도구3에_바로_넘어갈_수_있다(self, fake_gemini):
        # 도구 ② → 도구 ③ 연결이 끊기지 않는지 확인
        fake_gemini.reply = '{"score": 2.0, "reason": "불만이 많습니다."}'

        sentiment = tools.score_sentiment("배송도 늦고 품질도 별로예요.")
        report = tools.detect_review_exaggeration(5, sentiment["score"])

        assert report["is_exaggerated"] is True


# ---------------------------------------------------------------------------
# 두 도구가 공유하는 입력 검증
# ---------------------------------------------------------------------------


class TestReviewTextValidation:
    @pytest.mark.parametrize(
        "tool_name", ["classify_review_text", "score_sentiment"]
    )
    def test_너무_짧은_리뷰는_거부한다(self, fake_gemini, tool_name):
        tool = getattr(tools, tool_name)

        with pytest.raises(ValueError, match="짧습니다"):
            tool("짧음")

        assert fake_gemini.calls == []  # API를 낭비하지 않는다

    @pytest.mark.parametrize(
        "tool_name", ["classify_review_text", "score_sentiment"]
    )
    def test_길이_제한을_넘는_리뷰는_거부한다(self, fake_gemini, tool_name):
        tool = getattr(tools, tool_name)
        too_long = "좋" * (MAX_REVIEW_CHARS + 1)

        with pytest.raises(ValueError, match="깁니다"):
            tool(too_long)

        assert fake_gemini.calls == []

    def test_제한_길이_정확히는_통과한다(self, fake_gemini):
        fake_gemini.reply = '{"pros": [], "cons": []}'
        exactly_max = "좋" * MAX_REVIEW_CHARS

        tools.classify_review_text(exactly_max)

        assert len(fake_gemini.calls) == 1

    @pytest.mark.parametrize("bad_input", [None, 123, ["리뷰"]])
    def test_문자열이_아니면_거부한다(self, fake_gemini, bad_input):
        with pytest.raises(ValueError, match="문자열"):
            tools.classify_review_text(bad_input)

    def test_앞뒤_공백은_제거하고_길이를_잰다(self, fake_gemini):
        fake_gemini.reply = '{"pros": [], "cons": []}'
        padded = "   " + "좋" * MIN_REVIEW_CHARS + "   "

        tools.classify_review_text(padded)

        sent = fake_gemini.calls[0]["contents"]
        assert "   \n" not in sent


class TestBrokenResponses:
    """Gemini가 예상과 다른 응답을 줬을 때 알아볼 수 있는 오류를 낸다."""

    def test_빈_응답은_안내와_함께_오류를_낸다(self, fake_gemini):
        fake_gemini.reply = ""

        with pytest.raises(ValueError, match="빈 응답"):
            tools.classify_review_text("좋은 상품입니다.")

    def test_JSON이_아니면_응답_앞부분을_보여준다(self, fake_gemini):
        fake_gemini.reply = "죄송합니다, 답변할 수 없습니다."

        with pytest.raises(ValueError, match="JSON으로 읽지 못했습니다"):
            tools.classify_review_text("좋은 상품입니다.")

    def test_JSON_객체가_아닌_배열이면_오류를_낸다(self, fake_gemini):
        fake_gemini.reply = '["좋음", "나쁨"]'

        with pytest.raises(ValueError, match="JSON 객체"):
            tools.classify_review_text("좋은 상품입니다.")
