"""도구 3번(detect_review_exaggeration) 단위 테스트.

이 도구는 LLM을 호출하지 않는 순수 계산이라 API 키 없이 테스트할 수 있다.

실행:
    venv\\Scripts\\python.exe -m pytest -v
"""

import math

import pytest

from src.tools import DEFAULT_THRESHOLD, detect_review_exaggeration


class TestExaggerationDetection:
    """별점이 리뷰 내용보다 후한 경우 = 과장 의심."""

    def test_별점5_감성2는_과장으로_판단한다(self):
        result = detect_review_exaggeration(actual_rating=5, sentiment_score=2.0)

        assert result["gap"] == 3.0
        assert result["is_exaggerated"] is True
        assert result["is_understated"] is False
        assert result["verdict"] == "과장 의심"

    def test_기준치와_정확히_같은_괴리도_과장으로_판단한다(self):
        # 경계값: gap == threshold 는 "이상"이므로 과장에 포함된다.
        result = detect_review_exaggeration(
            actual_rating=5, sentiment_score=3.5, threshold=1.5
        )

        assert result["gap"] == 1.5
        assert result["is_exaggerated"] is True

    def test_기준치보다_아주_조금_작으면_과장이_아니다(self):
        result = detect_review_exaggeration(
            actual_rating=5, sentiment_score=3.6, threshold=1.5
        )

        assert result["gap"] == 1.4
        assert result["is_exaggerated"] is False
        assert result["verdict"] == "대체로 일치"


class TestUnderstatementDetection:
    """별점이 리뷰 내용보다 박한 경우 = 과소평가 의심."""

    def test_별점1_감성4는_과소평가로_판단한다(self):
        result = detect_review_exaggeration(actual_rating=1, sentiment_score=4.0)

        assert result["gap"] == -3.0
        assert result["abs_gap"] == 3.0
        assert result["is_understated"] is True
        assert result["is_exaggerated"] is False
        assert result["verdict"] == "과소평가 의심"

    def test_과장과_과소평가는_동시에_참이_될_수_없다(self):
        for actual in (1, 2, 3, 4, 5):
            for sentiment in (1.0, 2.5, 3.0, 4.2, 5.0):
                result = detect_review_exaggeration(actual, sentiment)
                assert not (result["is_exaggerated"] and result["is_understated"])


class TestAgreement:
    """별점과 내용이 비슷한 경우."""

    def test_점수가_같으면_괴리가_0이다(self):
        result = detect_review_exaggeration(actual_rating=4, sentiment_score=4.0)

        assert result["gap"] == 0.0
        assert result["abs_gap"] == 0.0
        assert result["is_exaggerated"] is False
        assert result["is_understated"] is False
        assert result["verdict"] == "대체로 일치"

    def test_소수점_감성점수도_정상_계산된다(self):
        result = detect_review_exaggeration(actual_rating=4, sentiment_score=3.7)

        assert result["gap"] == pytest.approx(0.3)
        assert result["verdict"] == "대체로 일치"


class TestThreshold:
    """임계값을 바꿨을 때의 동작."""

    def test_기본_임계값은_기획서대로_1점5다(self):
        assert DEFAULT_THRESHOLD == 1.5

    def test_임계값을_낮추면_같은_입력이_과장으로_바뀐다(self):
        loose = detect_review_exaggeration(5, 4.0, threshold=1.5)
        strict = detect_review_exaggeration(5, 4.0, threshold=0.5)

        assert loose["is_exaggerated"] is False
        assert strict["is_exaggerated"] is True

    def test_사용한_임계값이_결과에_그대로_담긴다(self):
        result = detect_review_exaggeration(5, 1.0, threshold=2.0)

        assert result["threshold"] == 2.0


class TestInputValidation:
    """잘못된 입력은 조용히 넘어가지 않고 ValueError를 낸다."""

    @pytest.mark.parametrize("bad_rating", [0, 6, -1, 5.5, 100])
    def test_범위를_벗어난_별점은_거부한다(self, bad_rating):
        with pytest.raises(ValueError, match="actual_rating"):
            detect_review_exaggeration(bad_rating, 3.0)

    @pytest.mark.parametrize("bad_score", [0.0, 5.1, -2.0])
    def test_범위를_벗어난_감성점수는_거부한다(self, bad_score):
        with pytest.raises(ValueError, match="sentiment_score"):
            detect_review_exaggeration(3, bad_score)

    @pytest.mark.parametrize("bad_value", ["다섯점", None, [5]])
    def test_숫자가_아닌_입력은_거부한다(self, bad_value):
        with pytest.raises(ValueError):
            detect_review_exaggeration(bad_value, 3.0)

    def test_NaN은_거부한다(self):
        # NaN은 모든 비교가 False라서 범위 검사를 통과해버린다. 따로 막아야 한다.
        with pytest.raises(ValueError, match="NaN"):
            detect_review_exaggeration(math.nan, 3.0)

    @pytest.mark.parametrize("bad_threshold", [0, -1.5])
    def test_0이하_임계값은_거부한다(self, bad_threshold):
        with pytest.raises(ValueError, match="threshold"):
            detect_review_exaggeration(5, 2.0, threshold=bad_threshold)

    def test_문자열_숫자는_허용한다(self):
        # Streamlit이나 LLM이 "5" 같은 문자열로 넘길 수 있으므로 변환을 허용한다.
        result = detect_review_exaggeration("5", "2.0")

        assert result["gap"] == 3.0
        assert result["is_exaggerated"] is True


class TestOutputShape:
    """반환값이 항상 같은 모양이어야 Streamlit/에이전트에서 쓰기 쉽다."""

    EXPECTED_KEYS = {
        "gap",
        "abs_gap",
        "is_exaggerated",
        "is_understated",
        "verdict",
        "explanation",
        "threshold",
    }

    @pytest.mark.parametrize(
        "actual,sentiment", [(5, 1.0), (1, 5.0), (3, 3.0), (4, 2.5)]
    )
    def test_항상_동일한_키_집합을_반환한다(self, actual, sentiment):
        result = detect_review_exaggeration(actual, sentiment)

        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_설명에_두_점수가_모두_들어간다(self):
        result = detect_review_exaggeration(5, 2.0)

        assert "5" in result["explanation"]
        assert "2" in result["explanation"]

    def test_결과가_JSON으로_직렬화_가능하다(self):
        # Gemini function calling은 도구 결과를 JSON으로 주고받는다.
        import json

        result = detect_review_exaggeration(5, 2.0)
        restored = json.loads(json.dumps(result, ensure_ascii=False))

        assert restored["verdict"] == "과장 의심"
