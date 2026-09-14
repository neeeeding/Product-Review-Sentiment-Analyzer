"""재시도/백오프 로직 테스트. 실제 API를 호출하지 않는다.

Gemini 오류를 가짜로 만들어 던져서, 어떤 오류에 재시도하고 어떤 오류에
즉시 포기하는지 검증한다. 무료 등급 한도(429) 대응이 이 프로젝트의
핵심 요구사항이라 테스트로 고정해 둔다.
"""

import pytest
from google.genai import errors as genai_errors

from src import gemini_client


def make_api_error(code: int, status: str = "ERROR", message: str = "테스트 오류"):
    """주어진 HTTP 코드를 가진 가짜 Gemini API 오류를 만든다."""
    return genai_errors.APIError(
        code, {"error": {"code": code, "status": status, "message": message}}
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """테스트가 실제로 몇 초씩 기다리지 않도록 sleep을 가로챈다.

    대신 호출된 대기 시간을 기록해서 백오프가 실제로 늘어나는지 확인한다.
    """
    recorded = []
    monkeypatch.setattr(gemini_client.time, "sleep", recorded.append)
    return recorded


class TestRetryBehavior:
    def test_성공하면_재시도하지_않는다(self, no_sleep):
        calls = []

        def succeed():
            calls.append(1)
            return "성공"

        assert gemini_client._call_with_retry(succeed) == "성공"
        assert len(calls) == 1
        assert no_sleep == []

    def test_429면_재시도하고_결국_성공한다(self, no_sleep):
        attempts = []

        def fail_twice_then_succeed():
            attempts.append(1)
            if len(attempts) < 3:
                raise make_api_error(429, "RESOURCE_EXHAUSTED")
            return "성공"

        result = gemini_client._call_with_retry(fail_twice_then_succeed)

        assert result == "성공"
        assert len(attempts) == 3
        assert len(no_sleep) == 2  # 실패 2번 사이에 2번 대기

    def test_503도_재시도한다(self, no_sleep):
        attempts = []

        def fail_once():
            attempts.append(1)
            if len(attempts) < 2:
                raise make_api_error(503, "UNAVAILABLE")
            return "성공"

        assert gemini_client._call_with_retry(fail_once) == "성공"
        assert len(attempts) == 2

    def test_계속_429면_RateLimitExceededError를_던진다(self, no_sleep):
        def always_rate_limited():
            raise make_api_error(429, "RESOURCE_EXHAUSTED")

        with pytest.raises(gemini_client.RateLimitExceededError) as excinfo:
            gemini_client._call_with_retry(always_rate_limited)

        assert "한도" in str(excinfo.value)

    def test_재시도_횟수는_MAX_RETRIES를_넘지_않는다(self, no_sleep):
        attempts = []

        def always_fail():
            attempts.append(1)
            raise make_api_error(429)

        with pytest.raises(gemini_client.RateLimitExceededError):
            gemini_client._call_with_retry(always_fail)

        assert len(attempts) == gemini_client.MAX_RETRIES
        # 마지막 실패 뒤에는 기다리지 않는다.
        assert len(no_sleep) == gemini_client.MAX_RETRIES - 1


class TestNonRetryableErrors:
    """키가 틀렸거나 요청이 잘못된 경우는 재시도해도 소용없으므로 즉시 포기한다."""

    @pytest.mark.parametrize("code", [400, 401, 403, 404])
    def test_클라이언트_오류는_즉시_포기한다(self, code, no_sleep):
        attempts = []

        def fail():
            attempts.append(1)
            raise make_api_error(code)

        with pytest.raises(gemini_client.GeminiCallError):
            gemini_client._call_with_retry(fail)

        assert len(attempts) == 1  # 재시도 없음
        assert no_sleep == []

    def test_403_오류에_권한_관련_안내가_붙는다(self, no_sleep):
        def fail():
            raise make_api_error(403, "PERMISSION_DENIED")

        with pytest.raises(gemini_client.GeminiCallError, match="권한"):
            gemini_client._call_with_retry(fail)

    def test_404_오류에_모델_이름_안내가_붙는다(self, no_sleep):
        def fail():
            raise make_api_error(404, "NOT_FOUND")

        with pytest.raises(gemini_client.GeminiCallError, match="모델"):
            gemini_client._call_with_retry(fail)


class TestBackoffDelay:
    def test_대기_시간이_시도마다_늘어난다(self):
        delays = [gemini_client._backoff_delay(i) for i in range(4)]

        assert delays == sorted(delays)
        assert delays[0] < delays[-1]

    def test_첫_대기는_기본값_이상_1점25배_이하다(self):
        delay = gemini_client._backoff_delay(0)
        base = gemini_client.BASE_DELAY_SEC

        assert base <= delay <= base * 1.25

    def test_대기_시간에_상한이_있다(self):
        # 시도 횟수가 아무리 커져도 무한정 기다리지 않는다.
        delay = gemini_client._backoff_delay(20)

        assert delay <= gemini_client.MAX_DELAY_SEC * 1.25

    def test_지터가_있어서_매번_같지_않다(self):
        delays = {gemini_client._backoff_delay(2) for _ in range(20)}

        # 무작위 지터가 있으므로 20번 뽑으면 값이 여러 개 나와야 한다.
        assert len(delays) > 1


class TestClientCaching:
    def test_reset_client는_캐시를_비운다(self, monkeypatch):
        monkeypatch.setattr(gemini_client, "_client", object())

        gemini_client.reset_client()

        assert gemini_client._client is None
