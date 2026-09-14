"""설정/API 키 로딩 테스트. 실제 API 키 없이 동작한다."""

import pytest

from src import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """각 테스트를 깨끗한 환경변수 상태에서 시작한다."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


class TestApiKeyLoading:
    def test_GOOGLE_API_KEY를_읽는다(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test-key")

        assert config.get_api_key() == "AIza-test-key"

    def test_GOOGLE_API_KEY가_없으면_GEMINI_API_KEY로_넘어간다(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-fallback")

        assert config.get_api_key() == "AIza-fallback"

    def test_GOOGLE_API_KEY가_우선한다(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "primary")
        monkeypatch.setenv("GEMINI_API_KEY", "fallback")

        assert config.get_api_key() == "primary"

    def test_따옴표와_공백을_제거한다(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", '  "AIza-quoted"  ')

        assert config.get_api_key() == "AIza-quoted"

    def test_키가_없으면_해결방법이_담긴_예외를_던진다(self):
        with pytest.raises(config.MissingApiKeyError) as excinfo:
            config.get_api_key()

        message = str(excinfo.value)
        assert ".env" in message
        assert "aistudio.google.com" in message

    def test_빈_문자열은_키가_없는_것으로_본다(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "   ")

        with pytest.raises(config.MissingApiKeyError):
            config.get_api_key()

    def test_템플릿_안내문구를_그대로_두면_잡아낸다(self, monkeypatch):
        # .env.example 을 복사만 하고 키를 안 바꾼 흔한 실수
        monkeypatch.setenv("GOOGLE_API_KEY", "여기에_본인의_API_키를_붙여넣으세요")

        with pytest.raises(config.MissingApiKeyError, match="안내 문구"):
            config.get_api_key()

    def test_has_api_key는_예외_대신_불리언을_준다(self, monkeypatch):
        assert config.has_api_key() is False

        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")
        assert config.has_api_key() is True


class TestModelName:
    def test_기본_모델은_기획서의_gemini_2_0_flash다(self):
        assert config.get_model_name() == "gemini-2.0-flash"
        assert config.DEFAULT_MODEL == "gemini-2.0-flash"

    def test_환경변수로_모델을_바꿀_수_있다(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

        assert config.get_model_name() == "gemini-2.5-flash"


class TestLimits:
    def test_입력_길이_가이드는_기획서대로_2000자다(self):
        assert config.MAX_REVIEW_CHARS == 2000

    def test_최소_길이가_최대_길이보다_작다(self):
        assert 0 < config.MIN_REVIEW_CHARS < config.MAX_REVIEW_CHARS
