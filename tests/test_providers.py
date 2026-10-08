from pith.config import Config
from pith.providers import detect_provider, is_responses_api, upstream


def test_detects_by_path():
    assert detect_provider("/v1/messages") == "anthropic"
    assert detect_provider("/v1/chat/completions") == "openai"
    assert detect_provider("/v1/responses") == "openai"
    assert detect_provider("/v1/messages/count_tokens") is None
    assert detect_provider("/v1/models") is None
    assert is_responses_api("/v1/responses") and not is_responses_api("/v1/chat/completions")


def test_upstream_from_config():
    c = Config(anthropic_upstream="http://a", openai_upstream="http://o")
    assert upstream("anthropic", c) == "http://a" and upstream("openai", c) == "http://o"
