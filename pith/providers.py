"""Which upstream a path belongs to. Unknown paths are forwarded without inspection (spec §7)."""
from pith.config import Config

_PATHS = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai", "/v1/responses": "openai"}


def detect_provider(path: str) -> str | None:
    return _PATHS.get(path.rstrip("/"))


def is_responses_api(path: str) -> bool:
    return path.rstrip("/") == "/v1/responses"


def upstream(provider: str, config: Config) -> str:
    if provider == "anthropic":
        return config.anthropic_upstream
    return config.litellm_upstream if provider == "litellm" else config.openai_upstream
