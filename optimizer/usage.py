"""Usage + stop reason from provider responses, non-streaming and SSE (spec §8)."""
import json
from dataclasses import dataclass

_enc = None


def estimate_tokens(text: str) -> int:
    global _enc
    try:
        if _enc is None:
            import tiktoken
            _enc = tiktoken.get_encoding("o200k_base")
        return len(_enc.encode(text))
    except Exception:  # tiktoken missing or encoding download blocked
        return max(1, len(text) // 4)


@dataclass
class Usage:
    input_tokens: int | None
    output_tokens: int | None
    cache_read: int | None
    cache_create: int | None
    stop_reason: str | None
    estimated: bool = False


def usage_from_body(provider: str, body: dict) -> Usage:
    u = body.get("usage") or {}
    if provider == "anthropic":
        return Usage(u.get("input_tokens"), u.get("output_tokens"), u.get("cache_read_input_tokens"),
                     u.get("cache_creation_input_tokens"), body.get("stop_reason"))
    if body.get("object") == "response" or "input_tokens" in u:
        stop = (body.get("incomplete_details") or {}).get("reason") or body.get("status")
        return Usage(u.get("input_tokens"), u.get("output_tokens"),
                     (u.get("input_tokens_details") or {}).get("cached_tokens"), None, stop)
    choices = body.get("choices") or [{}]
    return Usage(u.get("prompt_tokens"), u.get("completion_tokens"),
                 (u.get("prompt_tokens_details") or {}).get("cached_tokens"), None, choices[0].get("finish_reason"))


class StreamUsage:
    def __init__(self, provider: str):
        self.provider = provider
        self._buf = b""
        self._u = Usage(None, None, None, None, None)
        self._text = []

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                self._event(json.loads(payload))
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

    def _event(self, ev: dict) -> None:
        u = self._u
        if self.provider == "anthropic":
            if ev.get("type") == "message_start":
                mu = (ev.get("message") or {}).get("usage") or {}
                u.input_tokens, u.cache_read, u.cache_create = mu.get("input_tokens"), mu.get("cache_read_input_tokens"), mu.get("cache_creation_input_tokens")
            elif ev.get("type") == "message_delta":
                u.output_tokens = (ev.get("usage") or {}).get("output_tokens", u.output_tokens)
                u.stop_reason = (ev.get("delta") or {}).get("stop_reason", u.stop_reason)
            return
        if ev.get("type") == "response.completed":
            self._u = usage_from_body("openai", ev.get("response") or {})
            return
        for c in ev.get("choices") or []:
            txt = (c.get("delta") or {}).get("content")
            if txt:
                self._text.append(txt)
            if c.get("finish_reason"):
                u.stop_reason = c["finish_reason"]
        if ev.get("usage"):
            got = usage_from_body("openai", {"usage": ev["usage"], "choices": [{"finish_reason": u.stop_reason}]})
            got.stop_reason = u.stop_reason
            self._u = got

    def result(self) -> Usage:
        u = self._u
        if self.provider == "openai" and u.output_tokens is None and self._text:
            u.output_tokens, u.estimated = estimate_tokens("".join(self._text)), True
        return u
