"""Route key = provider + model + hash(normalized system prompt + sorted tool signatures). Spec §4."""
import hashlib
import json
from typing import NamedTuple


class Fingerprint(NamedTuple):
    key: str
    model: str
    system_hash: str


def normalize(text: str) -> str:
    return " ".join(text.split())


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type", "text") in ("text", "input_text"))
    return ""


def system_text(provider: str, body: dict) -> str:
    if provider == "anthropic":
        return _text_of(body.get("system", ""))
    parts = []
    if "messages" in body:  # chat completions
        for m in body.get("messages") or []:
            if m.get("role") not in ("system", "developer"):
                break
            parts.append(_text_of(m.get("content", "")))
    else:  # responses
        if body.get("instructions"):
            parts.append(str(body["instructions"]))
        inp = body.get("input")
        if isinstance(inp, list):
            for item in inp:
                if not isinstance(item, dict) or item.get("role") not in ("system", "developer"):
                    break
                parts.append(_text_of(item.get("content", "")))
    return "\n".join(parts)


def tool_signature(provider: str, body: dict) -> str:
    sigs = []
    for t in body.get("tools") or []:
        name = t.get("name") or (t.get("function") or {}).get("name") or t.get("type", "")
        sigs.append(f"{name}:{json.dumps(t, sort_keys=True, separators=(',', ':'))}")
    return "|".join(sorted(sigs))


def fingerprint(provider: str, body: dict, override: str | None = None) -> Fingerprint:
    model = str(body.get("model", ""))
    raw = normalize(system_text(provider, body)) + "\x00" + tool_signature(provider, body)
    system_hash = hashlib.sha256(raw.encode()).hexdigest()
    key = override if override else f"{provider}:{model}:{system_hash[:16]}"
    return Fingerprint(key, model, system_hash)
