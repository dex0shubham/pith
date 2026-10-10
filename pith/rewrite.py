"""Output profiles (spec §5) and cache-safe request rewriting (spec §7).

Never touches: system, tools, model, thinking, max_tokens/max_completion_tokens/max_output_tokens, temperature.
"""
import copy
from dataclasses import dataclass

PROFILES = ("P0", "P1", "P1b", "P2", "P3", "P4")
SHAPE_TEXT = ("Answer directly. No preamble, restatement, or closing summary. "
              "Target at most {n} words unless the task genuinely needs more.")
EXEMPLAR_PREFIX = "\n\nExample of the expected length:\n"
SHAPE_PREFIX = SHAPE_TEXT.split("{n}")[0]  # how strip_shape recognises pith's own appended text part

EFFORT_LADDER = ("low", "medium", "high", "xhigh", "max")
# Models whose default effort is not "high" (Anthropic pricing docs, 2026-09). Everything else defaults to "high".
ANTHROPIC_DEFAULT_EFFORT = {"claude-opus-5-5": "medium"}
# Models with no effort parameter: P1 is a no-op on these.
ANTHROPIC_NO_EFFORT = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-haiku-3", "claude-3-")


@dataclass
class RouteState:
    profile: str
    injection_form: str = "system"  # "system" | "user_text"
    target_words: int = 20
    exemplar: str | None = None


def _shape(state: RouteState) -> str:
    text = SHAPE_TEXT.format(n=max(20, int(state.target_words)))
    if state.profile == "P3" and state.exemplar:
        text += EXEMPLAR_PREFIX + state.exemplar
    return text


def _step_down(current: str) -> str:
    if current not in EFFORT_LADDER:
        return current  # Off-ladder values ("minimal", "none", etc.) pass through unchanged
    i = EFFORT_LADDER.index(current)
    return EFFORT_LADDER[max(0, i - 1)]


def _anthropic_effort(body: dict) -> None:
    model = body.get("model", "")
    if any(model.startswith(p) for p in ANTHROPIC_NO_EFFORT):
        return
    oc = body.get("output_config") or {}
    current = oc.get("effort") or ANTHROPIC_DEFAULT_EFFORT.get(model, "high")
    new = _step_down(current)
    if new != current:
        body["output_config"] = dict(oc, effort=new)


def _append_user_text(msg: dict, text: str) -> None:
    content = msg.get("content", "")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    msg["content"] = list(content) + [{"type": "text", "text": text}]


def append_shape(messages: list, state: RouteState) -> bool:
    """In place: the shape text as a text part of the last user message (the cache-safe user-text form).
    False when there is no user message to append to."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            _append_user_text(m, _shape(state))
            return True
    return False


def strip_shape(body: dict) -> tuple[dict, bool]:
    """Undo append_shape on a deep copy: (body, True) when the last user message ends with pith's shape text part
    (removed; a lone remaining plain text part becomes string content again), else (copy, False)."""
    out = copy.deepcopy(body)
    for m in reversed(out.get("messages") or []):
        if m.get("role") != "user":
            continue
        parts = m.get("content")
        if isinstance(parts, list) and parts and isinstance(parts[-1], dict) and parts[-1].get("type") == "text" \
                and str(parts[-1].get("text", "")).startswith(SHAPE_PREFIX):
            rest = parts[:-1]
            m["content"] = rest[0]["text"] if len(rest) == 1 and set(rest[0]) == {"type", "text"} else rest
            return out, True
        return out, False
    return out, False


def _anthropic_shape(body: dict, state: RouteState) -> None:
    msgs = body.setdefault("messages", [])
    if state.injection_form == "system" and msgs and msgs[-1].get("role") == "user":
        msgs.append({"role": "system", "content": _shape(state)})
        return
    append_shape(msgs, state)


def _openai_shape(body: dict, state: RouteState, responses_api: bool) -> None:
    text = _shape(state)
    if responses_api:
        inp = body.get("input", "")
        if isinstance(inp, str):
            inp = [{"role": "user", "content": inp}]
        body["input"] = list(inp) + [{"role": "developer", "content": text}]
    elif state.injection_form == "user_text":
        append_shape(body.setdefault("messages", []), state)  # the form a Headroom-recorded route is served in
    else:
        body["messages"] = list(body.get("messages") or []) + [{"role": "developer", "content": text}]


def apply_effort(provider: str, body: dict, profile: str, responses_api: bool = False) -> dict:
    """The parameter half of a profile: effort step-down (P1, P4) and the OpenAI verbosity flag (P1b). Deep copy."""
    out = copy.deepcopy(body)
    if provider == "anthropic":
        if profile in ("P1", "P4"):
            _anthropic_effort(out)
        return out
    if profile in ("P1", "P4"):
        if responses_api:
            r = out.get("reasoning") or {}
            out["reasoning"] = dict(r, effort=_step_down(r.get("effort", "medium")))
        else:
            out["reasoning_effort"] = _step_down(out.get("reasoning_effort", "medium"))
    if profile == "P1b":
        if responses_api:
            out["text"] = dict(out.get("text") or {}, verbosity="low")
        else:
            out["verbosity"] = "low"
    return out


def apply_profile(provider: str, body: dict, state: RouteState, responses_api: bool = False) -> dict:
    p = state.profile
    if p == "P0" or p not in PROFILES:
        return copy.deepcopy(body)
    if provider in ("litellm", "portkey"):
        # A gateway hook cannot retry a rejected request and folds system messages: only the user-text shape.
        out = copy.deepcopy(body)
        if p in ("P2", "P3"):
            append_shape(out.setdefault("messages", []), state)
        return out
    out = apply_effort(provider, body, p, responses_api)
    if p in ("P2", "P3", "P4"):
        if provider == "anthropic":
            _anthropic_shape(out, state)
        else:
            _openai_shape(out, state, responses_api)
    return out


def is_system_role_rejection(status: int, text: str) -> bool:
    return status == 400 and "role 'system' is not supported" in text
