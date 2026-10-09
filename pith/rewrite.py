"""Output profiles (spec §5) and cache-safe request rewriting (spec §7).

Never touches: system, tools, model, thinking, max_tokens/max_completion_tokens/max_output_tokens, temperature.
"""
import copy
from dataclasses import dataclass

PROFILES = ("P0", "P1", "P1b", "P2", "P3", "P4")
SHAPE_TEXT = ("Answer directly. No preamble, restatement, or closing summary. "
              "Target at most {n} words unless the task genuinely needs more.")
EXEMPLAR_PREFIX = "\n\nExample of the expected length:\n"

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


def _anthropic_shape(body: dict, state: RouteState) -> None:
    msgs = body.setdefault("messages", [])
    text = _shape(state)
    if state.injection_form == "system" and msgs and msgs[-1].get("role") == "user":
        msgs.append({"role": "system", "content": text})
        return
    for m in reversed(msgs):
        if m.get("role") == "user":
            _append_user_text(m, text)
            return


def _openai_shape(body: dict, state: RouteState, responses_api: bool) -> None:
    text = _shape(state)
    if responses_api:
        inp = body.get("input", "")
        if isinstance(inp, str):
            inp = [{"role": "user", "content": inp}]
        body["input"] = list(inp) + [{"role": "developer", "content": text}]
    else:
        body["messages"] = list(body.get("messages") or []) + [{"role": "developer", "content": text}]


def apply_profile(provider: str, body: dict, state: RouteState, responses_api: bool = False) -> dict:
    out = copy.deepcopy(body)
    p = state.profile
    if p == "P0" or p not in PROFILES:
        return out
    if provider == "anthropic":
        if p in ("P1", "P4"):
            _anthropic_effort(out)
        if p in ("P2", "P3", "P4"):
            _anthropic_shape(out, state)
        return out
    if provider == "litellm":
        # LiteLLM folds system/developer messages into the provider's system prompt (cache-breaking) and a guardrail
        # cannot retry a rejected request: only the user-text shape, nothing else.
        if p in ("P2", "P3"):
            _anthropic_shape(out, RouteState(p, "user_text", state.target_words, state.exemplar))
        return out
    if p in ("P1", "P4"):
        if responses_api:
            r = out.get("reasoning") or {}
            out["reasoning"] = dict(r, effort=_step_down(r.get("effort", "medium")))
        else:
            out["reasoning_effort"] = _step_down(out.get("reasoning_effort", "medium"))
    if p == "P1b":
        if responses_api:
            out["text"] = dict(out.get("text") or {}, verbosity="low")
        else:
            out["verbosity"] = "low"
    if p in ("P2", "P3", "P4"):
        _openai_shape(out, state, responses_api)
    return out


def is_system_role_rejection(status: int, text: str) -> bool:
    return status == 400 and "role 'system' is not supported" in text
