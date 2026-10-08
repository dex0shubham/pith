"""The equivalence judge: one frozen prompt, one call, one label. Spec (Plan 2) §6.

The model answers with a slot-qualified label (A-omits, B-broken, ...). normalize() maps it to the stored label set
seen from the candidate's side: equivalent missing-info extra-info contradiction format-broken judge-error.
"""
import random
import re

from optimizer.config import Config
from optimizer.replay import call

JUDGE_PROMPT_VERSION = "v1"
LABELS = ("equivalent", "A-omits", "B-omits", "contradiction", "A-broken", "B-broken")
SYSTEM_PROMPT = ("You compare two answers to the same request. Decide whether they convey the same facts, decisions and "
                 "required output. Reply with exactly one label.")
USER_TEMPLATE = (
    "REQUEST:\n{question}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}\n\n"
    "Labels:\n"
    "equivalent - both answers convey the same facts, decisions and required output, nothing contradictory.\n"
    "A-omits - Answer A omits a fact, decision or required output that Answer B states.\n"
    "B-omits - Answer B omits a fact, decision or required output that Answer A states.\n"
    "contradiction - the answers assert incompatible things.\n"
    "A-broken - Answer A is empty, truncated, or not a usable answer.\n"
    "B-broken - Answer B is empty, truncated, or not a usable answer.\n\n"
    "Reply with exactly one label.")
_LABEL_RE = re.compile("|".join(re.escape(l) for l in LABELS), re.IGNORECASE)
_CANON = {l.lower(): l for l in LABELS}
# (model label -> stored label) per slot order; A/B are (baseline, candidate) for baseline-first, reversed otherwise.
_STORED = {
    "baseline-first": {"B-omits": "missing-info", "A-omits": "extra-info", "B-broken": "format-broken", "A-broken": "judge-error"},
    "candidate-first": {"A-omits": "missing-info", "B-omits": "extra-info", "A-broken": "format-broken", "B-broken": "judge-error"},
}


class JudgeUnavailable(Exception):
    pass


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") in ("text", "input_text"))
    return ""


def last_user_text(provider: str, body: dict, limit: int = 4000) -> str:
    items = body.get("messages")
    if items is None:
        inp = body.get("input", "")
        items = [{"role": "user", "content": inp}] if isinstance(inp, str) else inp
    for m in reversed(items or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return _text_of(m.get("content", ""))[:limit]
    return ""


def build_judge_request(provider: str, model: str, question: str, answer_a: str, answer_b: str) -> dict:
    user = USER_TEMPLATE.format(question=question, a=answer_a, b=answer_b)
    if provider == "anthropic":
        # Thinking tokens count toward max_tokens on current Claude models: leave room and keep effort low.
        return {"model": model, "max_tokens": 1024, "output_config": {"effort": "low"}, "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user}]}
    # Hidden reasoning tokens count toward max_completion_tokens on reasoning models: leave room for them.
    return {"model": model, "max_completion_tokens": 1024,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}


def parse_label(text: str) -> str | None:
    m = _LABEL_RE.search(text or "")
    return _CANON[m.group(0).lower()] if m else None


def normalize(label: str, order_ab: str) -> str:
    return _STORED[order_ab].get(label, label)


def judge(client, cfg: Config, keys: dict, question: str, baseline: str, candidate: str, rng: random.Random,
          prices=None) -> tuple[str, str, float]:
    key = keys.get(cfg.judge_provider)
    if not key:
        raise JudgeUnavailable(f"no API key for judge provider {cfg.judge_provider!r}")
    order = "baseline-first" if rng.random() < 0.5 else "candidate-first"
    a, b = (baseline, candidate) if order == "baseline-first" else (candidate, baseline)
    body = build_judge_request(cfg.judge_provider, cfg.judge_model, question, a, b)
    cost = 0.0
    label = None
    for _ in range(2):  # one retry for an unparseable reply
        r = call(client, cfg, cfg.judge_provider, body, key, prices)
        cost += r.cost_usd
        if r.status == 0 or r.status >= 400:
            raise JudgeUnavailable(f"judge call failed with status {r.status}")
        label = parse_label(r.text)
        if label:
            break
    return normalize(label or "judge-error", order), order, cost
