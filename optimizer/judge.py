"""The equivalence judge: one frozen prompt, one call, one label. Spec (Plan 2) §6."""
import random
import re

from optimizer.config import Config
from optimizer.replay import call

JUDGE_PROMPT_VERSION = "v1"
LABELS = ("equivalent", "missing-info", "contradiction", "format-broken")
SYSTEM_PROMPT = ("You compare two answers to the same request. Judge whether Answer B conveys every fact, decision and "
                 "required output that Answer A does, with no contradiction. Reply with exactly one label.")
USER_TEMPLATE = (
    "REQUEST:\n{question}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}\n\n"
    "Labels:\n"
    "equivalent - B conveys everything A does, nothing contradictory.\n"
    "missing-info - B omits a fact, decision or required output that A states.\n"
    "contradiction - B asserts something A denies, or vice versa.\n"
    "format-broken - B is empty, truncated, or not a usable answer.\n\n"
    "Reply with exactly one label.")
_LABEL_RE = re.compile("|".join(re.escape(l) for l in LABELS), re.IGNORECASE)


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
    return {"model": model, "max_completion_tokens": 256,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}


def parse_label(text: str) -> str | None:
    m = _LABEL_RE.search(text or "")
    return m.group(0).lower() if m else None


def normalize(label: str, order_ab: str) -> str:
    if order_ab == "candidate-first":
        # A=candidate, B=baseline: "B omits" means the candidate added claims; "B broken" means the baseline is unusable.
        return {"missing-info": "extra-info", "format-broken": "judge-error"}.get(label, label)
    return label


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
