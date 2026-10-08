import random

import httpx
import pytest

from optimizer.config import Config
from optimizer.judge import (JUDGE_PROMPT_VERSION, LABELS, SYSTEM_PROMPT, JudgeUnavailable, build_judge_request,
                             judge, last_user_text, normalize, parse_label)


def test_constants():
    assert JUDGE_PROMPT_VERSION == "v1"
    assert LABELS == ("equivalent", "missing-info", "contradiction", "format-broken")
    assert "exactly one label" in SYSTEM_PROMPT


def test_last_user_text_shapes_and_truncation():
    assert last_user_text("anthropic", {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                                     {"role": "user", "content": [{"type": "text", "text": "c"}]}]}) == "c"
    assert last_user_text("openai", {"messages": [{"role": "user", "content": "q"}]}) == "q"
    assert last_user_text("openai", {"input": "plain"}) == "plain"
    assert last_user_text("openai", {"input": [{"role": "user", "content": [{"type": "input_text", "text": "it"}]}]}) == "it"
    assert last_user_text("anthropic", {"messages": [{"role": "user", "content": "x" * 5000}]}) == "x" * 4000
    assert last_user_text("anthropic", {"messages": []}) == ""


def test_build_request_both_providers():
    a = build_judge_request("anthropic", "claude-sonnet-5-5", "Q", "A1", "B1")
    assert a["model"] == "claude-sonnet-5-5" and a["system"] == SYSTEM_PROMPT and a["max_tokens"] >= 256
    assert a["output_config"] == {"effort": "low"}
    user = a["messages"][0]["content"]
    assert "REQUEST:\nQ" in user and "ANSWER A:\nA1" in user and "ANSWER B:\nB1" in user
    for lab in LABELS:
        assert lab in user
    o = build_judge_request("openai", "gpt-5", "Q", "A1", "B1")
    assert o["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT} and o["messages"][1]["role"] == "user"
    assert "max_completion_tokens" in o and "max_tokens" not in o


def test_parse_label_first_match_case_insensitive():
    assert parse_label("Equivalent") == "equivalent"
    assert parse_label("Label: missing-info. Also contradiction.") == "missing-info"
    assert parse_label("I think it is a CONTRADICTION") == "contradiction"
    assert parse_label("format-broken") == "format-broken"
    assert parse_label("nothing here") is None
    assert parse_label("") is None


def test_normalize_by_order():
    assert normalize("equivalent", "baseline-first") == "equivalent"
    assert normalize("missing-info", "baseline-first") == "missing-info"
    assert normalize("missing-info", "candidate-first") == "extra-info"
    assert normalize("format-broken", "candidate-first") == "judge-error"
    assert normalize("contradiction", "candidate-first") == "contradiction"
    assert normalize("judge-error", "candidate-first") == "judge-error"


def _client(replies):
    it = iter(replies)
    def h(req):
        return next(it)(req)
    return httpx.Client(transport=httpx.MockTransport(h))


def anth(text):
    return lambda req: httpx.Response(200, json={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                                                 "usage": {"input_tokens": 300, "output_tokens": 3}})


def test_judge_orders_randomize_and_cost_accumulates():
    cfg = Config()
    seen = []
    def h(req):
        seen.append(req.content.decode())
        return httpx.Response(200, json={"content": [{"type": "text", "text": "equivalent"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 300, "output_tokens": 3}})
    client = httpx.Client(transport=httpx.MockTransport(h))
    orders = set()
    for seed in range(12):
        label, order, cost = judge(client, cfg, {"anthropic": "k"}, "Q", "BASE", "CAND", random.Random(seed))
        assert label == "equivalent" and cost == (300 * 2 + 3 * 10) / 1e6
        orders.add(order)
        body = seen[-1]
        if order == "baseline-first":
            assert body.index("ANSWER A:\\nBASE") < body.index("ANSWER B:\\nCAND")
        else:
            assert body.index("ANSWER A:\\nCAND") < body.index("ANSWER B:\\nBASE")
    assert orders == {"baseline-first", "candidate-first"}


def test_judge_retries_unparseable_once_then_judge_error():
    client = _client([anth("hmm"), anth("still nothing")])
    label, _, cost = judge(client, Config(), {"anthropic": "k"}, "Q", "B", "C", random.Random(1))
    assert label == "judge-error" and cost == 2 * (300 * 2 + 3 * 10) / 1e6
    client = _client([anth("??"), anth("missing-info")])
    label, order, _ = judge(client, Config(), {"anthropic": "k"}, "Q", "B", "C", random.Random(1))
    assert label == normalize("missing-info", order)


def test_judge_unavailable_on_error_or_transport_failure():
    client = _client([lambda req: httpx.Response(500, json={})])
    with pytest.raises(JudgeUnavailable):
        judge(client, Config(), {"anthropic": "k"}, "Q", "B", "C", random.Random(1))
    def boom(req):
        raise httpx.ConnectError("x", request=req)
    with pytest.raises(JudgeUnavailable):
        judge(_client([boom]), Config(), {"anthropic": "k"}, "Q", "B", "C", random.Random(1))
    with pytest.raises(JudgeUnavailable):
        judge(_client([]), Config(), {}, "Q", "B", "C", random.Random(1))  # no judge key
