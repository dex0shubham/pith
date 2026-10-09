import json

import httpx
import pytest

from pith.config import Config
from pith.replay import Reply, auth_headers, call, cost_of, endpoint_for, response_text, sse_text, stored_response_text
from pith.usage import Usage

ANTH = {"id": "m", "stop_reason": "end_turn", "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}],
        "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}
CHAT = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
RESP = {"object": "response", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "yo"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 1}}


def test_endpoint_and_headers():
    assert endpoint_for("anthropic", {"messages": []}) == "/v1/messages"
    assert endpoint_for("openai", {"messages": []}) == "/v1/chat/completions"
    assert endpoint_for("openai", {"input": "q"}) == "/v1/responses"
    assert endpoint_for("litellm", {"messages": []}) == "/v1/chat/completions"
    assert auth_headers("anthropic", "k") == {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    assert auth_headers("openai", "k") == {"authorization": "Bearer k"}
    assert auth_headers("litellm", "k") == {"authorization": "Bearer k"}


def test_response_text_all_shapes():
    assert response_text("anthropic", ANTH) == "Hello"
    assert response_text("openai", CHAT) == "hi"
    assert response_text("openai", RESP) == "yo"
    assert response_text("openai", {"object": "response", "output_text": "direct"}) == "direct"
    assert response_text("anthropic", {}) == ""


def test_sse_and_stored_text():
    sse = ('event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"a"}}\n\n'
           'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"b"}}\n\n')
    assert sse_text("anthropic", sse) == "ab"
    chat = 'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: {"choices":[{"delta":{"content":"y"}}]}\n\ndata: [DONE]\n\n'
    assert sse_text("openai", chat) == "xy"
    assert stored_response_text("anthropic", json.dumps(ANTH)) == "Hello"
    assert stored_response_text("anthropic", sse) == "ab"
    assert stored_response_text("anthropic", "not json at all") == ""


def test_cost_of():
    assert cost_of("claude-opus-5-5", Usage(100, 10, 0, 0, "end_turn"), None) == (100 * 4 + 10 * 20) / 1e6
    assert cost_of("unknown", Usage(100, 10, 0, 0, "end_turn"), None) == 0.0
    assert cost_of("x", Usage(100, 10, 0, 0, "end_turn"), {"x": (1.0, 1.0)}) == 110 / 1e6
    assert cost_of("claude-opus-5-5", Usage(None, None, None, None, None), None) == 0.0
    cached = Usage(100, 10, 1000, 200, "end_turn")
    assert cost_of("claude-opus-5-5", cached, None) == pytest.approx((100 * 4 + 1000 * 4 * 0.1 + 200 * 4 * 1.25 + 10 * 20) / 1e6)
    assert cost_of("claude-opus-5-5", cached, None, "openai") == (100 * 4 + 10 * 20) / 1e6


def test_call_strips_stream_and_prices():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json=ANTH)
    client = httpx.Client(transport=httpx.MockTransport(h))
    r = call(client, Config(), "anthropic", {"model": "claude-opus-5-5", "stream": True,
                                        "stream_options": {"include_usage": True}, "messages": []}, "k")
    assert isinstance(r, Reply) and r.status == 200 and r.text == "Hello"
    assert r.usage.output_tokens == 10 and r.cost_usd == (100 * 4 + 10 * 20) / 1e6
    assert "stream" not in json.loads(seen[0].content) and "stream_options" not in json.loads(seen[0].content)
    assert seen[0].headers["x-api-key"] == "k" and str(seen[0].url) == "https://api.anthropic.com/v1/messages"


def test_call_retries_429_once_then_returns():
    n = []
    def h(req):
        n.append(1)
        return httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow"}) if len(n) == 1 else httpx.Response(200, json=CHAT)
    slept = []
    client = httpx.Client(transport=httpx.MockTransport(h))
    r = call(client, Config(), "openai", {"model": "gpt-5", "messages": []}, "k", sleep=slept.append)
    assert r.status == 200 and r.text == "hi" and slept == [7.0] and len(n) == 2
    n.clear(); slept.clear()
    def always(req):
        n.append(1)
        return httpx.Response(429, headers={"retry-after": "500"}, json={})
    r = call(httpx.Client(transport=httpx.MockTransport(always)), Config(), "openai", {"model": "gpt-5", "messages": []}, "k", sleep=slept.append)
    assert r.status == 429 and slept == [60.0] and len(n) == 2


def test_call_transport_error_is_status_zero():
    def h(req):
        raise httpx.ConnectError("down", request=req)
    r = call(httpx.Client(transport=httpx.MockTransport(h)), Config(), "anthropic", {"model": "m", "messages": []}, "k")
    assert r.status == 0 and r.body is None and r.text == "" and r.cost_usd == 0.0


def test_call_non_numeric_retry_after_does_not_raise():
    n, slept = [], []
    def h(req):
        n.append(1)
        return httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, json={}) if len(n) == 1 else httpx.Response(200, json=CHAT)
    r = call(httpx.Client(transport=httpx.MockTransport(h)), Config(), "openai", {"model": "gpt-5", "messages": []}, "k", sleep=slept.append)
    assert r.status == 200 and slept == [5.0]


def test_transport_failure_log_never_contains_header_values(caplog):
    def h(req):
        raise httpx.LocalProtocolError("Illegal header value b'\\rsk-ant-SECRET-KEY'")
    with caplog.at_level("WARNING", logger="pith.replay"):
        r = call(httpx.Client(transport=httpx.MockTransport(h)), Config(), "anthropic", {"model": "m", "messages": []}, "sk-ant-SECRET-KEY")
    assert r.status == 0
    assert "SECRET" not in caplog.text and "LocalProtocolError" in caplog.text


def test_call_through_litellm_uses_chat_endpoint_and_bypass_header():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json=CHAT)
    client = httpx.Client(transport=httpx.MockTransport(h))
    r = call(client, Config(litellm_upstream="http://l:4000"), "litellm", {"model": "mock", "messages": []}, "k", {"mock": (1.0, 2.0)})
    assert r.status == 200 and r.text == "hi" and r.cost_usd == (5 * 1.0 + 1 * 2.0) / 1e6
    assert str(seen[0].url) == "http://l:4000/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer k" and seen[0].headers["x-optimizer"] == "bypass"
    call(client, Config(), "openai", {"model": "m", "messages": []}, "k")
    assert "x-optimizer" not in seen[1].headers
