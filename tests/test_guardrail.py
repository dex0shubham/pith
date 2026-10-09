import json
import logging

import pytest

from pith import db
from pith.config import Config
from pith.guardrail import PithHooks
from pith.replay import stored_response_text

BODY = {"model": "mock", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], "max_tokens": 50,
        "metadata": {"pith_route": "ignored"}}
CLEAN = {k: v for k, v in BODY.items() if k != "metadata"}


def req(body=BODY, headers=None):
    """What LiteLLM hands async_pre_call_hook on /v1/chat/completions (spec §2)."""
    hdrs = {"content-type": "application/json", **{k.lower(): v for k, v in (headers or {}).items()}}
    return {**{k: v for k, v in body.items() if k != "metadata"}, "litellm_call_id": "c1", "litellm_logging_obj": object(),
            "secret_fields": {}, "metadata": {**body.get("metadata", {}), "headers": hdrs, "user_api_key_hash": "h"},
            "proxy_server_request": {"url": "http://l/v1/chat/completions", "method": "POST", "headers": hdrs, "body": dict(body)}}


def hooks(config=None):
    conn = db.connect(":memory:")
    return PithHooks(config or Config(sample_rate=1.0), conn), conn


class Boom:
    def execute(self, *a, **k):
        raise RuntimeError("x-api-key: sk-secret")


@pytest.mark.anyio
async def test_pre_call_registers_route_and_stashes_original_body_at_p0():
    h, conn = hooks()
    data = req()
    out = await h.async_pre_call_hook({}, None, data, "acompletion")
    assert out is data
    route = conn.execute("SELECT * FROM routes").fetchone()
    assert route["provider"] == "litellm" and route["model"] == "mock" and route["key"].startswith("litellm:mock:")
    stash = data["metadata"]["pith"]
    assert stash["route"] == route["key"] and stash["profile"] == "P0" and stash["body"] == CLEAN
    assert {k: data[k] for k in CLEAN} == CLEAN and data["metadata"]["pith_route"] == "ignored"


@pytest.mark.anyio
async def test_pre_call_ignores_non_chat_calls_and_bypass():
    h, conn = hooks()
    data = req()
    await h.async_pre_call_hook({}, None, data, "aembedding")
    assert "pith" not in data["metadata"] and conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0
    data = req(headers={"X-Optimizer": "bypass"})
    await h.async_pre_call_hook({}, None, data, "acompletion")
    assert "pith" not in data["metadata"] and conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


@pytest.mark.anyio
async def test_pre_call_applies_p2_pin_as_user_text_and_off_forces_p0():
    h, conn = hooks()
    await h.async_pre_call_hook({}, None, req(headers={"X-Optimizer-Route": "billing"}), "acompletion")
    assert db.get_route(conn, "billing")["provider"] == "litellm"
    db.set_pin(conn, "billing", "P2")
    data = req(headers={"X-Optimizer-Route": "billing"})
    await h.async_pre_call_hook({}, None, data, "acompletion")
    last = data["messages"][-1]
    assert last["role"] == "user" and last["content"][0] == {"type": "text", "text": "q"}
    assert last["content"][1]["type"] == "text" and last["content"][1]["text"].startswith("Answer directly.")
    assert data["messages"][0] == BODY["messages"][0] and data["model"] == "mock" and data["max_tokens"] == 50
    assert data["metadata"]["pith"]["profile"] == "P2" and data["metadata"]["pith"]["body"] == CLEAN
    data = req(headers={"X-Optimizer-Route": "billing", "X-Optimizer": "off"})
    await h.async_pre_call_hook({}, None, data, "acompletion")
    assert data["messages"] == BODY["messages"] and data["metadata"]["pith"]["profile"] == "P0"


@pytest.mark.anyio
async def test_pre_call_without_proxy_snapshot_uses_data_minus_litellm_keys():
    h, conn = hooks()
    data = req()
    del data["proxy_server_request"]
    await h.async_pre_call_hook({}, None, data, "acompletion")
    assert data["metadata"]["pith"]["body"] == CLEAN


@pytest.mark.anyio
async def test_pre_call_fails_open_and_logs_only_class_name(caplog):
    h = PithHooks(Config(), Boom())
    data = req()
    with caplog.at_level(logging.WARNING, logger="pith.guardrail"):
        out = await h.async_pre_call_hook({}, None, data, "acompletion")
    assert out is data and data["messages"] == BODY["messages"] and "pith" not in data["metadata"]
    assert "sk-secret" not in caplog.text and "RuntimeError" in caplog.text


def test_lazy_config_and_connection_from_env(tmp_path):
    h = PithHooks(env={"OPTIMIZER_DB_PATH": str(tmp_path / "g.db"), "OPTIMIZER_SAMPLE_RATE": "1"})
    cfg, conn = h._ready()
    assert cfg.db_path == str(tmp_path / "g.db") and cfg.sample_rate == 1.0 and (tmp_path / "g.db").exists()
    assert h._ready() == (cfg, conn)


class FakeResponse:
    def __init__(self, d):
        self._d = d

    def model_dump(self):
        return dict(self._d)


CHAT_RESP = {"id": "x", "object": "chat.completion", "model": "gpt-4o",
             "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 20}}


async def chunks(items):
    for it in items:
        yield it


def chunk(text=None, finish=None, usage=None):
    return FakeResponse({"object": "chat.completion.chunk", "model": "gpt-4o",
                         "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}],
                         **({"usage": usage} if usage else {})})


class Rejected(Exception):
    def __init__(self, status):
        super().__init__("provider said no: x-api-key sk-secret")
        self.status_code = status


@pytest.mark.anyio
async def test_post_call_records_usage_profile_and_samples_body():
    h, conn = hooks()
    data = req()
    await h.async_pre_call_hook({}, None, data, "acompletion")
    resp = FakeResponse(CHAT_RESP)
    assert await h.async_post_call_success_hook(data, {}, resp) is resp
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["profile"], row["input_tokens"], row["output_tokens"], row["stop_reason"], row["estimated"]) == ("P0", 10, 20, "stop", 0)
    assert row["latency_ms"] >= 0
    body = conn.execute("SELECT * FROM bodies").fetchone()
    assert json.loads(body["request_json"]) == CLEAN and json.loads(body["response_json"]) == CHAT_RESP
    assert stored_response_text("openai", body["response_json"]) == "Hello"


@pytest.mark.anyio
async def test_post_call_without_stash_or_model_dump_records_nothing_and_fails_open():
    h, conn = hooks()
    await h.async_post_call_success_hook(req(), {}, FakeResponse(CHAT_RESP))
    await h.async_post_call_success_hook(req(), {}, {"not": "a model"})
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    data = req()
    data["metadata"]["pith"] = {"route": "r", "profile": "P0", "body": CLEAN, "t0": 0.0}
    resp = FakeResponse(CHAT_RESP)
    assert await PithHooks(Config(), Boom()).async_post_call_success_hook(data, {}, resp) is resp


@pytest.mark.anyio
async def test_streaming_passes_chunks_through_and_records_usage_from_last_chunk():
    h, conn = hooks()
    data = req(dict(BODY, stream=True))
    await h.async_pre_call_hook({}, None, data, "acompletion")
    src = [chunk("Hel"), chunk("lo"), chunk(finish="stop"), chunk(usage={"prompt_tokens": 14, "completion_tokens": 3})]
    got = [c async for c in h.async_post_call_streaming_iterator_hook({}, chunks(src), data)]
    assert got == src
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["profile"], row["input_tokens"], row["output_tokens"], row["stop_reason"], row["estimated"]) == ("P0", 14, 3, "stop", 0)
    stored = conn.execute("SELECT response_json FROM bodies").fetchone()[0]
    assert stored_response_text("openai", stored) == "Hello" and json.loads(stored)["choices"][0]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_streaming_without_usage_estimates_and_early_stop_records_nothing():
    h, conn = hooks()
    data = req(dict(BODY, stream=True))
    await h.async_pre_call_hook({}, None, data, "acompletion")
    src = [chunk("Hello world"), chunk(finish="stop")]
    [c async for c in h.async_post_call_streaming_iterator_hook({}, chunks(src), data)]
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert row["output_tokens"] >= 1 and row["estimated"] == 1 and row["input_tokens"] is None and row["stop_reason"] == "stop"
    data = req(dict(BODY, stream=True))
    await h.async_pre_call_hook({}, None, data, "acompletion")
    agen = h.async_post_call_streaming_iterator_hook({}, chunks(src), data)
    assert await agen.__anext__() is src[0]
    await agen.aclose()
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
    # garbage chunks and a broken connection never break the customer's stream
    bad = [FakeResponse({"choices": "nope"}), object(), chunk(finish="stop")]
    data = req(dict(BODY, stream=True))
    data["metadata"]["pith"] = {"route": "r", "profile": "P0", "body": CLEAN, "t0": 0.0}
    assert [c async for c in PithHooks(Config(), Boom()).async_post_call_streaming_iterator_hook({}, chunks(bad), data)] == bad


@pytest.mark.anyio
async def test_failure_hook_counts_once_per_request_and_reverts_after_three():
    h, conn = hooks()
    await h.async_pre_call_hook({}, None, req(headers={"X-Optimizer-Route": "r"}), "acompletion")
    db.set_pin(conn, "r", "P2")
    for i in range(3):
        data = req(headers={"X-Optimizer-Route": "r"})
        await h.async_pre_call_hook({}, None, data, "acompletion")
        assert data["metadata"]["pith"]["profile"] == "P2"
        await h.async_post_call_failure_hook(data, Rejected(400), {})
        await h.async_post_call_failure_hook(data, Rejected(400), {})  # LiteLLM fires it twice per failure
        assert db.get_route(conn, "r")["rejections"] == (i + 1 if i < 2 else 0)
    r = db.get_route(conn, "r")
    assert r["pinned_profile"] == "P0" and r["status"] == "reverted"
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


@pytest.mark.anyio
async def test_failure_hook_ignores_p0_non_4xx_missing_stash_and_fails_open(caplog):
    h, conn = hooks()
    await h.async_pre_call_hook({}, None, req(headers={"X-Optimizer-Route": "r"}), "acompletion")
    db.set_pin(conn, "r", "P2")
    data = req(headers={"X-Optimizer-Route": "r"})
    await h.async_pre_call_hook({}, None, data, "acompletion")
    await h.async_post_call_failure_hook(data, Rejected(500), {})
    await h.async_post_call_failure_hook(data, RuntimeError("boom"), {})
    await h.async_post_call_failure_hook(req(), Rejected(400), {})
    p0 = req(headers={"X-Optimizer-Route": "r", "X-Optimizer": "off"})
    await h.async_pre_call_hook({}, None, p0, "acompletion")
    await h.async_post_call_failure_hook(p0, Rejected(400), {})
    assert db.get_route(conn, "r")["rejections"] == 0 and db.get_route(conn, "r")["pinned_profile"] == "P2"
    data["metadata"]["pith"]["failed"] = False
    with caplog.at_level(logging.WARNING, logger="pith.guardrail"):
        await PithHooks(Config(), Boom()).async_post_call_failure_hook(data, Rejected(400), {})
    assert "sk-secret" not in caplog.text and "RuntimeError" in caplog.text


def test_module_imports_without_litellm_and_defines_guardrail_only_with_it():
    import importlib.util

    import pith.guardrail as g
    has = importlib.util.find_spec("litellm") is not None
    assert hasattr(g, "PithGuardrail") == has
    if has:
        mro = g.PithGuardrail.__mro__
        assert mro.index(PithHooks) < mro.index(g.CustomGuardrail)
        assert isinstance(g.PithGuardrail(guardrail_name="pith"), PithHooks)
    else:
        assert g.CustomGuardrail is None
