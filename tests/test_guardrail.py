import json
import logging

import pytest

from pith import db
from pith.config import Config
from pith.guardrail import PithHooks

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
