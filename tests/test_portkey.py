import json
import logging

import httpx
import pytest

from pith import db
from pith.config import Config
from pith.portkey import handle
from pith.proxy import create_app

BODY = {"model": "gpt-4o", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], "max_tokens": 50}
RESP = {"id": "c", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5}}


def payload(event, body=BODY, resp=None, meta=None, request_type="chatComplete", provider="openai"):
    """What Portkey's default.webhook POSTs (spec §2)."""
    after = event == "afterRequestHook"
    return {"eventType": event, "provider": provider, "requestType": request_type, "metadata": meta or {},
            "request": {"json": body, "text": "q", "isStreamingRequest": bool(body.get("stream")), "isTransformed": False},
            "response": {"json": resp if after else {}, "text": "", "statusCode": 200 if after else None, "isTransformed": False}}


def setup():
    return Config(sample_rate=1.0), db.connect(":memory:")


class Boom:
    def execute(self, *a, **k):
        raise RuntimeError("x-portkey-api-key: sk-secret")


def test_before_hook_registers_the_route_and_leaves_p0_untouched():
    cfg, conn = setup()
    assert handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"})) == {"verdict": True}
    route = db.get_route(conn, "r")
    assert route["provider"] == "portkey" and route["model"] == "gpt-4o"
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    key = handle(cfg, conn, payload("beforeRequestHook")) and conn.execute("SELECT key FROM routes WHERE key != 'r'").fetchone()[0]
    assert key.startswith("portkey:gpt-4o:")


def test_before_hook_transforms_a_pinned_route_and_honours_off_and_bypass():
    cfg, conn = setup()
    handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"}))
    db.set_pin(conn, "r", "P2")
    out = handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"}))
    new = out["transformedData"]["request"]["json"]
    assert out["verdict"] is True and new["messages"][-1]["content"][0] == {"type": "text", "text": "q"}
    assert new["messages"][-1]["content"][1]["text"].startswith("Answer directly.")
    assert new["messages"][0] == BODY["messages"][0] and new["max_tokens"] == 50 and new["model"] == "gpt-4o"
    assert handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r", "pith": "off"})) == {"verdict": True}
    assert handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r", "pith_bypass": True})) == {"verdict": True}


def test_non_chat_and_malformed_payloads_are_untouched():
    cfg, conn = setup()
    assert handle(cfg, conn, payload("beforeRequestHook", request_type="messages")) == {"verdict": True}
    assert handle(cfg, conn, payload("beforeRequestHook", body={"input": "q"})) == {"verdict": True}
    assert handle(cfg, conn, None) == {"verdict": True}
    assert handle(cfg, conn, {"eventType": "beforeRequestHook", "requestType": "chatComplete"}) == {"verdict": True}
    assert conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


def test_after_hook_records_p0_and_strips_the_shape_for_a_pinned_route():
    cfg, conn = setup()
    assert handle(cfg, conn, payload("afterRequestHook", resp=RESP, meta={"pith_route": "r"})) == {"verdict": True}
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["route_key"], row["profile"], row["input_tokens"], row["output_tokens"], row["stop_reason"], row["latency_ms"]) == \
        ("r", "P0", 12, 5, "stop", 0)
    assert json.loads(conn.execute("SELECT request_json FROM bodies").fetchone()[0]) == BODY
    db.set_pin(conn, "r", "P2")
    shaped = handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"}))["transformedData"]["request"]["json"]
    handle(cfg, conn, payload("afterRequestHook", body=shaped, resp=RESP, meta={"pith_route": "r"}))
    row = conn.execute("SELECT * FROM requests ORDER BY id DESC").fetchone()
    assert row["profile"] == "P2"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == BODY
    assert json.loads(conn.execute("SELECT response_json FROM bodies ORDER BY id DESC").fetchone()[0]) == RESP
    assert db.get_route(conn, "r")["pinned_profile"] == "P2"


def test_after_hook_ignores_streams_bypass_and_failures(caplog):
    cfg, conn = setup()
    assert handle(cfg, conn, payload("afterRequestHook", body=dict(BODY, stream=True), resp=None)) == {"verdict": True}
    assert handle(cfg, conn, payload("afterRequestHook", resp=RESP, meta={"pith_bypass": True})) == {"verdict": True}
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    with caplog.at_level(logging.WARNING, logger="pith.portkey"):
        assert handle(cfg, Boom(), payload("afterRequestHook", resp=RESP)) == {"verdict": True}
        assert handle(cfg, Boom(), payload("beforeRequestHook")) == {"verdict": True}
    assert "sk-secret" not in caplog.text and "RuntimeError" in caplog.text


def test_bypass_flag_is_parsed_as_a_boolean():
    for val, bypassed in (("false", False), ("0", False), (" No ", False), ("true", True), ("1", True), (True, True)):
        cfg, conn = setup()
        handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r", "pith_bypass": val}))
        assert (db.get_route(conn, "r") is None) == bypassed, val


def test_after_hook_records_nothing_when_the_pin_changed_between_hooks():
    cfg, conn = setup()
    handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"}))
    db.set_pin(conn, "r", "P2")
    shaped = handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"}))["transformedData"]["request"]["json"]
    db.set_pin(conn, "r", "P0", status="reverted")
    assert handle(cfg, conn, payload("afterRequestHook", body=shaped, resp=RESP, meta={"pith_route": "r"})) == {"verdict": True}
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


def test_open_endpoint_warns_at_startup_only_when_unauthenticated_and_not_loopback(caplog):
    with caplog.at_level(logging.WARNING, logger="pith"):
        make_app(Config(webhook_token="t"))
        make_app(Config(listen="127.0.0.1:8787"))
        assert "unauthenticated" not in caplog.text
        make_app(Config())
    assert caplog.text.count("/optimizer/portkey accepts unauthenticated hook posts") == 1


def make_app(config):
    return create_app(config, db.connect(":memory:"),
                      client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))))


async def post(app, body, headers=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.post("/optimizer/portkey", content=body, headers={"content-type": "application/json", **(headers or {})})


@pytest.mark.anyio
async def test_route_answers_hooks_requires_the_token_when_set_and_tolerates_bad_json():
    app = make_app(Config(sample_rate=1.0))
    r = await post(app, json.dumps(payload("beforeRequestHook", meta={"pith_route": "r"})))
    assert r.status_code == 200 and r.json() == {"verdict": True}
    r = await post(app, b"{not json")
    assert r.status_code == 200 and r.json() == {"verdict": True}
    app = make_app(Config(webhook_token="t0k"))
    assert (await post(app, json.dumps(payload("beforeRequestHook")))).status_code == 401
    assert (await post(app, json.dumps(payload("beforeRequestHook")), {"authorization": "Bearer nope"})).status_code == 401
    assert (await post(app, json.dumps(payload("beforeRequestHook")), {"authorization": "Bearer tøk".encode()})).status_code == 401
    assert (await post(app, json.dumps(payload("beforeRequestHook")), {"authorization": "Bearer t0k"})).status_code == 200
