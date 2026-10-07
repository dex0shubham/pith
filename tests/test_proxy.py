import json

import httpx
import pytest

from optimizer import db
from optimizer.config import Config, RouteConfig
from optimizer.proxy import create_app

ANTH_REQ = {"model": "claude-opus-5-5", "max_tokens": 50, "system": "S", "messages": [{"role": "user", "content": "q"}]}
ANTH_RESP = {"id": "m1", "type": "message", "stop_reason": "end_turn",
             "content": [{"type": "text", "text": "A"}],
             "usage": {"input_tokens": 9, "output_tokens": 4, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}


UPSTREAM_BYTES = json.dumps(ANTH_RESP, separators=(",", ":")).encode()  # differs from default json.dumps spacing


def make(config=None, handler=None, seen=None):
    seen = [] if seen is None else seen

    def default_handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, content=UPSTREAM_BYTES, headers={"x-upstream": "1", "content-type": "application/json"})

    conn = db.connect(":memory:")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler or default_handler))
    app = create_app(config or Config(sample_rate=1.0), conn, client=client)
    return app, conn, seen


async def post(app, path, body, headers=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, content=json.dumps(body).encode(), headers={"content-type": "application/json",
                                                                               "x-api-key": "sk-secret", **(headers or {})})


@pytest.mark.anyio
async def test_passthrough_is_byte_identical_and_forwards_auth_only_upstream():
    app, conn, seen = make()
    raw = json.dumps(ANTH_REQ).encode()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/messages", content=raw, headers={"content-type": "application/json", "x-api-key": "sk-secret"})
    assert r.status_code == 200 and r.content == UPSTREAM_BYTES and r.headers["x-upstream"] == "1"
    assert seen[0].content == raw
    assert seen[0].headers["x-api-key"] == "sk-secret"
    assert str(seen[0].url) == "https://api.anthropic.com/v1/messages"


@pytest.mark.anyio
async def test_records_route_and_usage_and_samples_body():
    app, conn, seen = make()
    await post(app, "/v1/messages", ANTH_REQ)
    route = conn.execute("SELECT * FROM routes").fetchone()
    assert route["provider"] == "anthropic" and route["model"] == "claude-opus-5-5" and route["status"] == "observing"
    req = conn.execute("SELECT * FROM requests").fetchone()
    assert (req["profile"], req["input_tokens"], req["output_tokens"], req["stop_reason"]) == ("P0", 9, 4, "end_turn")
    body = conn.execute("SELECT * FROM bodies").fetchone()
    assert "sk-secret" not in body["request_json"] and json.loads(body["request_json"]) == ANTH_REQ


@pytest.mark.anyio
async def test_route_header_override_names_route():
    app, conn, _ = make()
    await post(app, "/v1/messages", ANTH_REQ, {"X-Optimizer-Route": "billing"})
    assert conn.execute("SELECT key FROM routes").fetchone()["key"] == "billing"


@pytest.mark.anyio
async def test_unknown_path_and_malformed_json_fail_open():
    app, conn, seen = make()
    r = await post(app, "/v1/models", {"x": 1})
    assert r.status_code == 200 and json.loads(seen[0].content) == {"x": 1}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/messages", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 200 and seen[1].content == b"{not json"
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


@pytest.mark.anyio
async def test_upstream_error_status_is_passed_through():
    def h(req):
        return httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "7"})
    app, conn, _ = make(handler=h)
    r = await post(app, "/v1/messages", ANTH_REQ)
    assert r.status_code == 429 and r.headers["retry-after"] == "7" and r.json() == {"error": "slow down"}


@pytest.mark.anyio
async def test_kill_switches_force_p0_and_bypass_skips_recording():
    app, conn, seen = make()
    await post(app, "/v1/messages", ANTH_REQ)
    key = conn.execute("SELECT key FROM routes").fetchone()["key"]
    db.set_pin(conn, key, "P2")
    await post(app, "/v1/messages", ANTH_REQ, {"X-Optimizer": "off"})
    assert json.loads(seen[-1].content) == ANTH_REQ
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P0"
    n = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    await post(app, "/v1/messages", ANTH_REQ, {"X-Optimizer": "bypass"})
    assert json.loads(seen[-1].content) == ANTH_REQ
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == n
    app2, conn2, seen2 = make(Config(enabled=False, sample_rate=0))
    await post(app2, "/v1/messages", ANTH_REQ)
    key2 = conn2.execute("SELECT key FROM routes").fetchone()["key"]
    db.set_pin(conn2, key2, "P2")
    await post(app2, "/v1/messages", ANTH_REQ)
    assert json.loads(seen2[-1].content) == ANTH_REQ
    app3, conn3, seen3 = make(Config(sample_rate=0, routes={"billing": RouteConfig(enabled=False)}))
    await post(app3, "/v1/messages", ANTH_REQ, {"X-Optimizer-Route": "billing"})
    db.set_pin(conn3, "billing", "P2")
    await post(app3, "/v1/messages", ANTH_REQ, {"X-Optimizer-Route": "billing"})
    assert json.loads(seen3[-1].content) == ANTH_REQ


@pytest.mark.anyio
async def test_health():
    app, _, _ = make()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/optimizer/health")).json() == {"ok": True}


@pytest.mark.anyio
async def test_non_post_is_forwarded_verbatim_with_query():
    app, conn, seen = make()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/models?limit=5&after=x", headers={"x-api-key": "sk-secret"})
    assert r.status_code == 200 and r.content == UPSTREAM_BYTES
    assert seen[0].method == "GET" and str(seen[0].url) == "https://api.openai.com/v1/models?limit=5&after=x"
    assert seen[0].headers["x-api-key"] == "sk-secret"
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


@pytest.mark.anyio
async def test_openai_chat_uses_openai_upstream():
    app, conn, seen = make()
    await post(app, "/v1/chat/completions", {"model": "gpt-5", "messages": [{"role": "user", "content": "q"}]})
    assert str(seen[0].url) == "https://api.openai.com/v1/chat/completions"
    assert conn.execute("SELECT provider FROM routes").fetchone()["provider"] == "openai"


@pytest.mark.anyio
async def test_upstream_connect_error_returns_502():
    def h(req):
        raise httpx.ConnectError("boom", request=req)
    app, conn, _ = make(handler=h)
    r = await post(app, "/v1/messages", ANTH_REQ)
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_unreachable"
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


@pytest.mark.anyio
async def test_upstream_timeout_returns_504():
    def h(req):
        raise httpx.ReadTimeout("slow", request=req)
    app, _, _ = make(handler=h)
    r = await post(app, "/v1/messages", ANTH_REQ)
    assert r.status_code == 504 and r.json()["error"]["type"] == "upstream_timeout"


SSE = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":11,"cache_read_input_tokens":3,"cache_creation_input_tokens":0}}}\n\n'
       b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
       b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}\n\n')


@pytest.mark.anyio
async def test_streaming_passthrough_relays_chunks_and_records_after_end():
    async def gen():
        for i in range(0, len(SSE), 37):
            yield SSE[i:i + 37]

    def h(req):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())
    app, conn, _ = make(handler=h)
    body = dict(ANTH_REQ, stream=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/messages", content=json.dumps(body).encode(),
                            headers={"content-type": "application/json"}) as r:
            assert r.status_code == 200 and r.headers["content-type"] == "text/event-stream"
            got = b"".join([chunk async for chunk in r.aiter_raw()])
    assert got == SSE
    req = conn.execute("SELECT * FROM requests").fetchone()
    assert (req["input_tokens"], req["output_tokens"], req["cache_read"], req["stop_reason"]) == (11, 6, 3, "end_turn")
