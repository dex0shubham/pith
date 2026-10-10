"""Boots Headroom's real proxy app with the pith extensions against a mock upstream: no API key, no network.

Run: .venv/bin/pip install headroom-ai && .venv/bin/pip install -e . && OPTIMIZER_HEADROOM_LIVE=1 .venv/bin/pytest tests/live/test_headroom_mock.py -v
Headroom is not a pith dependency and is not installed in CI, so this file is skipped unless the flag is set.
The second pip command re-registers pith's entry points so Headroom can discover `pith` by name.
"""
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from pith import db

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_HEADROOM_LIVE") != "1", reason="set OPTIMIZER_HEADROOM_LIVE=1")

SYSTEM = "You classify support tickets. " * 40
SEEN = []


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append((self.path, body))
        anthropic, stream = self.path.startswith("/v1/messages"), body.get("stream")
        if stream:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            if anthropic:
                events = [("message_start", {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": body["model"], "content": [], "usage": {"input_tokens": 11, "output_tokens": 1}}}),
                          ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                          ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
                          ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                          ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}}),
                          ("message_stop", {"type": "message_stop"})]
                for name, ev in events:
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(ev)}\n\n".encode())
            else:
                for c in [{"id": "c", "object": "chat.completion.chunk", "model": body["model"], "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
                          {"id": "c", "object": "chat.completion.chunk", "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                          {"id": "c", "object": "chat.completion.chunk", "model": body["model"], "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}]:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            return
        if anthropic:
            out = {"id": "m", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "end_turn",
                   "content": [{"type": "text", "text": "Hello"}], "usage": {"input_tokens": 11, "output_tokens": 5}}
        else:
            out = {"id": "c", "object": "chat.completion", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 12, "completion_tokens": 5}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def headroom(tmp_path, monkeypatch):
    """start(cache_enabled=False) -> (base_url, db_path): Headroom's app with both pith extensions found by name."""
    create_app = pytest.importorskip("headroom.proxy.server").create_app
    ProxyConfig = pytest.importorskip("headroom.proxy.models").ProxyConfig
    import uvicorn

    from pith.headroom import RUNTIME

    monkeypatch.setattr(RUNTIME, "config", None)  # the process-wide runtime would keep an earlier test's database
    monkeypatch.setattr(RUNTIME, "conn", None)
    monkeypatch.setenv("HEADROOM_BEACON", "off")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("HEADROOM_PIPELINE_EXTENSIONS", "pith")  # entry-point discovery, as an operator enables it
    monkeypatch.setenv("OPTIMIZER_DB_PATH", str(tmp_path / "pith.db"))
    monkeypatch.setenv("OPTIMIZER_SAMPLE_RATE", "1")
    servers = []

    def start(cache_enabled=False):
        up = free_port()
        threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", up), Upstream).serve_forever, daemon=True).start()
        cfg = ProxyConfig(anthropic_api_url=f"http://127.0.0.1:{up}", openai_api_url=f"http://127.0.0.1:{up}",
                          proxy_extensions=["pith"], discover_pipeline_extensions=True,
                          cache_enabled=cache_enabled, cost_tracking_enabled=False, subscription_tracking_enabled=False,
                          periodic_toin_stats_enabled=False, license_report_interval=10**6)
        app = create_app(cfg)
        assert any(getattr(getattr(m, "cls", None), "__name__", None) == "PithMiddleware" for m in app.user_middleware), \
            "Headroom did not install the pith proxy extension: run `pip install -e .` so the entry point is registered"
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        servers.append(server)
        threading.Thread(target=server.run, daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                httpx.get(base + "/health", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail("headroom did not start")
        return base, str(tmp_path / "pith.db")

    yield start
    for server in servers:
        server.should_exit = True


CHAT = {"model": "gpt-4o", "max_tokens": 50, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Where is my order?"}]}
MSG = {"model": "claude-opus-5-5", "max_tokens": 50, "system": SYSTEM, "messages": [{"role": "user", "content": "Where is my order?"}]}
JSON = {"content-type": "application/json"}
OA = {**JSON, "x-optimizer-route": "hr-oa", "authorization": "Bearer sk-x"}
AN = {**JSON, "x-optimizer-route": "hr-an", "x-api-key": "sk-ant-x", "anthropic-version": "2023-06-01"}


def rows(conn, route):
    return [tuple(r) for r in conn.execute("SELECT profile, input_tokens, output_tokens, stop_reason FROM requests "
                                           "WHERE route_key = ? ORDER BY id", (route,)).fetchall()]


def test_pith_inside_headroom_records_and_applies_pins(headroom):
    base, db_path = headroom()
    for path, body, hdr in [("/v1/chat/completions", CHAT, OA), ("/v1/chat/completions", dict(CHAT, stream=True, stream_options={"include_usage": True}), OA),
                            ("/v1/messages", MSG, AN), ("/v1/messages", dict(MSG, stream=True), AN)]:
        r = httpx.post(base + path, json=body, headers=hdr, timeout=30)
        assert r.status_code == 200, r.text
    time.sleep(0.5)
    conn = db.connect(db_path)
    assert rows(conn, "hr-oa") == [("P0", 12, 5, "stop"), ("P0", 12, 5, "stop")]
    assert rows(conn, "hr-an") == [("P0", 11, 5, "end_turn"), ("P0", 11, 5, "end_turn")]
    assert db.get_route(conn, "hr-oa")["injection_form"] == db.get_route(conn, "hr-an")["injection_form"] == "user_text"
    assert db.get_route(conn, "hr-oa")["provider"] == "openai" and db.get_route(conn, "hr-an")["provider"] == "anthropic"
    db.set_pin(conn, "hr-an", "P2")
    conn.close()
    assert httpx.post(base + "/v1/messages", json=MSG, headers=AN, timeout=30).status_code == 200
    time.sleep(0.5)
    path, sent = SEEN[-1]
    assert sent["system"] == SYSTEM and sent["messages"][-1]["content"][-1]["text"].startswith("Answer directly.")
    conn = db.connect(db_path)
    assert rows(conn, "hr-an")[-1][0] == "P2"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == MSG
    db.set_pin(conn, "hr-an", "P4")
    conn.close()
    assert httpx.post(base + "/v1/messages", json=MSG, headers=AN, timeout=30).status_code == 200
    time.sleep(0.5)
    path, sent = SEEN[-1]
    assert sent["output_config"] == {"effort": "low"} and sent["messages"][-1]["content"][-1]["text"].startswith("Answer directly.")
    conn = db.connect(db_path)
    assert rows(conn, "hr-an")[-1][0] == "P4" and rows(conn, "hr-oa") == [("P0", 12, 5, "stop")] * 2


def test_headroom_response_cache_hits_are_not_recorded(headroom):
    base, db_path = headroom(cache_enabled=True)
    before = len(SEEN)
    for _ in range(3):
        assert httpx.post(base + "/v1/messages", json=MSG, headers=AN, timeout=30).status_code == 200
    time.sleep(0.5)
    assert len(SEEN) - before == 1  # Headroom served two of the three from its cache
    assert rows(db.connect(db_path), "hr-an") == [("P0", 11, 5, "end_turn")]
