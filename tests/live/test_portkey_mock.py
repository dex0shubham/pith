"""Runs the open-source Portkey gateway with default.webhook hooks pointing at pith, against a mock upstream.

Run: OPTIMIZER_PORTKEY_LIVE=1 .venv/bin/pytest tests/live/test_portkey_mock.py -v
Needs Node (`npx`) and a free port 8787: the gateway's CLI always binds 8787 and ignores PORT/--port. Skipped otherwise.
No API key and no network beyond the npm download of @portkey-ai/gateway.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from pith import db
from pith.config import Config
from pith.proxy import create_app

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_PORTKEY_LIVE") != "1" or shutil.which("npx") is None,
                                reason="set OPTIMIZER_PORTKEY_LIVE=1 and install Node")
GATEWAY = "@portkey-ai/gateway@1.15.2"
GATEWAY_PORT = 8787
SEEN = []


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append((self.path, body))
        out = {"id": "c", "object": "chat.completion", "model": body["model"],
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
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


def port_busy(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def stack(tmp_path):
    if port_busy(GATEWAY_PORT):
        pytest.skip(f"port {GATEWAY_PORT} is busy and the Portkey gateway CLI cannot be moved")
    import uvicorn

    up = free_port()
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", up), Upstream).serve_forever, daemon=True).start()
    db_path = str(tmp_path / "pith.db")
    pith_port = free_port()
    app = create_app(Config(db_path=db_path, sample_rate=1.0, listen=f"127.0.0.1:{pith_port}"), db.connect(db_path))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=pith_port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    gw = subprocess.Popen(["npx", "-y", GATEWAY], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(240):  # the first run downloads the package; allow two minutes
            try:
                if httpx.get(f"http://127.0.0.1:{GATEWAY_PORT}/", timeout=1).status_code < 500:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("portkey gateway did not start")
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{pith_port}/optimizer/health", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        hook = f"http://127.0.0.1:{pith_port}/optimizer/portkey"
        config = {"provider": "openai", "api_key": "sk-fake", "custom_host": f"http://127.0.0.1:{up}",
                  "before_request_hooks": [{"type": "mutator", "id": "pith-before",
                                            "checks": [{"id": "default.webhook", "parameters": {"webhookURL": hook}}]}],
                  "after_request_hooks": [{"type": "guardrail", "id": "pith-after", "deny": False,
                                           "checks": [{"id": "default.webhook", "parameters": {"webhookURL": hook}}]}]}
        yield f"http://127.0.0.1:{GATEWAY_PORT}", json.dumps(config), db_path
    finally:
        os.killpg(os.getpgid(gw.pid), signal.SIGTERM)
        try:
            gw.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(gw.pid), signal.SIGKILL)
        server.should_exit = True


def test_portkey_hooks_record_and_apply_pins(stack):
    base, config, db_path = stack
    body = {"model": "gpt-4o", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Where is my order?"}], "max_tokens": 50}
    hdrs = {"x-portkey-config": config, "x-portkey-metadata": json.dumps({"pith_route": "pk"}), "content-type": "application/json"}
    r = httpx.post(base + "/v1/chat/completions", json=body, headers=hdrs, timeout=30)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Hello", r.text
    assert r.json()["hook_results"]["before_request_hooks"][0]["verdict"] is True
    time.sleep(0.5)
    conn = db.connect(db_path)
    assert db.get_route(conn, "pk")["provider"] == "portkey"
    row = conn.execute("SELECT profile, input_tokens, output_tokens, stop_reason FROM requests").fetchone()
    assert tuple(row) == ("P0", 12, 5, "stop")
    assert SEEN[-1][1]["messages"][-1]["content"] == "Where is my order?"
    db.set_pin(conn, "pk", "P2")
    conn.close()
    r = httpx.post(base + "/v1/chat/completions", json=body, headers=hdrs, timeout=30)
    assert r.status_code == 200 and r.json()["hook_results"]["before_request_hooks"][0]["transformed"] is True
    time.sleep(0.5)
    sent = SEEN[-1][1]["messages"][-1]["content"]
    assert sent[0] == {"type": "text", "text": "Where is my order?"} and sent[1]["text"].startswith("Answer directly.")
    conn = db.connect(db_path)
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == body
