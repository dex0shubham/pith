"""Boots a real LiteLLM proxy with a mock model and the pith guardrail: no API key, no network.

Run: .venv/bin/pip install 'litellm[proxy]' && OPTIMIZER_LITELLM_LIVE=1 .venv/bin/pytest tests/live/test_litellm_mock.py -v
LiteLLM is not a pith dependency and is not installed in CI, so this file is skipped unless the flag is set.
"""
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from pith import db

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_LITELLM_LIVE") != "1", reason="set OPTIMIZER_LITELLM_LIVE=1")

CONFIG = """
model_list:
  - model_name: mock
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-fake
      mock_response: "Hello from mock"
guardrails:
  - guardrail_name: pith
    litellm_params:
      guardrail: pith.guardrail.PithGuardrail
      mode: [pre_call, post_call]
      default_on: true
general_settings:
  master_key: sk-test
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def proxy(tmp_path):
    pytest.importorskip("litellm")
    (tmp_path / "config.yaml").write_text(CONFIG)
    port = free_port()
    env = {**os.environ, "OPTIMIZER_DB_PATH": str(tmp_path / "pith.db"), "OPTIMIZER_SAMPLE_RATE": "1"}
    exe = os.path.join(os.path.dirname(sys.executable), "litellm")
    p = subprocess.Popen([exe, "--config", str(tmp_path / "config.yaml"), "--port", str(port), "--host", "127.0.0.1"],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(base + "/health/liveliness", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("litellm proxy did not start")
        yield base, str(tmp_path / "pith.db")
    finally:
        p.terminate()
        p.wait(timeout=10)


def post(base, body, headers=None):
    return httpx.post(base + "/v1/chat/completions", json=body, timeout=30,
                      headers={"authorization": "Bearer sk-test", **(headers or {})})


def streamed_text(sse):
    """LiteLLM's mock model streams the reply in 3-char chunks, so reassemble the deltas before matching."""
    chunks = [json.loads(line[6:]) for line in sse.splitlines() if line.startswith("data: {")]
    return "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])


def test_guardrail_records_applies_pins_and_honours_bypass(proxy):
    base, db_path = proxy
    body = {"model": "mock", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]}
    hdr = {"X-Optimizer-Route": "mock-route"}
    assert post(base, body, hdr).status_code == 200
    r = post(base, dict(body, stream=True, stream_options={"include_usage": True}), hdr)
    assert r.status_code == 200 and "Hello from mock" in streamed_text(r.text)
    assert post(base, body, {**hdr, "X-Optimizer": "bypass"}).status_code == 200
    conn = db.connect(db_path)
    route = db.get_route(conn, "mock-route")
    assert route["provider"] == "litellm" and route["model"] == "mock"
    rows = conn.execute("SELECT profile, output_tokens, stop_reason FROM requests ORDER BY id").fetchall()
    assert len(rows) == 2 and all(r["profile"] == "P0" and r["output_tokens"] and r["stop_reason"] == "stop" for r in rows)
    db.set_pin(conn, "mock-route", "P2")
    assert post(base, body, hdr).status_code == 200
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
    assert conn.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == 3
