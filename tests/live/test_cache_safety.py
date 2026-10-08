"""Standing live test (spec §11): a route pinned to P2 must still hit the customer's prompt cache.

Run: OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live -v
Costs ~2 small Opus 5.5 requests with a ~1.5k-token cached system prompt.
"""
import json
import os

import httpx
import pytest

from pith import db
from pith.config import Config
from pith.proxy import create_app

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_LIVE") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
                                reason="set OPTIMIZER_LIVE=1 and ANTHROPIC_API_KEY to run")

SYSTEM = ("You are a support classifier for an online store. " * 150)  # ~1.5k tokens, above the 512-token cache minimum
REQ = {"model": "claude-opus-5-5", "max_tokens": 64,
       "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
       "messages": [{"role": "user", "content": "Classify: 'Where is my order #123?' Reply with one word."}]}


@pytest.mark.anyio
async def test_p2_pinned_route_keeps_cache_reads():
    conn = db.connect(":memory:")
    app = create_app(Config(sample_rate=0), conn, client=httpx.AsyncClient(timeout=120))
    hdrs = {"content-type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01", "X-Optimizer-Route": "live-cache-test"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=120) as c:
        r1 = await c.post("/v1/messages", content=json.dumps(REQ).encode(), headers=hdrs)
        assert r1.status_code == 200, r1.text
        db.set_pin(conn, "live-cache-test", "P2")
        r2 = await c.post("/v1/messages", content=json.dumps(REQ).encode(), headers=hdrs)
        assert r2.status_code == 200, r2.text
    u2 = r2.json()["usage"]
    assert u2["cache_read_input_tokens"] > 0, u2
    assert db.get_route(conn, "live-cache-test")["injection_form"] == "system"
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
