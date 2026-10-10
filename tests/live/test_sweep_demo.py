"""Spec §11/§13 demo (Plan 2): a 50-item support-ticket route through the proxy, then a real sweep.

Run: OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live/test_sweep_demo.py -v -s
Cost: 30 items ≈ $1.3; 100 items with a 30% holdout ≈ $4.
Knobs: OPTIMIZER_DEMO_SAMPLE (items, default 30), OPTIMIZER_DEMO_HOLDOUT (fraction, default 0),
OPTIMIZER_DEMO_BUDGET (USD ceiling, default 5).
"""
import json
import os
import random
import time

import httpx
import pytest

from pith import db
from pith.config import Config
from pith.proxy import create_app
from pith.sweep import run_sweep

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_LIVE") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
                                reason="set OPTIMIZER_LIVE=1 and ANTHROPIC_API_KEY to run")

SYSTEM = ("You are a support assistant for an online store. Classify the ticket as one of: shipping, refund, account, "
          "product, other. Then explain your reasoning to the customer in a friendly way.")
TICKETS = [
    "Where is my order #1234? It was supposed to arrive yesterday.", "I want my money back for the broken blender.",
    "I can't log in, password reset email never arrives.", "Does the blue jacket come in XL?",
    "Your website is slow today.", "Package arrived damaged, the box was crushed.", "Charged twice for one order.",
    "How do I change my shipping address?", "Is the laptop stand compatible with a 16-inch MacBook?",
    "Cancel my subscription please.",
    # shipping
    "My tracking number has shown 'label created' for five days now.",
    "Can you ship to a PO box in Alaska, and how long does it take?",
    "The courier left my parcel at the wrong address, what do I do?",
    "I paid for express delivery but it came by standard post.",
    "Only half of my order arrived, the second box is missing.",
    # refund
    "I returned the headphones two weeks ago and still have no refund.",
    "Can I get a refund on a gift card I never used?",
    "The price dropped a day after I bought it, can you refund the difference?",
    "My refund went to an expired card, how will I get the money?",
    "I was charged a restocking fee I wasn't told about.",
    # account
    "How do I delete my account and all my data?",
    "Someone changed the email on my account without my permission.",
    "I want to merge two accounts I accidentally created.",
    "Two-factor codes are not reaching my phone anymore.",
    "How do I turn off marketing emails but keep order updates?",
    # product
    "Is the cast-iron pan safe to use on an induction hob?",
    "What is the battery life of the wireless earbuds?",
    "Does the coffee grinder come with a warranty, and for how long?",
    "Are the running shoes true to size or should I size up?",
    "Is the kids' backpack machine washable?",
    # other
    "Do you have a physical store I can visit in Chicago?",
    "I'd like to give feedback about a rude delivery driver.",
    "Are you hiring for warehouse jobs over the holidays?",
    "Can I pay with a bank transfer instead of a card?",
] * 3
N = int(os.environ.get("OPTIMIZER_DEMO_SAMPLE", "30"))
HOLDOUT = float(os.environ.get("OPTIMIZER_DEMO_HOLDOUT", "0"))
BUDGET = float(os.environ.get("OPTIMIZER_DEMO_BUDGET", "5"))


@pytest.mark.anyio
async def test_demo_route_pins_a_profile_with_savings():
    conn = db.connect(":memory:")
    cfg = Config(sample_rate=1.0, sweep_budget_usd_month=BUDGET, equivalence_bar=0.9)
    app = create_app(cfg, conn, client=httpx.AsyncClient(timeout=120))
    hdrs = {"content-type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01", "X-Optimizer-Route": "demo-support"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=120) as c:
        for t in (TICKETS * (N // len(TICKETS) + 1))[:N]:
            body = {"model": "claude-haiku-4-5", "max_tokens": 400, "system": SYSTEM,
                    "messages": [{"role": "user", "content": t}]}
            r = await c.post("/v1/messages", content=json.dumps(body).encode(), headers=hdrs)
            assert r.status_code == 200, r.text
    route = db.get_route(conn, "demo-support")
    # A 30-ticket route can never repay a ~$1 sweep (cost is amortized over projected monthly volume, floor 1,000).
    # Simulate the volume of a real support route so the economics reflect the product's target workload.
    now = time.time()
    for i in range(3000):
        db.record_request(conn, ts=now - i * 100, route_key="demo-support", profile="P0", input_tokens=60, output_tokens=160,
                          cache_read=0, cache_create=0, estimated=False, stop_reason="end_turn", latency_ms=1)
    out = run_sweep(conn, cfg, route, httpx.Client(timeout=120), {"anthropic": os.environ["ANTHROPIC_API_KEY"]},
                    trials=3, sample_n=N, holdout=HOLDOUT, rng=random.Random(0))
    print(json.dumps(out.table, indent=1, default=str))
    holdout = json.loads(db.sweeps_for_route(conn, "demo-support")[0]["result_json"])["holdout"]
    if holdout:
        print("holdout:", json.dumps(holdout, indent=1, default=str))
    assert out.winner is not None, "no profile qualified — see table above"
    saved = 1 - out.table[out.winner]["mean_output"] / out.table["P0"]["mean_output"]
    assert saved >= 0.25, f"only {saved:.0%} fewer output tokens"
    assert out.table[out.winner]["rate"] >= 0.9
    assert db.get_route(conn, "demo-support")["pinned_profile"] == out.winner
