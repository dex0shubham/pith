"""Spec §11/§13 demo (Plan 2): a 50-item support-ticket route through the proxy, then a real sweep.

Run: OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live/test_sweep_demo.py -v -s
Cost: 30 items ≈ $1.3 (about 30 minutes); 100 items with a 30% holdout is estimated at about $3, has cost about $4
(about 75 minutes), and needs the default budget of 10.
Knobs: OPTIMIZER_DEMO_SAMPLE (items, default 30), OPTIMIZER_DEMO_HOLDOUT (fraction, default 0),
OPTIMIZER_DEMO_BUDGET (USD ceiling, default 10).
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
TICKETS = [  # 100 distinct prompts, 20 per category; the first ten are the original demo set
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
    "Do you ship internationally to Canada, and who pays customs?",
    "My package says delivered but it is not on my porch.",
    "Can I pick up my order from a locker instead of home delivery?",
    "The delivery window keeps getting pushed back, when will it actually come?",
    "I need this by Friday for a birthday, can you upgrade the shipping?",
    "The tracking link in my confirmation email gives a 404 error.",
    "Can you hold my parcel at the depot while I am on vacation?",
    "Why was I charged shipping when my order was over the free-shipping limit?",
    "My order was split into three shipments, is that normal?",
    "The driver asked for a signature but nobody was home, what now?",
    "Can I change the delivery date after the order has shipped?",
    "My parcel was returned to sender because the address was incomplete.",
    # refund
    "I returned the headphones two weeks ago and still have no refund.",
    "Can I get a refund on a gift card I never used?",
    "The price dropped a day after I bought it, can you refund the difference?",
    "My refund went to an expired card, how will I get the money?",
    "I was charged a restocking fee I wasn't told about.",
    "The sweater shrank after one wash, I want a refund.",
    "I cancelled within an hour but was still charged.",
    "How long do refunds take to show up on PayPal?",
    "I only got a partial refund for a fully returned order.",
    "The item is not as described in the listing, I want my money back.",
    "Can I get a refund instead of store credit for my return?",
    "My promo code didn't apply at checkout, can you refund the discount?",
    "I received the wrong colour and want a refund, not an exchange.",
    "You refunded the item but not the original shipping charge.",
    "The return window closed yesterday, can I still get a refund?",
    "My order never arrived and I would like a full refund.",
    "I was billed for a subscription renewal I had already cancelled.",
    "Can I return a product I bought on sale?",
    # account
    "How do I delete my account and all my data?",
    "Someone changed the email on my account without my permission.",
    "I want to merge two accounts I accidentally created.",
    "Two-factor codes are not reaching my phone anymore.",
    "How do I turn off marketing emails but keep order updates?",
    "My account was locked after too many login attempts.",
    "How can I download a copy of all the data you hold about me?",
    "I want to change the name on my account after getting married.",
    "My saved payment methods disappeared from my profile.",
    "Can I add a second user to my household account?",
    "The app logs me out every time I close it.",
    "I never signed up but I'm getting order emails for an account.",
    "How do I update my phone number when I can't receive the old code?",
    "My loyalty points vanished from my account.",
    "Can I pause my subscription instead of cancelling it?",
    "I can't find my order history in my account anymore.",
    "Sign in with Google stopped working for my account.",
    "How do I change the default delivery address on my profile?",
    # product
    "Is the cast-iron pan safe to use on an induction hob?",
    "What is the battery life of the wireless earbuds?",
    "Does the coffee grinder come with a warranty, and for how long?",
    "Are the running shoes true to size or should I size up?",
    "Is the kids' backpack machine washable?",
    "Does the desk lamp take standard E27 bulbs?",
    "Will the phone case fit a model with a camera bump?",
    "Is the yoga mat latex-free? I have an allergy.",
    "How many cups does the large teapot hold?",
    "Does the electric kettle have an automatic shut-off?",
    "Are the replacement filters for the water jug sold separately?",
    "Can the outdoor chairs be left out in the rain?",
    "Is the winter coat warm enough for minus twenty degrees?",
    "Does the blender come with a travel cup?",
    "What material is the bed frame made of, solid wood or veneer?",
    "Is the baby monitor's video feed encrypted?",
    "Can the smart plug work without a hub?",
    "Is the backpack big enough for a 17-inch laptop?",
    # other
    "Do you have a physical store I can visit in Chicago?",
    "I'd like to give feedback about a rude delivery driver.",
    "Are you hiring for warehouse jobs over the holidays?",
    "Can I pay with a bank transfer instead of a card?",
    "Do you offer discounts for students or teachers?",
    "What are your customer service hours on weekends?",
    "Can I order in bulk for my company and get an invoice?",
    "Do you have an affiliate or referral programme?",
    "Is there a way to send an order as a gift with a message?",
    "How do you handle the recycling of packaging?",
    "I'd like to suggest a product you should stock.",
    "Do you price-match other retailers?",
    "Can I speak to a manager about my experience?",
    "Do you donate unsold stock to charity?",
    "Are your products tested on animals?",
    "Is there a mobile app for iPhone?",
    "Where are your products manufactured?",
    "Do you sell gift cards that can be used in store?",
    "I found a security issue on your website, who should I tell?",
]
assert len(TICKETS) == len(set(TICKETS)) == 100


@pytest.mark.anyio
async def test_demo_route_pins_a_profile_with_savings():
    n = int(os.environ.get("OPTIMIZER_DEMO_SAMPLE", "30"))
    holdout = float(os.environ.get("OPTIMIZER_DEMO_HOLDOUT", "0"))
    budget = float(os.environ.get("OPTIMIZER_DEMO_BUDGET", "10"))
    conn = db.connect(":memory:")
    cfg = Config(sample_rate=1.0, sweep_budget_usd_month=budget, equivalence_bar=0.9)
    app = create_app(cfg, conn, client=httpx.AsyncClient(timeout=120))
    hdrs = {"content-type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01", "X-Optimizer-Route": "demo-support"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=120) as c:
        for t in (TICKETS * (n // len(TICKETS) + 1))[:n]:
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
                    trials=3, sample_n=n, holdout=holdout, rng=random.Random(0))
    print(json.dumps(out.table, indent=1, default=str))
    if out.holdout:
        print("holdout:", json.dumps(out.holdout, indent=1, default=str))
    assert out.winner is not None, "no profile qualified — see table above"
    saved = 1 - out.table[out.winner]["mean_output"] / out.table["P0"]["mean_output"]
    assert saved >= 0.25, f"only {saved:.0%} fewer output tokens"
    assert out.table[out.winner]["qualifies"] is True
    assert db.get_route(conn, "demo-support")["pinned_profile"] == out.winner
