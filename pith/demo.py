"""`pith demo`: record -> sweep -> pin -> serve, end to end in-process against a mock upstream. No key, no network."""
import asyncio
import json
import random
import sys
import time

import httpx

from pith import db
from pith.config import Config
from pith.judge import SYSTEM_PROMPT
from pith.proxy import create_app
from pith.rewrite import SHAPE_PREFIX
from pith.sweep import run_sweep

ROUTE = "demo"
MODEL = "claude-haiku-4-5"
SYSTEM = ("You are a support assistant for an online store. Classify the ticket as one of: shipping, refund, account, "
          "product, other. Then explain your reasoning to the customer in a friendly way.")
TICKETS = {
    "Where is my order? It was supposed to arrive yesterday.": "shipping",
    "The parcel tracking has not updated in five days.": "shipping",
    "I want my money back for the broken blender.": "refund",
    "I returned the shoes two weeks ago and still have no refund.": "refund",
    "I can't log in, the password reset email never arrives.": "account",
    "How do I change the email address on my account?": "account",
    "Does the blue jacket come in XL?": "product",
    "Is the coffee grinder dishwasher safe?": "product",
    "Do you have a physical store in Berlin?": "other",
    "Can I leave feedback about your delivery driver?": "other",
}
CHATTY = ("Thank you so much for reaching out, and I am really sorry for the trouble this has caused you.",
          "I completely understand how frustrating this kind of thing can be, especially when you were counting on it.",
          "Let me walk you through what I found and exactly what happens next on our side.",
          "I have looked into the details you shared and I want to make sure we get this sorted out properly for you.",
          "Our team handles requests like this every day, so you are in good hands and this should be quick.",
          "If anything else comes up in the meantime, please do not hesitate to get back in touch with us.",
          "We really appreciate your patience and your understanding while we take care of this for you.",
          "Just to recap, I have noted everything above on your ticket so nobody will ask you to repeat yourself.",
          "Wishing you a wonderful rest of your day, and thank you again for being such a valued customer!")
TERSE = ("I have escalated it to the right team.", "You will get an email update within 24 hours.",
         "No action is needed from you.", "Reply here if anything changes.", "The ticket stays open until resolved.")
NOISY_ITEMS = 0.05  # share of items the mock judge finds ambiguous: keeps the noise floor below 1.0, as on real traffic


def _ticket(i: int) -> str:
    return f"{list(TICKETS)[i % len(TICKETS)]} (ticket #{1001 + i})"


def _text_parts(content) -> list[str]:
    return [content] if isinstance(content, str) else [p.get("text", "") for p in content if isinstance(p, dict)]


def _clip(sentences, rng, n: int) -> str:
    words = []
    while len(words) < n:
        words += rng.choice(sentences).split()
    return " ".join(words[:n])


def mock_upstream(seed: int = 0):
    """An httpx.MockTransport handler shaped like Anthropic's /v1/messages: chatty unshaped answers, short shaped
    ones (a pith system message or shape text part), and a judge that says `equivalent` except on ambiguous items."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        msgs = body.get("messages") or []
        in_tok = len(json.dumps(msgs) + str(body.get("system", ""))) // 4
        if body.get("system") == SYSTEM_PROMPT:
            question = _text_parts(msgs[-1]["content"])[0].split("REQUEST:\n", 1)[-1].split("\n\nANSWER A:", 1)[0]
            rng = random.Random(f"{seed}|judge|{question}")
            text = rng.choice(("A-omits", "B-omits")) if rng.random() < NOISY_ITEMS else "equivalent"
            return _reply(body, text, in_tok, 1)
        user = next(m for m in reversed(msgs) if m.get("role") == "user")
        parts = _text_parts(user["content"])
        shaped = any(m.get("role") == "system" for m in msgs) or any(p.startswith(SHAPE_PREFIX) for p in parts)
        question = parts[0]
        category = TICKETS.get(question.rsplit(" (ticket #", 1)[0], "other")
        rng = random.Random(f"{seed}|{question}|{shaped}")
        if shaped:
            n = rng.randint(30, 60)
            text = f"Category: {category}. " + _clip(TERSE, rng, n - 2)
        else:
            n = rng.randint(120, 180)
            text = f"Hi there! This one is a {category} request. " + _clip(CHATTY, rng, n - 8)
        return _reply(body, text, in_tok, n)
    return handler


def _reply(body: dict, text: str, in_tok: int, out_tok: int) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "msg_demo", "type": "message", "role": "assistant", "model": body.get("model"),
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok}})


async def _send(proxy: httpx.AsyncClient, first: int, count: int) -> list[int]:
    hdrs = {"x-api-key": "demo", "anthropic-version": "2023-06-01", "X-Optimizer-Route": ROUTE}
    outs = []
    for i in range(first, first + count):
        body = {"model": MODEL, "max_tokens": 400, "system": SYSTEM, "messages": [{"role": "user", "content": _ticket(i)}]}
        r = await proxy.post("/v1/messages", json=body, headers=hdrs)
        r.raise_for_status()
        outs.append(r.json()["usage"]["output_tokens"])
    return outs


async def _flow(say, conn, seed: int, items: int, trials: int, db_path: str) -> int:
    from pith.__main__ import format_table  # function-level: pith.__main__ imports this module for `pith demo`
    mock = httpx.MockTransport(mock_upstream(seed))
    cfg = Config(listen="127.0.0.1:8787", sample_rate=1.0, sweep_budget_usd_month=10, db_path=db_path)
    async with httpx.AsyncClient(transport=mock) as upstream:
        app = create_app(cfg, conn, client=upstream, warm=False)  # no tiktoken vocabulary fetch
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://pith.demo") as proxy:
            say(f"pith demo (synthetic): in-process proxy, mock {MODEL} upstream, mock judge; no API key; "
                "no request data leaves this machine")
            baseline = await _send(proxy, 0, items)
            say(f"recorded {len(baseline)} requests, route {ROUTE} {db.get_route(conn, ROUTE)['status']}"
                f" (mean {sum(baseline) / len(baseline):.0f} output tokens)")

            # A 40-request route can never repay a sweep: cost is amortised over projected monthly volume. Simulate
            # the volume of a real support route, as tests/live/test_sweep_demo.py does.
            now = time.time()
            for i in range(3000):
                db.record_request(conn, ts=now - i * 100, route_key=ROUTE, profile="P0", input_tokens=60,
                                  output_tokens=160, cache_read=0, cache_create=0, estimated=False,
                                  stop_reason="end_turn", latency_ms=1)
            say("seeded 3000 P0 rows of projected volume so the sweep cost amortises like a real support route")

            say(f"sweeping {ROUTE}: {items} items x {trials} trials per profile, judge {cfg.judge_model} (mocked)")
            with httpx.Client(transport=mock) as client:
                out = run_sweep(conn, cfg, db.get_route(conn, ROUTE), client, {"anthropic": "demo"}, trials=trials,
                                sample_n=items, rng=random.Random(seed))
            say(format_table(ROUTE, out))
            if not out.winner:
                say("no profile pinned; see the table above")
                return 1

            served = await _send(proxy, items, 3)
            last = conn.execute("SELECT profile, output_tokens FROM requests WHERE route_key=? ORDER BY id DESC LIMIT 1",
                                (ROUTE,)).fetchone()
            base = sum(baseline) / len(baseline)
            say(f"3 more requests: last served with {last['profile']} ({last['output_tokens']} output tokens); "
                f"mean {sum(served) / 3:.0f} vs baseline {base:.0f} ({_change(sum(served) / 3, base)})")
    row = out.table[out.winner]
    floor = "n/a" if out.floor is None else f"{out.floor:.2f}"
    say(f"pinned {out.winner} (synthetic): {_change(row['mean_output'], out.table['P0']['mean_output'])} output tokens "
        f"at equivalence {row['rate']:.2f} (noise floor {floor}); no request data left this machine")
    return 0


def _change(new: float, base: float) -> str:
    return f"{new / base - 1:+.0%}"  # sign from the value itself: "-71%", "+5%"


def run_demo(out=None, seed: int = 0, items: int = 40, trials: int = 3, db_path: str = ":memory:") -> int:
    out = out or sys.stdout
    say = lambda s: print(s, file=out, flush=True)
    conn = db.connect(db_path)
    try:
        # The demo writes synthetic rows (3000 of them fake volume): never into a database that holds anything else.
        if db.get_route(conn, ROUTE):
            say("database already has a demo route; use a fresh file")
            return 2
        if conn.execute("SELECT 1 FROM routes LIMIT 1").fetchone():
            say("database already has other routes; the demo writes synthetic rows, use a fresh file")
            return 2
        return asyncio.run(_flow(say, conn, seed, items, trials, db_path))
    finally:
        conn.close()
