import httpx
import pytest

from optimizer import db
from optimizer.config import Config
from optimizer.proxy import create_app
from optimizer.report import PRICES, price_for, render_html, route_rows


def seed():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "claude-opus-5-5", "h", name="billing", now=7.0)
    for prof, out in (("P0", 1000), ("P0", 1000), ("P2", 400)):
        db.record_request(conn, ts=1, route_key="k", profile=prof, input_tokens=1000, output_tokens=out, cache_read=0,
                          cache_create=0, estimated=False, stop_reason="end_turn", latency_ms=1)
    db.set_pin(conn, "k", "P2")
    return conn


def test_rows_compute_cost_and_savings():
    rows = route_rows(seed())
    assert len(rows) == 1
    r = rows[0]
    assert (r["key"], r["name"], r["status"], r["pinned_profile"], r["last_seen"]) == ("k", "billing", "pinned", "P2", 7.0)
    assert r["baseline_n"] == 2 and r["baseline_avg_output"] == 1000
    assert r["pinned_n"] == 1 and r["pinned_avg_output"] == 400
    inp, out = PRICES["claude-opus-5-5"]
    assert r["baseline_usd_per_1k"] == pytest.approx((1000 * inp + 1000 * out) / 1e6 * 1000)
    assert r["pinned_usd_per_1k"] == pytest.approx((1000 * inp + 400 * out) / 1e6 * 1000)
    assert r["estimated_savings_pct"] == pytest.approx(100 * (1 - r["pinned_usd_per_1k"] / r["baseline_usd_per_1k"]))


def test_unknown_model_price_is_none_not_error():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "x", "openai", "gpt-unknown", "h")
    db.record_request(conn, ts=1, route_key="x", profile="P0", input_tokens=1, output_tokens=1, cache_read=0,
                      cache_create=0, estimated=False, stop_reason="stop", latency_ms=1)
    r = route_rows(conn)[0]
    assert r["baseline_usd_per_1k"] is None and r["estimated_savings_pct"] is None and r["pinned_n"] == 0


def test_html_escapes_and_lists_routes():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "m", "h", name="<b>x</b>")
    html = render_html(route_rows(conn))
    assert "&lt;b&gt;x&lt;/b&gt;" in html and "<table" in html


@pytest.mark.anyio
async def test_endpoints():
    app = create_app(Config(), seed(), client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/optimizer/report")).json()[0]["key"] == "k"
        r = await c.get("/optimizer/report.html")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "billing" in r.text


def test_price_for_prefers_override_then_builtin():
    assert price_for("claude-opus-5-5") == PRICES["claude-opus-5-5"]
    assert price_for("claude-opus-5-5", {"claude-opus-5-5": (1.0, 2.0)}) == (1.0, 2.0)
    assert price_for("gpt-5", {"gpt-5": (1.25, 10.0)}) == (1.25, 10.0)
    assert price_for("gpt-5") is None
