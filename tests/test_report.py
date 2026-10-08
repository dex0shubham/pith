import httpx
import pytest

from pith import db
from pith.config import Config
from pith.proxy import create_app
from pith.report import PRICES, price_for, render_html, route_rows


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


import json

from pith.report import sweep_rows


def test_rows_include_sweep_and_recheck_columns():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "claude-opus-5-5", "h")
    sid = db.create_sample(conn, "k", [1, 2])
    sw = db.create_sweep(conn, "k", sid, "j", "v1", now=100.0)
    result = {"table": {"P0": {"rate": 0.98}, "P2": {"rate": 0.96}}, "floor": 0.98, "sample_n": 2}
    db.finish_sweep(conn, sw, 0.42, json.dumps(result), "P2", now=150.0)
    db.set_pin(conn, "k", "P2")
    db.set_route_fields(conn, "k", last_sweep_id=sw)
    for lab in ("equivalent", "missing-info"):
        db.add_shadow(conn, "k", None, lab)
    r = route_rows(conn)[0]
    assert r["equivalence_pct"] == pytest.approx(96.0) and r["noise_floor_pct"] == pytest.approx(98.0)
    assert r["sample_n"] == 2 and r["last_sweep_at"] == 150.0 and r["sweep_cost_usd"] == 0.42
    assert r["recheck_pct"] == pytest.approx(50.0)
    rows = sweep_rows(conn, "k")
    assert rows[0]["winner"] == "P2" and rows[0]["result"]["floor"] == 0.98 and "result_json" not in rows[0]


def test_rows_without_sweep_have_none_columns_and_price_override():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "x", "openai", "gpt-5", "h")
    db.record_request(conn, ts=1, route_key="x", profile="P0", input_tokens=1000, output_tokens=100, cache_read=0,
                      cache_create=0, estimated=False, stop_reason="stop", latency_ms=1)
    r = route_rows(conn)[0]
    assert r["equivalence_pct"] is None and r["recheck_pct"] is None and r["baseline_usd_per_1k"] is None
    r = route_rows(conn, prices={"gpt-5": (1.0, 10.0)})[0]
    assert r["baseline_usd_per_1k"] == pytest.approx((1000 * 1.0 + 100 * 10.0) / 1e6 * 1000)
    html = render_html(route_rows(conn))
    assert "equivalence_pct" in html
