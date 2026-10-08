import json
import time

import pytest

from optimizer import db


def test_schema_and_route_roundtrip():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k1", "anthropic", "claude-opus-5-5", "h", now=1.0)
    r = db.get_route(conn, "k1")
    assert r["status"] == "observing" and r["pinned_profile"] == "P0"
    assert r["injection_form"] == "system" and r["target_words"] == 20 and r["rejections"] == 0
    assert r["first_seen"] == 1.0 and r["last_seen"] == 1.0
    db.upsert_route(conn, "k1", "anthropic", "claude-opus-5-5", "h", now=5.0)
    r = db.get_route(conn, "k1")
    assert r["first_seen"] == 1.0 and r["last_seen"] == 5.0
    assert db.get_route(conn, "missing") is None


def test_pin_injection_and_rejections():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "openai", "gpt-5", "h")
    db.set_pin(conn, "k", "P2")
    db.set_injection_form(conn, "k", "user_text")
    assert db.bump_rejection(conn, "k") == 1
    assert db.bump_rejection(conn, "k") == 2
    r = db.get_route(conn, "k")
    assert (r["pinned_profile"], r["status"], r["injection_form"], r["rejections"]) == ("P2", "pinned", "user_text", 2)
    db.set_pin(conn, "k", "P0", status="reverted")
    assert db.get_route(conn, "k")["status"] == "reverted"


def test_requests_bodies_and_stats():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "m", "h")
    ref = db.store_body(conn, '{"a":1}', '{"b":2}', expires_at=100.0)
    for prof, out, stop in (("P0", 100, "end_turn"), ("P0", 200, "max_tokens"), ("P2", 50, "end_turn")):
        db.record_request(conn, ts=1.0, route_key="k", profile=prof, input_tokens=10, output_tokens=out,
                          cache_read=5, cache_create=0, estimated=False, stop_reason=stop, latency_ms=12,
                          body_ref=ref)
    rows = {(r["route_key"], r["profile"]): r for r in db.route_stats(conn)}
    assert rows[("k", "P0")]["n"] == 2 and rows[("k", "P0")]["avg_output"] == 150
    assert rows[("k", "P0")]["max_tokens_stops"] == 1
    assert rows[("k", "P2")]["avg_output"] == 50 and rows[("k", "P2")]["avg_cache_read"] == 5
    assert db.purge_expired(conn, now=50.0) == 0
    assert db.purge_expired(conn, now=150.0) == 1


def test_connect_sets_wal_on_file_db(tmp_path):
    conn = db.connect(str(tmp_path / "o.db"))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def _req(conn, key, profile="P0", out=100, stop="end_turn", body_ref=None, ts=None):
    return db.record_request(conn, ts=time.time() if ts is None else ts, route_key=key, profile=profile,
                             input_tokens=50, output_tokens=out, cache_read=0, cache_create=0, estimated=False,
                             stop_reason=stop, latency_ms=1, body_ref=body_ref)


def test_upsert_route_unpins_on_hash_or_model_change():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "named", "anthropic", "m1", "hash-a")
    db.set_pin(conn, "named", "P2")
    db.upsert_route(conn, "named", "anthropic", "m1", "hash-a")
    assert db.get_route(conn, "named")["pinned_profile"] == "P2"
    db.upsert_route(conn, "named", "anthropic", "m2", "hash-b")
    r = db.get_route(conn, "named")
    assert (r["pinned_profile"], r["status"], r["system_hash"], r["model"]) == ("P0", "observing", "hash-b", "m2")
    db.set_pin(conn, "named", "P2")
    db.upsert_route(conn, "named", "anthropic", "m3", "hash-b")
    r = db.get_route(conn, "named")
    assert (r["pinned_profile"], r["status"], r["model"]) == ("P0", "observing", "m3")
    db.set_pin(conn, "named", "P2")
    db.upsert_route(conn, "named", "anthropic", "m3", "hash-b")
    r = db.get_route(conn, "named")
    assert (r["pinned_profile"], r["status"], r["model"], r["system_hash"]) == ("P2", "pinned", "m3", "hash-b")


def test_request_stats_and_candidates():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "m", "h")
    assert db.route_request_stats(conn, "k") == {"p0_total": 0, "p0_sampled": 0, "text_frac": None}
    refs = [db.store_body(conn, json.dumps({"model": "m", "messages": [], "n": i}), "{}", 9e9) for i in range(3)]
    _req(conn, "k", out=10, body_ref=refs[0], ts=1)
    _req(conn, "k", out=20, body_ref=refs[1], ts=2)
    _req(conn, "k", out=30, body_ref=refs[2], ts=3, stop="tool_use")
    _req(conn, "k", out=40, ts=4)                     # no body
    _req(conn, "k", profile="P2", out=5, ts=5)         # not P0
    s = db.route_request_stats(conn, "k")
    assert s == {"p0_total": 4, "p0_sampled": 3, "text_frac": 0.75}
    cands = db.sample_candidates(conn, "k")
    assert [c["output_tokens"] for c in cands] == [20, 10]
    assert json.loads(cands[0]["request_json"])["n"] == 1
    db.purge_expired(conn, now=1e10)  # bodies gone, body_ref left dangling
    assert db.route_request_stats(conn, "k")["p0_sampled"] == 0


def test_samples_sweeps_judgments_and_month_cost():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "m", "h")
    sid = db.create_sample(conn, "k", [1, 2, 3], now=100.0)
    sw = db.create_sweep(conn, "k", sid, "judge-m", "v1", now=100.0)
    db.add_judgment(conn, sw, 1, "P2", 1, "equivalent", "baseline-first")
    db.update_sweep_cost(conn, sw, 0.5)
    db.finish_sweep(conn, sw, 1.25, json.dumps({"P2": {"rate": 1.0}}), "P2", now=200.0)
    rows = db.sweeps_for_route(conn, "k")
    assert rows[0]["winner"] == "P2" and rows[0]["cost_usd"] == 1.25 and rows[0]["finished_at"] == 200.0
    assert json.loads(rows[0]["result_json"])["P2"]["rate"] == 1.0
    assert conn.execute("SELECT label, order_ab FROM judgments").fetchone()[:] == ("equivalent", "baseline-first")
    db.create_sweep(conn, "k", sid, "j", "v1", now=time.time())
    sw2 = conn.execute("SELECT MAX(id) FROM sweeps").fetchone()[0]
    db.update_sweep_cost(conn, sw2, 2.0)
    assert db.month_sweep_cost(conn, time.time()) >= 2.0
    assert db.month_sweep_cost(conn, 50.0) == 0.0


def test_set_route_fields_allowlist_and_recent_bodies():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "m", "h")
    db.set_route_fields(conn, "k", status="no-savings", eligible=1, target_words=33, exemplar="Yes.", last_sweep_id=7)
    r = db.get_route(conn, "k")
    assert (r["status"], r["eligible"], r["target_words"], r["exemplar"], r["last_sweep_id"]) == ("no-savings", 1, 33, "Yes.", 7)
    with pytest.raises(ValueError):
        db.set_route_fields(conn, "k", pinned_profile="P9")
    refs = [db.store_body(conn, '{"i":%d}' % i, '{"r":%d}' % i, 9e9) for i in range(3)]
    for i, ref in enumerate(refs):
        _req(conn, "k", profile="P2", body_ref=ref, ts=i)
    _req(conn, "k", profile="P0", body_ref=refs[0], ts=9)
    rows = db.recent_bodies(conn, "k", "P2", 2)
    assert [json.loads(x["request_json"])["i"] for x in rows] == [2, 1]
    assert json.loads(rows[0]["response_json"]) == {"r": 2}


def test_shadow_rate_and_projected_volume():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "m", "h")
    assert db.shadow_rate(conn, "k") == (None, 0)
    for lab in ("equivalent", "equivalent", "missing-info", "judge-error"):
        db.add_shadow(conn, "k", None, lab)
    rate, n = db.shadow_rate(conn, "k")
    assert n == 3 and rate == pytest.approx(2 / 3)
    db.set_pin(conn, "k", "P2")
    assert db.shadow_rate(conn, "k") == (None, 0)  # set_pin clears the shadow window
    now = time.time()
    for i in range(14):
        _req(conn, "k", ts=now - i * 3600)         # 14 requests in the last day
    _req(conn, "k", ts=now - 10 * 86400)           # outside the window
    assert db.projected_monthly_volume(conn, "k", now) == 1000  # floored
    for i in range(700):
        _req(conn, "k", ts=now - 60 - i)
    assert db.projected_monthly_volume(conn, "k", now) == round(714 * 30 / 7)
