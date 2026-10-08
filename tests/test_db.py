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
