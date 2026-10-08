import json
import time

from optimizer import db
from optimizer.config import Config
from optimizer.sweep import PROFILES_BY_PROVIDER, derive_targets, eligible_routes, pick_sample, stratify


def seed_route(conn, key="k", n=50, text=True, bodies=True, status=None):
    db.upsert_route(conn, key, "anthropic", "claude-opus-5-5", "h")
    for i in range(n):
        ref = db.store_body(conn, json.dumps({"model": "claude-opus-5-5", "stream": True, "messages": [{"role": "user", "content": f"q{i}"}]}),
                            "{}", 9e9) if bodies else None
        db.record_request(conn, ts=time.time() - i, route_key=key, profile="P0", input_tokens=10, output_tokens=i * 10,
                          cache_read=0, cache_create=0, estimated=False, stop_reason="end_turn" if text else "tool_use",
                          latency_ms=1, body_ref=ref)
    if status:
        db.set_route_fields(conn, key, status=status)


def test_profiles_constant():
    assert PROFILES_BY_PROVIDER == {"anthropic": ("P0", "P1", "P2", "P3", "P4"), "openai": ("P0", "P1", "P1b", "P2", "P3", "P4")}


def test_eligibility_rules():
    conn = db.connect(":memory:")
    seed_route(conn, "ok")
    seed_route(conn, "few", n=10)
    seed_route(conn, "tools", text=False)
    seed_route(conn, "pinned", status="pinned")
    seed_route(conn, "reverted", status="reverted")
    keys = sorted(r["key"] for r in eligible_routes(conn, Config()))
    assert keys == ["ok", "reverted"]
    assert db.get_route(conn, "ok")["eligible"] == 1
    assert db.get_route(conn, "tools")["status"] == "not-applicable" and db.get_route(conn, "tools")["eligible"] == 0
    assert db.get_route(conn, "few")["status"] == "observing" and db.get_route(conn, "few")["eligible"] is None
    assert [r["key"] for r in eligible_routes(conn, Config(), only="pinned")] == ["pinned"]
    assert eligible_routes(conn, Config(), only="few") == []


def test_stratify_round_robin_across_quintiles():
    cands = [{"id": i, "output_tokens": i} for i in range(100)]
    picked = stratify(cands, 10)
    assert len(picked) == 10 and len({p["id"] for p in picked}) == 10
    buckets = sorted(p["output_tokens"] // 20 for p in picked)
    assert buckets == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    skewed = [{"id": i, "output_tokens": 1} for i in range(8)] + [{"id": 100, "output_tokens": 999}]
    picked = stratify(skewed, 6)
    assert len(picked) == 6 and any(p["id"] == 100 for p in picked)
    assert stratify([], 5) == []
    assert len(stratify(cands[:3], 50)) == 3


def test_pick_sample_strips_stream_and_writes_nothing():
    conn = db.connect(":memory:")
    seed_route(conn, "k", n=60)
    items = pick_sample(conn, "k", 50)
    assert len(items) == 50 and all("stream" not in it["body"] for it in items)
    assert all(it["body"]["messages"][0]["content"].startswith("q") for it in items)
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0


def test_derive_targets():
    assert derive_targets(["one two three four five six", "a b c d e f g h", "x y"]) == (20, "x y")
    texts = ["w " * 100, "w " * 120, "w " * 200]
    assert derive_targets(texts) == (60, "w " * 100)
    assert derive_targets([]) == (20, "")
