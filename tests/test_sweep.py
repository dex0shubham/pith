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


import random

import httpx
import pytest

from optimizer.sweep import (MECHANICAL_FAILS, BudgetRefused, NoPrice, SweepAborted, SweepOutcome, estimate_cost, pin_rule,
                             run_sweep)


def _row(rate, cost, stops=0, cache=0.0, skipped=False):
    return {"n": 50, "skipped": skipped, "equivalent": 0, "judged": 1, "judge_error": 0, "stops": stops, "mean_input": 10,
            "mean_output": 10, "mean_cache_read": cache, "rate": rate, "cost_per_request": cost}


def test_estimate_cost_and_no_price():
    route = {"model": "claude-opus-5-5", "provider": "anthropic"}
    items = [{"id": 1, "body": {}}] * 10
    cfg = Config()
    est = estimate_cost(route, items, ("P0", "P2"), 3, cfg, mean_input=1000, mean_output=200)
    replays = 10 * 3 * 2 * (1000 * 4 + 200 * 20) / 1e6
    judge_calls = 10 * 3 * (1 + 1)  # 1 candidate + 1 noise pair per trial
    judges = judge_calls * ((4000 + 2 * 200) * 2 + 8 * 10) / 1e6
    assert est == pytest.approx(replays + judges)
    with pytest.raises(NoPrice):
        estimate_cost({"model": "gpt-unknown", "provider": "openai"}, items, ("P0",), 1, cfg, mean_input=1, mean_output=1)


def test_pin_rule_each_condition():
    bar, floor = 0.95, 0.98
    table = {"P0": _row(0.98, 1.0, cache=100), "P1": _row(0.96, 0.9, cache=100), "P2": _row(0.97, 0.5, cache=100)}
    assert pin_rule(table, bar, floor) == "P2" and table["P2"]["qualifies"] and table["P1"]["qualifies"]
    table = {"P0": _row(0.98, 1.0), "P2": _row(0.94, 0.5)}
    assert pin_rule(table, bar, floor) is None and table["P2"]["reason"] == "rate below bar"
    table = {"P0": _row(0.99, 1.0), "P2": _row(0.95, 0.5)}
    assert pin_rule(table, 0.9, 0.99) is None and table["P2"]["reason"] == "rate below noise floor - 0.03"
    table = {"P0": _row(0.98, 1.0, stops=1), "P2": _row(0.98, 0.5, stops=2)}
    assert pin_rule(table, bar, floor) is None and table["P2"]["reason"] == "more max_tokens stops than P0"
    table = {"P0": _row(0.98, 1.0, cache=100), "P2": _row(0.98, 0.5, cache=10)}
    assert pin_rule(table, bar, floor) is None and table["P2"]["reason"] == "cache reads below P0"
    table = {"P0": _row(0.98, 1.0, cache=0), "P2": _row(0.98, 0.5, cache=0)}
    assert pin_rule(table, bar, floor) == "P2"  # cache check skipped when P0 has none
    table = {"P0": _row(0.98, 1.0), "P2": _row(0.98, 1.2)}
    assert pin_rule(table, bar, floor) is None and table["P2"]["reason"] == "not cheaper than P0"
    table = {"P0": _row(0.98, 1.0), "P1": _row(None, 0.0, skipped=True)}
    assert pin_rule(table, bar, floor) is None and table["P1"]["reason"] == "skipped"
    table = {"P0": _row(0.5, 1.0), "P2": _row(0.5, 0.5)}
    assert pin_rule(table, bar, None) is None  # no floor: only the bar applies


class Script:
    """Scripted upstream: P0 answers are long, P2/P4 answers short; judge says equivalent unless told otherwise."""

    def __init__(self, judge_label="equivalent", fail_profile=None, transport_fail_p0=0):
        self.calls = []
        self.judge_label = judge_label
        self.fail_profile = fail_profile
        self.transport_fail_p0 = transport_fail_p0

    def __call__(self, req):
        body = json.loads(req.content)
        self.calls.append(body)
        if body.get("system") and "You compare two answers" in body["system"]:
            return httpx.Response(200, json={"content": [{"type": "text", "text": self.judge_label}], "stop_reason": "end_turn",
                                             "usage": {"input_tokens": 300, "output_tokens": 3}})
        shaped = any(m.get("role") == "system" for m in body["messages"])
        effort = body.get("output_config", {}).get("effort")
        if self.fail_profile == "shape" and shaped:
            return httpx.Response(400, json={"error": {"message": "bad"}})
        if not shaped and effort is None and self.transport_fail_p0 > 0:
            self.transport_fail_p0 -= 1
            raise httpx.ConnectError("down", request=req)
        text = "short answer here" if shaped else "this is a long baseline answer with many more words in it " * 3
        return httpx.Response(200, json={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 100, "output_tokens": 5 if shaped else 40,
                                                   "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0}})


def _sweep_setup(n=50):
    conn = db.connect(":memory:")
    seed_route(conn, "k", n=n)
    cfg = Config(sweep_budget_usd_month=100.0)
    return conn, cfg, db.get_route(conn, "k")


def test_run_sweep_pins_cheapest_qualifying_and_records():
    conn, cfg, route = _sweep_setup()
    script = Script()
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                    trials=2, sample_n=10, rng=random.Random(0), now=time.time())
    assert isinstance(out, SweepOutcome) and out.winner in ("P2", "P4") and out.floor == 1.0
    r = db.get_route(conn, "k")
    assert r["pinned_profile"] == out.winner and r["status"] == "pinned" and r["last_sweep_id"] == out.sweep_id
    assert r["target_words"] == 20 and r["exemplar"].startswith("this is a long baseline")
    sw = db.sweeps_for_route(conn, "k")[0]
    assert sw["winner"] == out.winner and sw["finished_at"] is not None and sw["cost_usd"] == pytest.approx(out.cost_usd)
    sample = conn.execute("SELECT item_ids_json FROM samples WHERE id=?", (sw["sample_id"],)).fetchone()
    assert len(json.loads(sample[0])) == 10
    table = json.loads(sw["result_json"])["table"]
    assert table["P1"]["skipped"] is False  # opus-5-5 has effort
    assert table["P2"]["rate"] == 1.0 and table["P0"]["rate"] == 1.0
    assert conn.execute("SELECT COUNT(*) FROM judgments WHERE profile='P0'").fetchone()[0] == 10  # noise pairs: 10 items × (2-1)
    assert conn.execute("SELECT COUNT(*) FROM judgments WHERE profile='P2'").fetchone()[0] == 20
    assert out.cost_usd > 0


def test_run_sweep_no_savings_when_judge_rejects():
    conn, cfg, route = _sweep_setup()
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(Script(judge_label="B-omits"))),
                    {"anthropic": "k"}, trials=1, sample_n=5, rng=random.Random(0))
    assert out.winner is None and db.get_route(conn, "k")["status"] == "no-savings"
    assert db.sweeps_for_route(conn, "k")[0]["winner"] == "P0"


def test_run_sweep_dry_run_writes_nothing():
    conn, cfg, route = _sweep_setup()
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(Script())), {"anthropic": "k"},
                    trials=1, sample_n=5, dry_run=True, rng=random.Random(0))
    assert out.winner is not None and out.sweep_id is None
    assert db.get_route(conn, "k")["pinned_profile"] == "P0"
    assert conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 0


def test_run_sweep_budget_refused_before_any_call():
    conn, cfg, route = _sweep_setup()
    script = Script()
    with pytest.raises(BudgetRefused) as e:
        run_sweep(conn, Config(sweep_budget_usd_month=0.0), route, httpx.Client(transport=httpx.MockTransport(script)),
                  {"anthropic": "k"}, trials=1, sample_n=5)
    assert script.calls == [] and e.value.ceiling == 0.0 and e.value.estimate > 0
    with pytest.raises(BudgetRefused):
        run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                  trials=1, sample_n=5, budget_usd=0.0)
    assert conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0  # refused = nothing written
    # --budget-usd overrides a zero monthly ceiling for this run
    out = run_sweep(conn, Config(sweep_budget_usd_month=0.0), route, httpx.Client(transport=httpx.MockTransport(script)),
                    {"anthropic": "k"}, trials=1, sample_n=5, budget_usd=5.0, rng=random.Random(0))
    assert out.sweep_id is not None


def test_run_sweep_mechanical_failure_marks_format_broken_without_judge():
    conn, cfg, route = _sweep_setup()
    script = Script(fail_profile="shape")
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                    trials=1, sample_n=5, rng=random.Random(0))
    assert out.table["P2"]["rate"] == 0.0 and out.table["P2"]["reason"] == "rate below bar"
    labels = [r[0] for r in conn.execute("SELECT label FROM judgments WHERE profile='P2'")]
    assert labels and set(labels) == {"format-broken"}
    assert MECHANICAL_FAILS == ("max_tokens", "length", "max_output_tokens")


def test_run_sweep_aborts_on_transport_failures_and_leaves_open_row():
    conn, cfg, route = _sweep_setup()
    script = Script(transport_fail_p0=5)  # all 5 P0 trial-1 calls fail -> >20%
    with pytest.raises(SweepAborted):
        run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                  trials=1, sample_n=5, rng=random.Random(0))
    sw = db.sweeps_for_route(conn, "k")[0]
    assert sw["finished_at"] is None and sw["winner"] is None
    assert db.get_route(conn, "k")["pinned_profile"] == "P0"
