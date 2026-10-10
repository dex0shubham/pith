import json
import time

from pith import db
from pith.config import Config
from pith.sweep import PROFILES_BY_PROVIDER, derive_targets, eligible_routes, pick_sample, stratify


def seed_route(conn, key="k", n=50, text=True, bodies=True, status=None, model="claude-opus-5-5"):
    db.upsert_route(conn, key, "anthropic", model, "h")
    for i in range(n):
        ref = db.store_body(conn, json.dumps({"model": model, "stream": True, "messages": [{"role": "user", "content": f"q{i}"}]}),
                            "{}", 9e9) if bodies else None
        db.record_request(conn, ts=time.time() - i, route_key=key, profile="P0", input_tokens=10, output_tokens=i * 10,
                          cache_read=0, cache_create=0, estimated=False, stop_reason="end_turn" if text else "tool_use",
                          latency_ms=1, body_ref=ref)
    if status:
        db.set_route_fields(conn, key, status=status)


def test_profiles_constant():
    assert PROFILES_BY_PROVIDER == {"anthropic": ("P0", "P1", "P2", "P3", "P4"),
                                    "openai": ("P0", "P1", "P1b", "P2", "P3", "P4"),
                                    "litellm": ("P0", "P2", "P3"), "portkey": ("P0", "P2", "P3")}


def test_eligibility_rules():
    conn = db.connect(":memory:")
    seed_route(conn, "ok")
    seed_route(conn, "few", n=10)
    seed_route(conn, "tools", text=False)
    db.set_pin(conn, "tools", "P2", status="observing")  # stale pin on a route that is still sweep-eligible by status
    seed_route(conn, "pinned", status="pinned")
    seed_route(conn, "reverted", status="reverted")
    keys = sorted(r["key"] for r in eligible_routes(conn, Config()))
    assert keys == ["ok", "reverted"]
    assert db.get_route(conn, "ok")["eligible"] == 1
    assert db.get_route(conn, "tools")["status"] == "not-applicable" and db.get_route(conn, "tools")["eligible"] == 0
    assert db.get_route(conn, "tools")["pinned_profile"] == "P0"
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
    assert [p["id"] for p in stratify(cands, 5)] == [19, 39, 59, 79, 99]  # newest first within each quintile


def test_pick_sample_strips_stream_and_writes_nothing():
    conn = db.connect(":memory:")
    seed_route(conn, "k", n=60)
    items = pick_sample(conn, "k", 50)
    assert len(items) == 50 and all("stream" not in it["body"] for it in items)
    assert all(it["body"]["messages"][0]["content"].startswith("q") for it in items)
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0


def test_pick_sample_excludes_server_side_tools():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "claude-opus-5-5", "h")
    for i, tools in enumerate([[{"type": "web_search_20260209", "name": "web_search"}], [{"name": "f", "input_schema": {}}]]):
        ref = db.store_body(conn, json.dumps({"model": "m", "tools": tools, "messages": []}), "{}", 9e9)
        db.record_request(conn, ts=time.time() - i, route_key="k", profile="P0", input_tokens=1, output_tokens=1, cache_read=0,
                          cache_create=0, estimated=False, stop_reason="end_turn", latency_ms=1, body_ref=ref)
    items = pick_sample(conn, "k", 10)
    assert len(items) == 1 and items[0]["body"]["tools"] == [{"name": "f", "input_schema": {}}]


def test_derive_targets():
    assert derive_targets(["one two three four five six", "a b c d e f g h", "x y"]) == (20, "x y")
    texts = ["w " * 100, "w " * 120, "w " * 200]
    assert derive_targets(texts) == (60, "w " * 100)
    assert derive_targets([]) == (20, "")


import random

import httpx
import pytest

from pith.sweep import (MECHANICAL_FAILS, BudgetRefused, NoPrice, NothingToSample, SweepAborted, SweepOutcome,
                             estimate_cost, pin_rule, run_sweep)


def _row(rate, cost, stops=0, cache=0.0, skipped=False):
    return {"n": 50, "skipped": skipped, "equivalent": 0, "judged": 1, "judge_error": 0, "stops": stops, "mean_input": 10,
            "mean_output": 10, "mean_cache_read": cache, "rate": rate, "cost_per_request": cost}


def test_estimate_cost_and_no_price():
    route = {"model": "claude-opus-5-5", "provider": "anthropic"}
    items = [{"id": 1, "body": {}}] * 10
    cfg = Config()
    est = estimate_cost(route, items, ("P0", "P2"), 3, cfg, mean_input=1000, mean_output=200)
    replays = 10 * 3 * 2 * (1000 * 4 + 200 * 20) / 1e6
    judge_calls = 10 * (3 * 1 + 3)  # per item: trials × candidates + C(trials, 2) noise pairs
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
    assert pin_rule(table, 0.9, 0.99) is None and table["P2"]["reason"] == "rate below noise floor tolerance"
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


def test_pin_rule_rejects_mostly_errored_judge():
    table = {"P0": _row(0.98, 1.0), "P2": {**_row(0.98, 0.5), "judged": 10, "judge_error": 3}}
    assert pin_rule(table, 0.95, 0.98) is None and table["P2"]["reason"] == "judge mostly errored"
    table = {"P0": _row(0.98, 1.0), "P2": {**_row(0.98, 0.5), "judged": 10, "judge_error": 2}}
    assert pin_rule(table, 0.95, 0.98) == "P2"


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
        last_user = next((m for m in reversed(body["messages"]) if m.get("role") == "user"), {})
        user_text_shaped = isinstance(last_user.get("content"), list) and any(
            "Answer directly" in b.get("text", "") for b in last_user["content"] if isinstance(b, dict))
        shaped = any(m.get("role") == "system" for m in body["messages"]) or user_text_shaped
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


def _sweep_setup(n=50, model="claude-opus-5-5"):
    conn = db.connect(":memory:")
    seed_route(conn, "k", n=n, model=model)
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
    db.set_pin(conn, "k", "P2")
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(Script(judge_label="B-omits"))),
                    {"anthropic": "k"}, trials=1, sample_n=5, rng=random.Random(0))
    assert out.winner is None and db.get_route(conn, "k")["status"] == "no-savings"
    assert db.get_route(conn, "k")["pinned_profile"] == "P0"
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


def test_run_sweep_profile_not_skipped_when_only_some_items_are_noop():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "claude-opus-5-5", "h")
    for i in range(10):
        body = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": f"q{i}"}]}
        if i == 0:
            body["output_config"] = {"effort": "low"}
        ref = db.store_body(conn, json.dumps(body), "{}", 9e9)
        db.record_request(conn, ts=time.time() - i, route_key="k", profile="P0", input_tokens=10, output_tokens=i * 10,
                          cache_read=0, cache_create=0, estimated=False, stop_reason="end_turn", latency_ms=1, body_ref=ref)
    cfg = Config(sweep_budget_usd_month=100.0)
    out = run_sweep(conn, cfg, db.get_route(conn, "k"), httpx.Client(transport=httpx.MockTransport(Script())), {"anthropic": "k"},
                    trials=1, sample_n=5, rng=random.Random(0))
    assert out.table["P1"]["skipped"] is False and out.table["P1"]["n"] == 5


def test_run_sweep_skips_profile_when_all_items_noop():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "claude-haiku-4-5", "h")
    for i in range(10):
        ref = db.store_body(conn, json.dumps({"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": f"q{i}"}]}), "{}", 9e9)
        db.record_request(conn, ts=time.time() - i, route_key="k", profile="P0", input_tokens=10, output_tokens=i * 10,
                          cache_read=0, cache_create=0, estimated=False, stop_reason="end_turn", latency_ms=1, body_ref=ref)
    script = Script()
    out = run_sweep(conn, Config(sweep_budget_usd_month=100.0), db.get_route(conn, "k"),
                    httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"}, trials=1, sample_n=5,
                    rng=random.Random(0))
    assert out.table["P1"]["skipped"] is True
    assert conn.execute("SELECT COUNT(*) FROM judgments WHERE profile='P1'").fetchone()[0] == 0
    assert not any("output_config" in c for c in script.calls if c["model"] == "claude-haiku-4-5")  # judge calls carry their own


def test_run_sweep_persists_spend_on_unexpected_error(monkeypatch):
    conn, cfg, route = _sweep_setup()

    def boom(_texts):
        raise RuntimeError("boom")

    monkeypatch.setattr("pith.sweep.derive_targets", boom)
    with pytest.raises(RuntimeError):
        run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(Script())), {"anthropic": "k"},
                  trials=1, sample_n=5, rng=random.Random(0))
    sw = db.sweeps_for_route(conn, "k")[0]
    assert sw["finished_at"] is None and sw["winner"] is None and sw["cost_usd"] > 0


def test_run_sweep_aborts_on_transport_failures_and_leaves_open_row():
    conn, cfg, route = _sweep_setup()
    script = Script(transport_fail_p0=5)  # all 5 P0 trial-1 calls fail -> >20%
    with pytest.raises(SweepAborted):
        run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                  trials=1, sample_n=5, rng=random.Random(0))
    sw = db.sweeps_for_route(conn, "k")[0]
    assert sw["finished_at"] is None and sw["winner"] is None
    assert db.get_route(conn, "k")["pinned_profile"] == "P0"



def _no_pin_unfinished(conn):
    sw = db.sweeps_for_route(conn, "k")[0]
    assert sw["finished_at"] is None and sw["winner"] is None
    return sw


def test_run_sweep_aborts_when_provider_returns_5xx():
    conn, cfg, route = _sweep_setup()
    script = Script()

    def h(req):
        if "You compare" in (json.loads(req.content).get("system") or ""):
            return script(req)
        return httpx.Response(529, json={"error": {"message": "overloaded"}})
    with pytest.raises(SweepAborted, match="transport failures"):
        run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(h)), {"anthropic": "k"},
                  trials=1, sample_n=5, rng=random.Random(0))
    _no_pin_unfinished(conn)
    r = db.get_route(conn, "k")
    assert r["status"] == "observing" and r["pinned_profile"] == "P0"


def test_run_sweep_refuses_tiny_sample():
    conn = db.connect(":memory:")
    seed_route(conn, "k", n=3)
    with pytest.raises(NothingToSample):
        run_sweep(conn, Config(sweep_budget_usd_month=100.0), db.get_route(conn, "k"),
                  httpx.Client(transport=httpx.MockTransport(Script())), {"anthropic": "k"}, trials=1, sample_n=10)
    assert conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0


def test_run_sweep_no_pin_when_noise_floor_undefined():
    conn, cfg, route = _sweep_setup()
    script = Script()

    def h(req):
        body = json.loads(req.content)
        if "You compare" in (body.get("system") or "") and "short answer here" not in body["messages"][0]["content"]:
            # noise pair (two long baseline answers): unparseable -> judge-error
            return httpx.Response(200, json={"content": [{"type": "text", "text": "hmm"}], "stop_reason": "end_turn",
                                             "usage": {"input_tokens": 300, "output_tokens": 3}})
        return script(req)
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(h)), {"anthropic": "k"},
                    trials=2, sample_n=5, rng=random.Random(0))
    assert out.floor is None and out.winner is None
    assert db.get_route(conn, "k")["status"] == "no-savings" and db.get_route(conn, "k")["pinned_profile"] == "P0"
    cands = {p: row for p, row in out.table.items() if p != "P0"}
    assert cands and all(row["qualifies"] is False and row["reason"] == "noise floor undefined" for row in cands.values())


def test_run_sweep_aborts_when_spend_exceeds_ceiling_mid_sweep():
    conn, cfg, route = _sweep_setup()
    script = Script()

    def h(req):
        r = script(req)
        if "You compare" in (json.loads(req.content).get("system") or ""):
            return r
        data = json.loads(r.content)  # real usage far above the recorded means the estimate is built from
        data["usage"].update(input_tokens=20000, output_tokens=2000)
        return httpx.Response(200, json=data)
    stats = conn.execute("SELECT AVG(input_tokens), AVG(output_tokens) FROM requests WHERE route_key='k' AND profile='P0'").fetchone()
    est = estimate_cost(route, [None] * 10, PROFILES_BY_PROVIDER["anthropic"], 3, cfg, mean_input=stats[0], mean_output=stats[1])
    with pytest.raises(SweepAborted, match="over ceiling"):
        run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(h)), {"anthropic": "k"},
                  trials=3, sample_n=10, budget_usd=est * 1.01, rng=random.Random(0))
    assert _no_pin_unfinished(conn)["cost_usd"] > est
    assert db.get_route(conn, "k")["pinned_profile"] == "P0"


def test_run_sweep_aborts_on_judge_unavailable():
    conn, cfg, route = _sweep_setup()
    script = Script()

    def h(req):
        if "You compare" in (json.loads(req.content).get("system") or ""):
            return httpx.Response(500, json={})
        return script(req)
    with pytest.raises(SweepAborted, match="judge unavailable"):
        run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(h)), {"anthropic": "k"},
                  trials=1, sample_n=5, rng=random.Random(0))
    assert _no_pin_unfinished(conn)["cost_usd"] > 0
    assert db.get_route(conn, "k")["pinned_profile"] == "P0"


from pith.sweep import RECHECK_MIN_ROWS, RecheckOutcome, recheck


def _pinned_route_with_live(conn, n_live=25, live_text="short live answer", profile="P2", content=None):
    db.upsert_route(conn, "k", "anthropic", "claude-opus-5-5", "h")
    db.set_pin(conn, "k", profile)
    content = [{"type": "text", "text": live_text}] if content is None else content
    for i in range(n_live):
        ref = db.store_body(conn, json.dumps({"model": "claude-opus-5-5", "messages": [{"role": "user", "content": f"q{i}"}]}),
                            json.dumps({"content": content, "stop_reason": "end_turn", "usage": {}}), 9e9)
        db.record_request(conn, ts=time.time() - i, route_key="k", profile=profile, input_tokens=1, output_tokens=1, cache_read=0,
                          cache_create=0, estimated=False, stop_reason="end_turn", latency_ms=1, body_ref=ref)
    return db.get_route(conn, "k")


def test_recheck_judges_live_vs_p0_and_keeps_pin_when_fine():
    conn = db.connect(":memory:")
    route = _pinned_route_with_live(conn)
    script = Script()
    out = recheck(conn, Config(), route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"}, n=20,
                  rng=random.Random(0))
    assert isinstance(out, RecheckOutcome) and out.judged == 20 and out.rate == 1.0 and out.reverted is False
    assert db.shadow_rate(conn, "k") == (1.0, 20)
    assert db.get_route(conn, "k")["pinned_profile"] == "P2"
    p0_calls = [c for c in script.calls if "You compare" not in (c.get("system") or "")]
    assert len(p0_calls) == 20 and all(not any(m["role"] == "system" for m in c["messages"]) for c in p0_calls)


def test_recheck_reverts_below_bar_with_enough_rows():
    conn = db.connect(":memory:")
    route = _pinned_route_with_live(conn)
    out = recheck(conn, Config(), route, httpx.Client(transport=httpx.MockTransport(Script(judge_label="B-omits"))),
                  {"anthropic": "k"}, n=20, rng=random.Random(0))
    assert out.reverted is True and out.rate == 0.0
    r = db.get_route(conn, "k")
    assert r["pinned_profile"] == "P0" and r["status"] == "reverted"


def test_recheck_needs_min_rows_before_reverting():
    conn = db.connect(":memory:")
    route = _pinned_route_with_live(conn, n_live=5)
    out = recheck(conn, Config(), route, httpx.Client(transport=httpx.MockTransport(Script(judge_label="contradiction"))),
                  {"anthropic": "k"}, n=20, rng=random.Random(0))
    assert out.judged == 5 and out.reverted is False and RECHECK_MIN_ROWS == 20
    assert db.get_route(conn, "k")["pinned_profile"] == "P2"


def test_recheck_skips_unpinned_and_no_key():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "u", "anthropic", "m", "h")
    out = recheck(conn, Config(), db.get_route(conn, "u"), httpx.Client(transport=httpx.MockTransport(Script())), {"anthropic": "k"})
    assert out == RecheckOutcome(0, None, False)
    route = _pinned_route_with_live(conn)
    with pytest.raises(SweepAborted):
        recheck(conn, Config(), route, httpx.Client(transport=httpx.MockTransport(Script())), {})


def test_recheck_window_resets_on_repin():
    conn = db.connect(":memory:")
    route = _pinned_route_with_live(conn)
    out = recheck(conn, Config(), route, httpx.Client(transport=httpx.MockTransport(Script(judge_label="B-omits"))),
                  {"anthropic": "k"}, n=20, rng=random.Random(0))
    assert out.reverted is True
    route = _pinned_route_with_live(conn, profile="P4")
    out = recheck(conn, Config(), route, httpx.Client(transport=httpx.MockTransport(Script())), {"anthropic": "k"}, n=20,
                  rng=random.Random(0))
    assert out.reverted is False and db.shadow_rate(conn, "k") == (1.0, 20)


def test_recheck_min_rows_counts_judged_only():
    conn = db.connect(":memory:")
    route = _pinned_route_with_live(conn, content=[])
    out = recheck(conn, Config(), route, httpx.Client(transport=httpx.MockTransport(Script())), {"anthropic": "k"}, n=20,
                  rng=random.Random(0))
    assert out.judged == 0 and out.reverted is False and db.shadow_rate(conn, "k") == (None, 0)


class SystemRoleRejectingScript(Script):
    """Provider that rejects mid-conversation system messages (e.g. claude-haiku-4-5) but accepts the user-text form."""

    def __call__(self, req):
        body = json.loads(req.content)
        if not (body.get("system") and "You compare two answers" in body["system"]) and \
                any(m.get("role") == "system" for m in body["messages"]):
            self.calls.append(body)
            return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error",
                                                                          "message": "role 'system' is not supported on this model"}})
        return super().__call__(req)


def test_run_sweep_switches_to_user_text_when_provider_rejects_system_role():
    conn, cfg, route = _sweep_setup()
    script = SystemRoleRejectingScript()
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                    trials=1, sample_n=5, rng=random.Random(0))
    assert db.get_route(conn, "k")["injection_form"] == "user_text"
    assert out.table["P2"]["rate"] == 1.0 and out.table["P2"]["n"] == 5 and out.winner in ("P2", "P4")
    rejected = [c for c in script.calls if any(m.get("role") == "system" for m in c.get("messages", []))]
    assert len(rejected) == 1  # one probe failure, then every later shaped replay used the user-text form


def test_alias_survives_system_role_restart():
    conn, cfg, route = _sweep_setup(model="claude-haiku-4-5")
    script = SystemRoleRejectingScript()
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                    trials=1, sample_n=5, rng=random.Random(0))
    assert out.table["P4"]["alias_of"] == "P2"
    assert out.winner == "P2"
    assert db.get_route(conn, "k")["injection_form"] == "user_text"


def test_pin_rule_names_unrecovered_sweep_cost_separately():
    # candidate is cheaper per request before amortization but not after: the sweep didn't pay for itself
    table = {"P0": _row(0.98, 1.0), "P2": {**_row(0.98, 1.2), "raw_cost_per_request": 0.5}}
    assert pin_rule(table, 0.95, 0.98) is None
    assert table["P2"]["reason"] == "sweep cost not recovered at projected volume"
    # genuinely more expensive even before amortization
    table = {"P0": _row(0.98, 1.0), "P2": {**_row(0.98, 1.2), "raw_cost_per_request": 1.1}}
    assert pin_rule(table, 0.95, 0.98) is None and table["P2"]["reason"] == "not cheaper than P0"


def test_run_sweep_rows_carry_raw_cost_before_amortization():
    conn, cfg, route = _sweep_setup()
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(Script())), {"anthropic": "k"},
                    trials=1, sample_n=5, rng=random.Random(0))
    p2 = out.table["P2"]
    assert p2["raw_cost_per_request"] < p2["cost_per_request"]  # amortization added on top
    assert out.table["P0"]["raw_cost_per_request"] == out.table["P0"]["cost_per_request"]


def test_pin_rule_bar_is_capped_by_the_noise_floor():
    # the unconstrained model agrees with itself only 86% of the time: a 0.9 bar can't be demanded of any profile
    table = {"P0": _row(0.866, 1.0), "P2": _row(0.866, 0.5), "P4": _row(0.85, 0.45)}
    assert pin_rule(table, 0.9, 0.866) == "P2"
    assert table["P2"]["qualifies"] and table["P4"]["reason"] == "rate below bar"
    # a deterministic route (floor 1.0) still has to meet the absolute bar
    table = {"P0": _row(1.0, 1.0), "P2": _row(0.88, 0.5)}
    assert pin_rule(table, 0.9, 1.0) is None and table["P2"]["reason"] == "rate below bar"


def test_pin_rule_refuses_routes_too_noisy_to_judge():
    table = {"P0": _row(0.4, 1.0), "P2": _row(0.4, 0.5)}
    assert pin_rule(table, 0.9, 0.4) is None and table["P2"]["reason"] == "route too noisy to judge (noise floor < 0.5)"


from pith.sweep import item_scores


def test_item_scores_majority_with_half_credit_for_ties():
    rate, se, n = item_scores({1: ["equivalent", "equivalent", "B-omits"], 2: ["equivalent", "B-omits"],
                               3: ["B-omits"], 4: ["judge-error"], 5: ["judge-error", "equivalent"]})
    assert n == 4 and rate == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert se > 0
    assert item_scores({}) == (None, None, 0)
    assert item_scores({1: ["equivalent"]}) == (1.0, 0.0, 1)


def test_pin_rule_tolerance_widens_with_sampling_error():
    p0 = {**_row(0.95, 1.0), "rate_se": 0.04}
    wide = {"P0": p0, "P2": {**_row(0.90, 0.5), "rate_se": 0.05}}   # sqrt(.04²+.05²)=.064 > .05 gap
    assert pin_rule(wide, 0.9, 0.95) == "P2"
    tight = {"P0": {**_row(0.95, 1.0), "rate_se": 0.005}, "P2": {**_row(0.90, 0.5), "rate_se": 0.005}}
    assert pin_rule(tight, 0.9, 0.95) is None and tight["P2"]["reason"] == "rate below noise floor tolerance"


class CyclingJudgeScript(Script):
    """Judge replies cycle through the given labels; replays behave like Script."""

    def __init__(self, cycle):
        super().__init__()
        self.cycle, self.k = cycle, 0

    def __call__(self, req):
        body = json.loads(req.content)
        if body.get("system") and "You compare two answers" in body["system"]:
            self.calls.append(body)
            label = self.cycle[self.k % len(self.cycle)]
            self.k += 1
            return httpx.Response(200, json={"content": [{"type": "text", "text": label}], "stop_reason": "end_turn",
                                             "usage": {"input_tokens": 300, "output_tokens": 3}})
        return super().__call__(req)


def test_run_sweep_judges_all_p0_pairs_and_uses_item_majority():
    conn, cfg, route = _sweep_setup()
    script = CyclingJudgeScript(("equivalent", "equivalent", "B-omits"))  # every item: 2 of 3 pass
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                    trials=3, sample_n=4, rng=random.Random(0))
    assert conn.execute("SELECT COUNT(*) FROM judgments WHERE profile='P0'").fetchone()[0] == 12  # 3 pairs × 4 items
    assert out.table["P0"]["rate"] == 1.0 and out.table["P2"]["rate"] == 1.0 and out.table["P2"]["rate_se"] == 0.0
    assert out.table["P2"]["items_judged"] == 4 and out.table["P2"]["judged"] == 12 and out.winner in ("P2", "P4")


def test_run_sweep_judges_identical_requests_once():
    # claude-haiku-4-5 has no effort parameter, so P4's effective request is byte-identical to P2's.
    conn, cfg, route = _sweep_setup(model="claude-haiku-4-5")
    script = Script()
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                    trials=2, sample_n=10, rng=random.Random(0), now=time.time())
    table = out.table
    assert table["P1"]["skipped"] is True and table["P1"]["reason"] == "skipped"
    assert table["P4"] == {"skipped": True, "alias_of": "P2", "qualifies": False, "reason": "same request as P2 on this model"}
    assert out.winner == "P2" and db.get_route(conn, "k")["pinned_profile"] == "P2"
    assert conn.execute("SELECT COUNT(*) FROM judgments WHERE profile='P4'").fetchone()[0] == 0
    shaped_calls = [b for b in script.calls if not (b.get("system") and "You compare two answers" in b["system"])
                    and any(m.get("role") == "system" or isinstance(m.get("content"), list) for m in b["messages"])]
    assert len(shaped_calls) == 2 * 10 * 2  # P2 and P3 only (P3 differs by its exemplar), trials × items each
    stored = json.loads(db.sweeps_for_route(conn, "k")[0]["result_json"])["table"]
    assert stored["P4"]["alias_of"] == "P2"
