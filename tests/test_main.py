import json
import time

import httpx

from pith import db
from pith.__main__ import format_table, keys_from_env, main
from pith.config import Config
from pith.sweep import PROFILES_BY_PROVIDER, SweepOutcome, estimate_cost


def test_keys_from_env():
    assert keys_from_env({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "LITELLM_API_KEY": "l", "X": "1"}) == \
        {"anthropic": "a", "openai": "o", "litellm": "l"}
    assert keys_from_env({"OPENAI_API_KEY": ""}) == {}


def seed(conn, n=50, key="k", provider="anthropic", model="claude-opus-5-5"):
    db.upsert_route(conn, key, provider, model, "h")
    for i in range(n):
        ref = db.store_body(conn, json.dumps({"model": model, "messages": [{"role": "user", "content": f"q{i}"}]}), "{}", 9e9)
        db.record_request(conn, ts=time.time() - i, route_key=key, profile="P0", input_tokens=10, output_tokens=10, cache_read=0,
                          cache_create=0, estimated=False, stop_reason="end_turn", latency_ms=1, body_ref=ref)


def script(req):
    body = json.loads(req.content)
    if "You compare" in (body.get("system") or ""):
        return httpx.Response(200, json={"content": [{"type": "text", "text": "equivalent"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 10, "output_tokens": 1}})
    shaped = any(m.get("role") == "system" for m in body["messages"])
    return httpx.Response(200, json={"content": [{"type": "text", "text": "s" if shaped else "long " * 30}], "stop_reason": "end_turn",
                                     "usage": {"input_tokens": 10, "output_tokens": 2 if shaped else 30}})


def test_sweep_subcommand_pins_and_exit_codes(tmp_path, capsys):
    cfg = tmp_path / "o.toml"
    cfg.write_text("sweep_budget_usd_month = 50\n")
    conn = db.connect(":memory:")
    seed(conn)
    client = httpx.Client(transport=httpx.MockTransport(script))
    rc = main(["sweep", "--config", str(cfg), "--trials", "1", "--sample", "5"], env={"ANTHROPIC_API_KEY": "a"}, conn=conn, client=client)
    out = capsys.readouterr().out
    assert rc == 0 and "k" in out and "P2" in out and "qualifies" in out
    assert db.get_route(conn, "k")["pinned_profile"] in ("P2", "P4")
    rc = main(["sweep", "--config", str(cfg), "--route", "k"], env={"ANTHROPIC_API_KEY": "a"}, conn=conn, client=client)
    assert rc == 0  # pinned route swept again only because --route names it
    rc = main(["sweep", "--config", str(cfg)], env={}, conn=conn, client=client)
    assert rc == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_sweep_budget_refusal_exit_2_and_dry_run(tmp_path, capsys):
    cfg = tmp_path / "o.toml"
    cfg.write_text("sweep_budget_usd_month = 0\n")
    conn = db.connect(":memory:")
    seed(conn)
    client = httpx.Client(transport=httpx.MockTransport(script))
    rc = main(["sweep", "--config", str(cfg), "--sample", "5"], env={"ANTHROPIC_API_KEY": "a"}, conn=conn, client=client)
    assert rc == 2 and "exceeds ceiling" in capsys.readouterr().out
    rc = main(["sweep", "--config", str(cfg), "--sample", "5", "--budget-usd", "5", "--dry-run", "--trials", "1"],
              env={"ANTHROPIC_API_KEY": "a"}, conn=conn, client=client)
    assert rc == 0 and db.get_route(conn, "k")["pinned_profile"] == "P0"


def test_sweep_continues_past_refused_routes(tmp_path, capsys):
    cfg = tmp_path / "o.toml"
    cfg.write_text("sweep_budget_usd_month = 50\n")
    conn = db.connect(":memory:")
    seed(conn)
    seed(conn, key="nop", provider="openai", model="gpt-unpriced")
    client = httpx.Client(transport=httpx.MockTransport(script))
    rc = main(["sweep", "--config", str(cfg), "--trials", "1", "--sample", "5"],
              env={"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"}, conn=conn, client=client)
    out = capsys.readouterr().out
    assert rc == 2 and "no price for model 'gpt-unpriced'" in out
    assert out.index("route nop") < out.index("route k:")  # refused route came first, k still ran
    assert db.get_route(conn, "k")["pinned_profile"] != "P0"


def test_recheck_subcommand(tmp_path, capsys):
    cfg = tmp_path / "o.toml"
    cfg.write_text("")
    conn = db.connect(":memory:")
    seed(conn)
    db.set_pin(conn, "k", "P2")
    for i in range(3):
        ref = db.store_body(conn, json.dumps({"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "q"}]}),
                            json.dumps({"content": [{"type": "text", "text": "s"}], "stop_reason": "end_turn", "usage": {}}), 9e9)
        db.record_request(conn, ts=time.time(), route_key="k", profile="P2", input_tokens=1, output_tokens=1, cache_read=0,
                          cache_create=0, estimated=False, stop_reason="end_turn", latency_ms=1, body_ref=ref)
    rc = main(["recheck", "--config", str(cfg), "--n", "3"], env={"ANTHROPIC_API_KEY": "a"}, conn=conn,
              client=httpx.Client(transport=httpx.MockTransport(script)))
    assert rc == 0 and "judged=3" in capsys.readouterr().out


def test_format_table_lists_profiles():
    out = SweepOutcome("P2", {"P0": {"rate": 1.0, "cost_per_request": 0.001, "mean_output": 30, "stops": 0, "skipped": False, "qualifies": False, "reason": "baseline"},
                              "P2": {"rate": 1.0, "cost_per_request": 0.0005, "mean_output": 2, "stops": 0, "skipped": False, "qualifies": True, "reason": "qualifies"}},
                       1.0, 0.01, 1, 20, "s")
    text = format_table("k", out)
    assert "P0" in text and "P2" in text and "winner: P2" in text and "qualifies" in text


def test_budget_usd_is_per_run(tmp_path, capsys):
    cfg = tmp_path / "o.toml"
    cfg.write_text("sweep_budget_usd_month = 0\n")
    client = httpx.Client(transport=httpx.MockTransport(script))
    args = ["sweep", "--config", str(cfg), "--trials", "1", "--sample", "5"]
    env = {"ANTHROPIC_API_KEY": "a"}
    probe = db.connect(":memory:")
    seed(probe)
    assert main(args + ["--budget-usd", "50"], env=env, conn=probe, client=client) == 0
    one = db.sweeps_for_route(probe, "k")[0]["cost_usd"]
    capsys.readouterr()
    # the estimate (judge overhead included) is far above actual spend: a ceiling just over one estimate fits the first
    # route, and what is left after its real cost no longer fits the second
    mi, mo = probe.execute("SELECT AVG(input_tokens), AVG(output_tokens) FROM requests WHERE profile='P0'").fetchone()
    est = estimate_cost(db.get_route(probe, "k"), [None] * 5, PROFILES_BY_PROVIDER["anthropic"], 1, Config(),
                        mean_input=mi, mean_output=mo)
    conn = db.connect(":memory:")
    seed(conn, key="k1")
    seed(conn, key="k2")
    rc = main(args + ["--budget-usd", str(est + one / 2)], env=env, conn=conn, client=client)
    out = capsys.readouterr().out
    assert rc == 2 and out.count("refused") == 1
    assert conn.execute("SELECT COUNT(*) FROM sweeps WHERE finished_at IS NOT NULL").fetchone()[0] == 1
