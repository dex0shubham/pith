# Output-Token Optimizer — Plan 2: Control Plane

**Goal:** An operator-run `sweep` / `recheck` CLI that replays a frozen sample of a route's real requests under each output profile, judges equivalence against the unconstrained baseline, pins the cheapest profile that clears the bar, and later detects drift — turning the observe-only proxy from Plan 1 into one that actually saves money.

**Architecture:** Three new pure-ish modules — `replay.py` (one provider call, text extraction, cost), `judge.py` (one versioned prompt, label parsing, order normalization), `sweep.py` (eligibility, sampling, cost gate, replay loop, pin rule, recheck) — driven by argparse subcommands in `__main__.py`, all out of process from the proxy. State goes through `db.py` into the SQLite the proxy already reads per request, so a pin takes effect on the next request. The proxy gains only the keyless hash-change unpin (in `upsert_route`), new report columns, and a sweep-audit endpoint.

**Tech Stack:** Python 3.12, httpx (sync `httpx.Client` for the CLI; `MockTransport` in tests), stdlib `sqlite3`/`argparse`/`json`/`random`/`statistics`, pytest.

**Spec:** `docs/design/specs/2026-10-07-control-plane-design.md` (authoritative), which amends §6/§9 of `docs/design/specs/2026-10-07-output-token-optimizer-design.md`. Read both before starting.

## Global Constraints

- Python ≥ 3.12; code under `pith/`, tests under `tests/`; pytest only, one `test_*.py` per module, no fixture frameworks; the suite must stay warning-free under `.venv/bin/pytest -q -W error`. Run everything with `.venv/bin/pytest` / `.venv/bin/python`.
- **The proxy stays keyless.** Sweeps and rechecks read `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` from the CLI's environment only; nothing in `proxy.py` ever sees or stores a key.
- **Replays never trigger tools:** samples contain only P0 requests whose `stop_reason` is in `("end_turn", "stop", "completed")`.
- **Budget:** a sweep refuses (exit 2, nothing written) when its cost estimate exceeds `--budget-usd` or `sweep_budget_usd_month − sum(sweeps.cost_usd this month)`; default `sweep_budget_usd_month = 0` means refuse. Actual spend is re-checked every 50 calls; exceeding it aborts (exit 1, `finished_at NULL`, no pin).
- **Judge labels** are exactly `equivalent | missing-info | contradiction | format-broken` from the model, plus the stored-only `extra-info` (candidate adds claims; produced by order normalization) and `judge-error` (unparseable twice; excluded from rates). `JUDGE_PROMPT_VERSION = "v1"`.
- **Pin rule** (all must hold): `rate ≥ bar`; `rate ≥ floor − 0.03`; `max_tokens` stops ≤ P0's; mean `cache_read` ≥ P0's when P0's mean > 0; `$/request < P0's`. Cheapest qualifying wins; none → `status = no-savings`.
- **Statuses** are exactly `observing pinned no-savings not-applicable sweeping reverted`.
- `--dry-run` spends the same money, prints the table, writes nothing. `recheck` never pins; it may revert.
- Profile lists: Anthropic `P0 P1 P2 P3 P4`; OpenAI `P0 P1 P1b P2 P3 P4`. A profile whose rewritten body equals the original is `skipped`, not billed.
- `target_words = max(20, round(0.5 × median word count of P0 trial-1 texts))`; `exemplar` = the shortest P0 trial-1 text.

## File structure

| File | Responsibility |
|---|---|
| `pith/config.py` (modify) | drop `shadow_rate`; add `prices` from `[prices."model"]` |
| `pith/report.py` (modify) | `price_for`; sweep columns in `route_rows`; `sweep_rows` |
| `pith/db.py` (modify) | hash-change unpin in `upsert_route`; sweep/sample/judgment/shadow queries |
| `pith/replay.py` (new) | one provider call (sync), endpoint/header selection, response-text extraction (JSON and SSE), per-call cost |
| `pith/judge.py` (new) | prompt, request building, label parsing, order normalization, `judge()` |
| `pith/sweep.py` (new) | eligibility, stratified sampling, targets, cost estimate, `run_sweep`, `pin_rule`, `recheck` |
| `pith/__main__.py` (modify) | `serve` / `sweep` / `recheck` subcommands, exit codes |
| `pith/proxy.py` (modify) | `GET /optimizer/sweeps/{route_key}` |
| `tests/test_config.py`, `test_db.py`, `test_report.py`, `test_proxy.py` (modify); `tests/test_replay.py`, `test_judge.py`, `test_sweep.py`, `test_main.py` (new); `tests/live/test_sweep_demo.py` (new) | tests |
| `README.md`, `pith.example.toml` (modify) | docs |

---

### Task 1: Config `prices`, drop `shadow_rate`, `price_for`

**Files:**
- Modify: `pith/config.py`, `pith/report.py`, `pith.example.toml`
- Test: `tests/test_config.py`, `tests/test_report.py`

**Interfaces:**
- Produces: `Config.prices: dict[str, tuple[float, float]]` (no `shadow_rate` field); `report.price_for(model: str, overrides: Mapping[str, tuple[float, float]] | None = None) -> tuple[float, float] | None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, change `test_defaults_match_spec`: delete the line `assert c.shadow_rate == 0.02` and add `assert c.prices == {}` and `assert not hasattr(c, "shadow_rate")`. Append:

```python
def test_prices_table_parsed(tmp_path):
    p = tmp_path / "pith.toml"
    p.write_text('[prices."gpt-5"]\ninput = 1.25\noutput = 10\n[prices."my-model"]\ninput = 0.5\noutput = 2.5\n')
    c = load_config(str(p), env={})
    assert c.prices == {"gpt-5": (1.25, 10.0), "my-model": (0.5, 2.5)}
```

In `tests/test_report.py` add the import `from pith.report import PRICES, price_for, render_html, route_rows` (replacing the existing import line) and append:

```python
def test_price_for_prefers_override_then_builtin():
    assert price_for("claude-opus-5-5") == PRICES["claude-opus-5-5"]
    assert price_for("claude-opus-5-5", {"claude-opus-5-5": (1.0, 2.0)}) == (1.0, 2.0)
    assert price_for("gpt-5", {"gpt-5": (1.25, 10.0)}) == (1.25, 10.0)
    assert price_for("gpt-5") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py tests/test_report.py -q`
Expected: FAIL — `AttributeError`/`AssertionError` on `prices`, `ImportError: cannot import name 'price_for'`.

- [ ] **Step 3: Implement**

`pith/config.py` — replace the `Config` dataclass and the loader body:

```python
@dataclass
class Config:
    listen: str = "0.0.0.0:8787"
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"
    db_path: str = "./pith.db"
    sample_rate: float = 0.05
    retention_days: int = 14
    sweep_budget_usd_month: float = 0.0
    equivalence_bar: float = 0.95
    judge_model: str = "claude-sonnet-5-5"
    judge_provider: str = "anthropic"
    enabled: bool = True
    routes: dict[str, RouteConfig] = field(default_factory=dict)
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)  # $/M tokens (input, output)
```

and in `load_config` change the scalar filter and add price parsing:

```python
    scalar = {f.name: f.type for f in fields(Config) if f.name not in ("routes", "prices")}
    ...
    for model, p in (data.get("prices") or {}).items():
        cfg.prices[model] = (float(p["input"]), float(p["output"]))
    return cfg
```

`pith/report.py` — add after `PRICES`:

```python
def price_for(model: str, overrides=None) -> tuple[float, float] | None:
    """($/M input, $/M output): config [prices] override first, then the built-in table."""
    return (overrides or {}).get(model) or PRICES.get(model)
```

`pith.example.toml` — delete the `shadow_rate` line and add, before the `[routes...]` block:

```toml
[prices."gpt-5"]            # $/M tokens; extends/overrides the built-in Claude table (illustrative values — check current pricing)
input = 1.25
output = 10.0
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add pith/config.py pith/report.py pith.example.toml tests/test_config.py tests/test_report.py
git commit -m "feat: price overrides in config; retire shadow_rate"
```

---

### Task 2: DB — hash-change unpin and control-plane queries

**Files:**
- Modify: `pith/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces (all in `pith.db`):
  - `TEXT_STOPS = ("end_turn", "stop", "completed")`
  - `upsert_route(...)` now resets `pinned_profile='P0'`, `status='observing'` and updates `model`/`system_hash` when the hash changes.
  - `route_request_stats(conn, key) -> dict` with `p0_total`, `p0_sampled` (P0 rows with `body_ref`), `text_frac` (fraction of P0 rows with `stop_reason IN TEXT_STOPS`, `None` if `p0_total == 0`).
  - `sample_candidates(conn, key, limit=500) -> list[dict]` rows `id, output_tokens, request_json` — P0, `body_ref` set, text-ending stop, newest first.
  - `create_sample(conn, key, item_ids: list[int], now=None) -> int`
  - `create_sweep(conn, key, sample_id, judge_model, judge_prompt_version, now=None) -> int`
  - `update_sweep_cost(conn, sweep_id, cost_usd)`; `finish_sweep(conn, sweep_id, cost_usd, result_json: str, winner: str | None, now=None)`
  - `add_judgment(conn, sweep_id, item_id, profile, trial, label, order_ab)`
  - `month_sweep_cost(conn, now) -> float`
  - `set_route_fields(conn, key, **fields)` — allowed keys only: `status eligible target_words exemplar last_sweep_id`
  - `sweeps_for_route(conn, key) -> list[dict]` (newest first)
  - `recent_bodies(conn, key, profile, n) -> list[dict]` rows `id, request_json, response_json`, newest first
  - `add_shadow(conn, key, request_id, label, now=None)`; `shadow_rate(conn, key, window=100) -> tuple[float | None, int]` — `(equivalent / (n − judge_error), n)`
  - `projected_monthly_volume(conn, key, now) -> int` — `max(1000, round(count of requests in last 7 days × 30/7))`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_db.py`)

```python
import json
import time


def _req(conn, key, profile="P0", out=100, stop="end_turn", body_ref=None, ts=None):
    return db.record_request(conn, ts=ts or time.time(), route_key=key, profile=profile, input_tokens=50,
                             output_tokens=out, cache_read=0, cache_create=0, estimated=False, stop_reason=stop,
                             latency_ms=1, body_ref=body_ref)


def test_upsert_route_unpins_on_hash_change_only():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "named", "anthropic", "m1", "hash-a")
    db.set_pin(conn, "named", "P2")
    db.upsert_route(conn, "named", "anthropic", "m1", "hash-a")
    assert db.get_route(conn, "named")["pinned_profile"] == "P2"
    db.upsert_route(conn, "named", "anthropic", "m2", "hash-b")
    r = db.get_route(conn, "named")
    assert (r["pinned_profile"], r["status"], r["system_hash"], r["model"]) == ("P0", "observing", "hash-b", "m2")


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
    assert n == 4 and rate == pytest.approx(2 / 3)
    now = time.time()
    for i in range(14):
        _req(conn, "k", ts=now - i * 3600)         # 14 requests in the last day
    _req(conn, "k", ts=now - 10 * 86400)           # outside the window
    assert db.projected_monthly_volume(conn, "k", now) == 1000  # floored
    for i in range(700):
        _req(conn, "k", ts=now - 60 - i)
    assert db.projected_monthly_volume(conn, "k", now) == round(714 * 30 / 7)
```

Add `import pytest` at the top of `tests/test_db.py` if it is not already there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_db.py -q`
Expected: FAIL with `AttributeError: module 'pith.db' has no attribute 'route_request_stats'` (and the hash-change test failing on `pinned_profile`).

- [ ] **Step 3: Implement**

In `pith/db.py` replace `upsert_route` and append the new functions:

```python
TEXT_STOPS = ("end_turn", "stop", "completed")
ROUTE_FIELDS = ("status", "eligible", "target_words", "exemplar", "last_sweep_id")


def upsert_route(conn, key, provider, model, system_hash, name=None, now=None):
    now = time.time() if now is None else now
    # A changed prompt/tools hash means the pinned profile was tuned for a different route: unpin, keylessly.
    conn.execute(
        "INSERT INTO routes(key, provider, model, system_hash, name, first_seen, last_seen) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET last_seen=excluded.last_seen, model=excluded.model, "
        "pinned_profile=CASE WHEN routes.system_hash != excluded.system_hash THEN 'P0' ELSE routes.pinned_profile END, "
        "status=CASE WHEN routes.system_hash != excluded.system_hash THEN 'observing' ELSE routes.status END, "
        "system_hash=excluded.system_hash",
        (key, provider, model, system_hash, name, now, now))
    conn.commit()


def route_request_stats(conn, key) -> dict:
    total, sampled, text = conn.execute(
        "SELECT COUNT(*), SUM(body_ref IS NOT NULL), SUM(stop_reason IN (?,?,?)) FROM requests "
        "WHERE route_key=? AND profile='P0'", (*TEXT_STOPS, key)).fetchone()
    return {"p0_total": total, "p0_sampled": sampled or 0, "text_frac": (text or 0) / total if total else None}


def sample_candidates(conn, key, limit=500) -> list[dict]:
    rows = conn.execute(
        "SELECT r.id, r.output_tokens, b.request_json FROM requests r JOIN bodies b ON b.id = r.body_ref "
        "WHERE r.route_key=? AND r.profile='P0' AND r.stop_reason IN (?,?,?) ORDER BY r.ts DESC LIMIT ?",
        (key, *TEXT_STOPS, limit)).fetchall()
    return [dict(r) for r in rows]


def create_sample(conn, key, item_ids, now=None) -> int:
    cur = conn.execute("INSERT INTO samples(route_key, created_at, item_ids_json) VALUES(?,?,?)",
                       (key, time.time() if now is None else now, json.dumps(list(item_ids))))
    conn.commit()
    return cur.lastrowid


def create_sweep(conn, key, sample_id, judge_model, judge_prompt_version, now=None) -> int:
    cur = conn.execute(
        "INSERT INTO sweeps(route_key, sample_id, judge_model, judge_prompt_version, started_at, cost_usd) "
        "VALUES(?,?,?,?,?,0)", (key, sample_id, judge_model, judge_prompt_version, time.time() if now is None else now))
    conn.commit()
    return cur.lastrowid


def update_sweep_cost(conn, sweep_id, cost_usd):
    conn.execute("UPDATE sweeps SET cost_usd=? WHERE id=?", (cost_usd, sweep_id))
    conn.commit()


def finish_sweep(conn, sweep_id, cost_usd, result_json, winner, now=None):
    conn.execute("UPDATE sweeps SET cost_usd=?, result_json=?, winner=?, finished_at=? WHERE id=?",
                 (cost_usd, result_json, winner, time.time() if now is None else now, sweep_id))
    conn.commit()


def add_judgment(conn, sweep_id, item_id, profile, trial, label, order_ab):
    conn.execute("INSERT INTO judgments(sweep_id, item_id, profile, trial, label, order_ab) VALUES(?,?,?,?,?,?)",
                 (sweep_id, item_id, profile, trial, label, order_ab))
    conn.commit()


def month_sweep_cost(conn, now) -> float:
    """Sweep spend since the start of the UTC month containing `now`, not counting anything after `now`."""
    start = datetime.fromtimestamp(now, timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM sweeps WHERE started_at >= ? AND started_at <= ?",
                       (start, now)).fetchone()
    return float(row[0])


def set_route_fields(conn, key, **fields):
    bad = set(fields) - set(ROUTE_FIELDS)
    if bad:
        raise ValueError(f"not settable here: {sorted(bad)}")
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE routes SET {sets} WHERE key=?", (*fields.values(), key))
    conn.commit()


def sweeps_for_route(conn, key) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM sweeps WHERE route_key=? ORDER BY id DESC", (key,))]


def recent_bodies(conn, key, profile, n) -> list[dict]:
    rows = conn.execute(
        "SELECT r.id, b.request_json, b.response_json FROM requests r JOIN bodies b ON b.id = r.body_ref "
        "WHERE r.route_key=? AND r.profile=? ORDER BY r.ts DESC LIMIT ?", (key, profile, n)).fetchall()
    return [dict(r) for r in rows]


def add_shadow(conn, key, request_id, label, now=None):
    conn.execute("INSERT INTO shadow(ts, route_key, request_id, label) VALUES(?,?,?,?)",
                 (time.time() if now is None else now, key, request_id, label))
    conn.commit()


def shadow_rate(conn, key, window=100):
    labels = [r[0] for r in conn.execute(
        "SELECT label FROM shadow WHERE route_key=? ORDER BY id DESC LIMIT ?", (key, window))]
    judged = [l for l in labels if l != "judge-error"]
    rate = (sum(l == "equivalent" for l in judged) / len(judged)) if judged else None
    return rate, len(labels)


def projected_monthly_volume(conn, key, now) -> int:
    n = conn.execute("SELECT COUNT(*) FROM requests WHERE route_key=? AND ts >= ?", (key, now - 7 * 86400)).fetchone()[0]
    return max(1000, round(n * 30 / 7))
```

Add `import json` and `from datetime import datetime, timezone` at the top of `pith/db.py`.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add pith/db.py tests/test_db.py
git commit -m "feat: hash-change unpin and control-plane queries"
```

---

### Task 3: `replay.py` — one provider call, text extraction, cost

**Files:**
- Create: `pith/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Produces:
  - `Reply` dataclass: `status: int` (0 = transport failure), `body: dict | None`, `text: str`, `usage: Usage`, `cost_usd: float`
  - `endpoint_for(provider: str, body: dict) -> str` — `"/v1/messages"`; OpenAI `"/v1/chat/completions"` if `"messages" in body` else `"/v1/responses"`
  - `auth_headers(provider: str, key: str) -> dict`
  - `response_text(provider: str, body: dict) -> str`
  - `sse_text(provider: str, raw: str) -> str` — reassembles text from a stored SSE transcript
  - `stored_response_text(provider: str, response_json: str) -> str` — JSON or SSE, whichever the stored body is
  - `cost_of(model: str, usage: Usage, prices) -> float` — `(input_tokens × in + output_tokens × out) / 1e6`; 0.0 if unpriced or tokens missing
  - `call(client: httpx.Client, cfg: Config, provider: str, body: dict, key: str, prices=None, sleep=time.sleep) -> Reply` — strips `stream`, POSTs, retries a 429 once after `retry-after` (capped at 60 s); any `httpx.HTTPError` → `Reply(status=0, ...)`.
- Consumes: `providers.upstream`, `usage.usage_from_body`, `report.price_for`, `config.Config`.

- [ ] **Step 1: Write the failing tests**

`tests/test_replay.py`:
```python
import json

import httpx

from pith.config import Config
from pith.replay import Reply, auth_headers, call, cost_of, endpoint_for, response_text, sse_text, stored_response_text
from pith.usage import Usage

ANTH = {"id": "m", "stop_reason": "end_turn", "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}],
        "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}
CHAT = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
RESP = {"object": "response", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "yo"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 1}}


def test_endpoint_and_headers():
    assert endpoint_for("anthropic", {"messages": []}) == "/v1/messages"
    assert endpoint_for("openai", {"messages": []}) == "/v1/chat/completions"
    assert endpoint_for("openai", {"input": "q"}) == "/v1/responses"
    assert auth_headers("anthropic", "k") == {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    assert auth_headers("openai", "k") == {"authorization": "Bearer k"}


def test_response_text_all_shapes():
    assert response_text("anthropic", ANTH) == "Hello"
    assert response_text("openai", CHAT) == "hi"
    assert response_text("openai", RESP) == "yo"
    assert response_text("openai", {"object": "response", "output_text": "direct"}) == "direct"
    assert response_text("anthropic", {}) == ""


def test_sse_and_stored_text():
    sse = ('event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"a"}}\n\n'
           'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"b"}}\n\n')
    assert sse_text("anthropic", sse) == "ab"
    chat = 'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: {"choices":[{"delta":{"content":"y"}}]}\n\ndata: [DONE]\n\n'
    assert sse_text("openai", chat) == "xy"
    assert stored_response_text("anthropic", json.dumps(ANTH)) == "Hello"
    assert stored_response_text("anthropic", sse) == "ab"
    assert stored_response_text("anthropic", "not json at all") == ""


def test_cost_of():
    assert cost_of("claude-opus-5-5", Usage(100, 10, 0, 0, "end_turn"), None) == (100 * 4 + 10 * 20) / 1e6
    assert cost_of("unknown", Usage(100, 10, 0, 0, "end_turn"), None) == 0.0
    assert cost_of("x", Usage(100, 10, 0, 0, "end_turn"), {"x": (1.0, 1.0)}) == 110 / 1e6
    assert cost_of("claude-opus-5-5", Usage(None, None, None, None, None), None) == 0.0


def test_call_strips_stream_and_prices():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json=ANTH)
    client = httpx.Client(transport=httpx.MockTransport(h))
    r = call(client, Config(), "anthropic", {"model": "claude-opus-5-5", "stream": True, "messages": []}, "k")
    assert isinstance(r, Reply) and r.status == 200 and r.text == "Hello"
    assert r.usage.output_tokens == 10 and r.cost_usd == (100 * 4 + 10 * 20) / 1e6
    assert "stream" not in json.loads(seen[0].content)
    assert seen[0].headers["x-api-key"] == "k" and str(seen[0].url) == "https://api.anthropic.com/v1/messages"


def test_call_retries_429_once_then_returns():
    n = []
    def h(req):
        n.append(1)
        return httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow"}) if len(n) == 1 else httpx.Response(200, json=CHAT)
    slept = []
    client = httpx.Client(transport=httpx.MockTransport(h))
    r = call(client, Config(), "openai", {"model": "gpt-5", "messages": []}, "k", sleep=slept.append)
    assert r.status == 200 and r.text == "hi" and slept == [7.0] and len(n) == 2
    n.clear(); slept.clear()
    def always(req):
        n.append(1)
        return httpx.Response(429, headers={"retry-after": "500"}, json={})
    r = call(httpx.Client(transport=httpx.MockTransport(always)), Config(), "openai", {"model": "gpt-5", "messages": []}, "k", sleep=slept.append)
    assert r.status == 429 and slept == [60.0] and len(n) == 2


def test_call_transport_error_is_status_zero():
    def h(req):
        raise httpx.ConnectError("down", request=req)
    r = call(httpx.Client(transport=httpx.MockTransport(h)), Config(), "anthropic", {"model": "m", "messages": []}, "k")
    assert r.status == 0 and r.body is None and r.text == "" and r.cost_usd == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_replay.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pith.replay'`.

- [ ] **Step 3: Implement**

`pith/replay.py`:
```python
"""One synchronous provider call for the control plane, plus response-text extraction and per-call cost."""
import json
import logging
import time
from dataclasses import dataclass

import httpx

from pith.config import Config
from pith.providers import upstream
from pith.report import price_for
from pith.usage import Usage, usage_from_body

log = logging.getLogger("pith.replay")


@dataclass
class Reply:
    status: int          # 0 = transport failure
    body: dict | None
    text: str
    usage: Usage
    cost_usd: float


def endpoint_for(provider: str, body: dict) -> str:
    if provider == "anthropic":
        return "/v1/messages"
    return "/v1/chat/completions" if "messages" in body else "/v1/responses"


def auth_headers(provider: str, key: str) -> dict:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"authorization": f"Bearer {key}"}


def response_text(provider: str, body: dict) -> str:
    if provider == "anthropic":
        return "".join(b.get("text", "") for b in body.get("content") or [] if b.get("type") == "text")
    if body.get("object") == "response" or "output" in body or "output_text" in body:
        if body.get("output_text"):
            return body["output_text"]
        parts = []
        for item in body.get("output") or []:
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
        return "".join(parts)
    choices = body.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


def sse_text(provider: str, raw: str) -> str:
    parts = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        if provider == "anthropic":
            d = ev.get("delta") or {}
            if ev.get("type") == "content_block_delta" and d.get("type") == "text_delta":
                parts.append(d.get("text", ""))
        else:
            for c in ev.get("choices") or []:
                parts.append((c.get("delta") or {}).get("content") or "")
            if ev.get("type") == "response.output_text.delta":
                parts.append(ev.get("delta", ""))
    return "".join(parts)


def stored_response_text(provider: str, response_json: str) -> str:
    s = response_json.lstrip()
    if s.startswith("{"):
        try:
            return response_text(provider, json.loads(s))
        except Exception:
            return ""
    if s.startswith(("event:", "data:")):
        return sse_text(provider, s)
    return ""


def cost_of(model: str, usage: Usage, prices) -> float:
    p = price_for(model, prices)
    if not p or usage.input_tokens is None or usage.output_tokens is None:
        return 0.0
    return (usage.input_tokens * p[0] + usage.output_tokens * p[1]) / 1e6


def call(client: httpx.Client, cfg: Config, provider: str, body: dict, key: str, prices=None, sleep=time.sleep) -> Reply:
    body = {k: v for k, v in body.items() if k != "stream"}
    url = upstream(provider, cfg) + endpoint_for(provider, body)
    headers = {**auth_headers(provider, key), "content-type": "application/json"}
    empty = Usage(None, None, None, None, None)
    try:
        resp = client.post(url, headers=headers, content=json.dumps(body).encode())
        if resp.status_code == 429:
            wait = min(60.0, float(resp.headers.get("retry-after", "5") or 5))
            sleep(wait)
            resp = client.post(url, headers=headers, content=json.dumps(body).encode())
    except httpx.HTTPError as exc:
        log.warning("replay to %s failed: %r", url, exc)
        return Reply(0, None, "", empty, 0.0)
    try:
        data = resp.json()
    except Exception:
        data = None
    if resp.status_code >= 400 or not isinstance(data, dict):
        return Reply(resp.status_code, data if isinstance(data, dict) else None, "", empty, 0.0)
    usage = usage_from_body(provider, data)
    return Reply(resp.status_code, data, response_text(provider, data), usage, cost_of(body.get("model", ""), usage, prices))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_replay.py -q -W error`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add pith/replay.py tests/test_replay.py
git commit -m "feat: replay module — provider call, text extraction, cost"
```

---

### Task 4: `judge.py`

**Files:**
- Create: `pith/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Produces:
  - `JUDGE_PROMPT_VERSION = "v1"`, `LABELS = ("equivalent", "missing-info", "contradiction", "format-broken")`, `SYSTEM_PROMPT`
  - `last_user_text(provider: str, body: dict, limit: int = 4000) -> str`
  - `build_judge_request(provider: str, model: str, question: str, answer_a: str, answer_b: str) -> dict`
  - `parse_label(text: str) -> str | None`
  - `normalize(label: str, order_ab: str) -> str` — `baseline-first`: unchanged; `candidate-first`: `missing-info → extra-info`, `format-broken → judge-error`, others unchanged
  - `class JudgeUnavailable(Exception)`
  - `judge(client, cfg, keys: dict, question: str, baseline: str, candidate: str, rng: random.Random, prices=None) -> tuple[str, str, float]` — `(normalized label, order_ab, cost_usd)`; raises `JudgeUnavailable` when the judge call's status is 0 or ≥ 400 (after the single retry allowed for unparseable replies, not for errors).
- Consumes: `replay.call`, `replay.Reply`.

- [ ] **Step 1: Write the failing tests**

`tests/test_judge.py`:
```python
import random

import httpx
import pytest

from pith.config import Config
from pith.judge import (JUDGE_PROMPT_VERSION, LABELS, SYSTEM_PROMPT, JudgeUnavailable, build_judge_request,
                             judge, last_user_text, normalize, parse_label)


def test_constants():
    assert JUDGE_PROMPT_VERSION == "v1"
    assert LABELS == ("equivalent", "missing-info", "contradiction", "format-broken")
    assert "exactly one label" in SYSTEM_PROMPT


def test_last_user_text_shapes_and_truncation():
    assert last_user_text("anthropic", {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                                     {"role": "user", "content": [{"type": "text", "text": "c"}]}]}) == "c"
    assert last_user_text("openai", {"messages": [{"role": "user", "content": "q"}]}) == "q"
    assert last_user_text("openai", {"input": "plain"}) == "plain"
    assert last_user_text("openai", {"input": [{"role": "user", "content": [{"type": "input_text", "text": "it"}]}]}) == "it"
    assert last_user_text("anthropic", {"messages": [{"role": "user", "content": "x" * 5000}]}) == "x" * 4000
    assert last_user_text("anthropic", {"messages": []}) == ""


def test_build_request_both_providers():
    a = build_judge_request("anthropic", "claude-sonnet-5-5", "Q", "A1", "B1")
    assert a["model"] == "claude-sonnet-5-5" and a["system"] == SYSTEM_PROMPT and a["max_tokens"] >= 256
    assert a["output_config"] == {"effort": "low"}
    user = a["messages"][0]["content"]
    assert "REQUEST:\nQ" in user and "ANSWER A:\nA1" in user and "ANSWER B:\nB1" in user
    for lab in LABELS:
        assert lab in user
    o = build_judge_request("openai", "gpt-5", "Q", "A1", "B1")
    assert o["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT} and o["messages"][1]["role"] == "user"
    assert "max_completion_tokens" in o and "max_tokens" not in o


def test_parse_label_first_match_case_insensitive():
    assert parse_label("Equivalent") == "equivalent"
    assert parse_label("Label: missing-info. Also contradiction.") == "missing-info"
    assert parse_label("I think it is a CONTRADICTION") == "contradiction"
    assert parse_label("format-broken") == "format-broken"
    assert parse_label("nothing here") is None
    assert parse_label("") is None


def test_normalize_by_order():
    assert normalize("equivalent", "baseline-first") == "equivalent"
    assert normalize("missing-info", "baseline-first") == "missing-info"
    assert normalize("missing-info", "candidate-first") == "extra-info"
    assert normalize("format-broken", "candidate-first") == "judge-error"
    assert normalize("contradiction", "candidate-first") == "contradiction"
    assert normalize("judge-error", "candidate-first") == "judge-error"


def _client(replies):
    it = iter(replies)
    def h(req):
        return next(it)(req)
    return httpx.Client(transport=httpx.MockTransport(h))


def anth(text):
    return lambda req: httpx.Response(200, json={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                                                 "usage": {"input_tokens": 300, "output_tokens": 3}})


def test_judge_orders_randomize_and_cost_accumulates():
    cfg = Config()
    seen = []
    def h(req):
        seen.append(req.content.decode())
        return httpx.Response(200, json={"content": [{"type": "text", "text": "equivalent"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 300, "output_tokens": 3}})
    client = httpx.Client(transport=httpx.MockTransport(h))
    orders = set()
    for seed in range(12):
        label, order, cost = judge(client, cfg, {"anthropic": "k"}, "Q", "BASE", "CAND", random.Random(seed))
        assert label == "equivalent" and cost == (300 * 2 + 3 * 10) / 1e6
        orders.add(order)
        body = seen[-1]
        if order == "baseline-first":
            assert body.index("ANSWER A:\\nBASE") < body.index("ANSWER B:\\nCAND")
        else:
            assert body.index("ANSWER A:\\nCAND") < body.index("ANSWER B:\\nBASE")
    assert orders == {"baseline-first", "candidate-first"}


def test_judge_retries_unparseable_once_then_judge_error():
    client = _client([anth("hmm"), anth("still nothing")])
    label, _, cost = judge(client, Config(), {"anthropic": "k"}, "Q", "B", "C", random.Random(1))
    assert label == "judge-error" and cost == 2 * (300 * 2 + 3 * 10) / 1e6
    client = _client([anth("??"), anth("missing-info")])
    label, order, _ = judge(client, Config(), {"anthropic": "k"}, "Q", "B", "C", random.Random(1))
    assert label == normalize("missing-info", order)


def test_judge_unavailable_on_error_or_transport_failure():
    client = _client([lambda req: httpx.Response(500, json={})])
    with pytest.raises(JudgeUnavailable):
        judge(client, Config(), {"anthropic": "k"}, "Q", "B", "C", random.Random(1))
    def boom(req):
        raise httpx.ConnectError("x", request=req)
    with pytest.raises(JudgeUnavailable):
        judge(_client([boom]), Config(), {"anthropic": "k"}, "Q", "B", "C", random.Random(1))
    with pytest.raises(JudgeUnavailable):
        judge(_client([]), Config(), {}, "Q", "B", "C", random.Random(1))  # no judge key
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_judge.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pith.judge'`.

- [ ] **Step 3: Implement**

`pith/judge.py`:
```python
"""The equivalence judge: one frozen prompt, one call, one label. Spec (Plan 2) §6."""
import random
import re

from pith.config import Config
from pith.replay import call

JUDGE_PROMPT_VERSION = "v1"
LABELS = ("equivalent", "missing-info", "contradiction", "format-broken")
SYSTEM_PROMPT = ("You compare two answers to the same request. Judge whether Answer B conveys every fact, decision and "
                 "required output that Answer A does, with no contradiction. Reply with exactly one label.")
USER_TEMPLATE = (
    "REQUEST:\n{question}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}\n\n"
    "Labels:\n"
    "equivalent - B conveys everything A does, nothing contradictory.\n"
    "missing-info - B omits a fact, decision or required output that A states.\n"
    "contradiction - B asserts something A denies, or vice versa.\n"
    "format-broken - B is empty, truncated, or not a usable answer.\n\n"
    "Reply with exactly one label.")
_LABEL_RE = re.compile("|".join(re.escape(l) for l in LABELS), re.IGNORECASE)


class JudgeUnavailable(Exception):
    pass


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") in ("text", "input_text"))
    return ""


def last_user_text(provider: str, body: dict, limit: int = 4000) -> str:
    items = body.get("messages")
    if items is None:
        inp = body.get("input", "")
        items = [{"role": "user", "content": inp}] if isinstance(inp, str) else inp
    for m in reversed(items or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return _text_of(m.get("content", ""))[:limit]
    return ""


def build_judge_request(provider: str, model: str, question: str, answer_a: str, answer_b: str) -> dict:
    user = USER_TEMPLATE.format(question=question, a=answer_a, b=answer_b)
    if provider == "anthropic":
        # Thinking tokens count toward max_tokens on current Claude models: leave room and keep effort low.
        return {"model": model, "max_tokens": 1024, "output_config": {"effort": "low"}, "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user}]}
    return {"model": model, "max_completion_tokens": 256,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}


def parse_label(text: str) -> str | None:
    m = _LABEL_RE.search(text or "")
    return m.group(0).lower() if m else None


def normalize(label: str, order_ab: str) -> str:
    if order_ab == "candidate-first":
        # A=candidate, B=baseline: "B omits" means the candidate added claims; "B broken" means the baseline is unusable.
        return {"missing-info": "extra-info", "format-broken": "judge-error"}.get(label, label)
    return label


def judge(client, cfg: Config, keys: dict, question: str, baseline: str, candidate: str, rng: random.Random,
          prices=None) -> tuple[str, str, float]:
    key = keys.get(cfg.judge_provider)
    if not key:
        raise JudgeUnavailable(f"no API key for judge provider {cfg.judge_provider!r}")
    order = "baseline-first" if rng.random() < 0.5 else "candidate-first"
    a, b = (baseline, candidate) if order == "baseline-first" else (candidate, baseline)
    body = build_judge_request(cfg.judge_provider, cfg.judge_model, question, a, b)
    cost = 0.0
    label = None
    for _ in range(2):  # one retry for an unparseable reply
        r = call(client, cfg, cfg.judge_provider, body, key, prices)
        cost += r.cost_usd
        if r.status == 0 or r.status >= 400:
            raise JudgeUnavailable(f"judge call failed with status {r.status}")
        label = parse_label(r.text)
        if label:
            break
    return normalize(label or "judge-error", order), order, cost
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_judge.py -q -W error`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add pith/judge.py tests/test_judge.py
git commit -m "feat: equivalence judge"
```

---

### Task 5: `sweep.py` part 1 — eligibility, sampling, targets

**Files:**
- Create: `pith/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Produces:
  - `PROFILES_BY_PROVIDER = {"anthropic": ("P0", "P1", "P2", "P3", "P4"), "openai": ("P0", "P1", "P1b", "P2", "P3", "P4")}`
  - `TEXT_GATE = 0.8`
  - `eligible_routes(conn, cfg, sample_n: int = 50, only: str | None = None) -> list[dict]` — routes with status `observing`/`reverted` (or the one named by `only`, any status except `sweeping`), `p0_sampled ≥ sample_n`, `text_frac ≥ TEXT_GATE`; sets `eligible=1`, or `status='not-applicable', eligible=0` when `p0_total ≥ sample_n` and the gate fails.
  - `stratify(candidates: list[dict], n: int) -> list[dict]` — up to `n` items across five `output_tokens` quintiles, round-robin so no quintile is over-represented while others have items
  - `pick_sample(conn, key, n) -> list[dict]` — read-only; each item is `{"id", "body"}` with `body` the parsed request JSON minus `stream`. (The `samples` row is written by `run_sweep` only after the budget gate passes.)
  - `derive_targets(p0_texts: list[str]) -> tuple[int, str]`
- Consumes: `db.route_request_stats`, `db.sample_candidates`, `db.set_route_fields`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sweep.py`:
```python
import json
import time

from pith import db
from pith.config import Config
from pith.sweep import PROFILES_BY_PROVIDER, derive_targets, eligible_routes, pick_sample, stratify


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sweep.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pith.sweep'`.

- [ ] **Step 3: Implement**

`pith/sweep.py` (part 1; Tasks 6–7 append to it):
```python
"""Control plane: eligibility, sampling, sweeps, pin rule, recheck. Spec (Plan 2) §4-§9."""
import json
import statistics

from pith import db
from pith.config import Config

PROFILES_BY_PROVIDER = {"anthropic": ("P0", "P1", "P2", "P3", "P4"), "openai": ("P0", "P1", "P1b", "P2", "P3", "P4")}
TEXT_GATE = 0.8
SWEEPABLE = ("observing", "reverted")


def eligible_routes(conn, cfg: Config, sample_n: int = 50, only: str | None = None) -> list[dict]:
    keys = [only] if only else [r["key"] for r in conn.execute("SELECT key FROM routes ORDER BY last_seen DESC")]
    out = []
    for key in keys:
        route = db.get_route(conn, key)
        if route is None or route["status"] == "sweeping" or (not only and route["status"] not in SWEEPABLE):
            continue
        s = db.route_request_stats(conn, key)
        if s["p0_total"] >= sample_n and (s["text_frac"] or 0) < TEXT_GATE:
            db.set_route_fields(conn, key, status="not-applicable", eligible=0)
            continue
        if s["p0_sampled"] < sample_n or (s["text_frac"] or 0) < TEXT_GATE:
            continue
        db.set_route_fields(conn, key, eligible=1)
        out.append(db.get_route(conn, key))
    return out


def stratify(candidates: list[dict], n: int) -> list[dict]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c["output_tokens"] or 0)
    size = max(1, -(-len(ordered) // 5))
    buckets = [ordered[i:i + size] for i in range(0, len(ordered), size)]
    picked = []
    while len(picked) < n and any(buckets):
        for b in buckets:
            if b and len(picked) < n:
                picked.append(b.pop(0))
    return picked


def pick_sample(conn, key: str, n: int) -> list[dict]:
    items = []
    for c in stratify(db.sample_candidates(conn, key), n):
        body = json.loads(c["request_json"])
        body.pop("stream", None)
        items.append({"id": c["id"], "body": body})
    return items


def derive_targets(p0_texts: list[str]) -> tuple[int, str]:
    texts = [t for t in p0_texts if t.strip()]
    if not texts:
        return 20, ""
    median = statistics.median(len(t.split()) for t in texts)
    return max(20, round(0.5 * median)), min(texts, key=lambda t: len(t.split()))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_sweep.py -q -W error`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pith/sweep.py tests/test_sweep.py
git commit -m "feat: sweep eligibility, stratified sampling, targets"
```

---

### Task 6: `sweep.py` part 2 — cost estimate, pin rule, `run_sweep`

**Files:**
- Modify: `pith/sweep.py` (append), `tests/test_sweep.py` (append)

**Interfaces:**
- Produces:
  - `class NoPrice(Exception)`, `class BudgetRefused(Exception)` (attrs `estimate`, `ceiling`), `class SweepAborted(Exception)`
  - `estimate_cost(route: dict, items: list[dict], profiles, trials: int, cfg: Config, *, mean_input: float, mean_output: float) -> float` — raises `NoPrice`
  - `MECHANICAL_FAILS = ("max_tokens", "length", "max_output_tokens")`
  - `pin_rule(table: dict[str, dict], bar: float, floor: float | None) -> str | None` — mutates each row to add `qualifies: bool` and `reason: str`; returns the cheapest qualifying candidate or `None`
  - `@dataclass SweepOutcome: winner: str | None; table: dict; floor: float | None; cost_usd: float; sweep_id: int | None; target_words: int; exemplar: str`
  - `run_sweep(conn, cfg, route: dict, client, keys: dict, *, trials=3, sample_n=50, dry_run=False, budget_usd=None, rng=None, now=None) -> SweepOutcome`
- Consumes: `replay.call`, `replay.Reply`, `judge.judge`, `judge.last_user_text`, `judge.JudgeUnavailable`, `judge.JUDGE_PROMPT_VERSION`, `rewrite.apply_profile`, `rewrite.RouteState`, `report.price_for`, `providers.is_responses_api`-style shape detection via `replay.endpoint_for`, `db.*` from Task 2.

**Table row shape** (per profile): `{"n", "skipped", "equivalent", "judged", "judge_error", "stops", "mean_input", "mean_output", "mean_cache_read", "rate", "cost_per_request"}`; `table["P0"]` has no amortization. After `pin_rule`: `"qualifies"`, `"reason"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sweep.py`)

```python
import random

import httpx
import pytest

from pith.sweep import (MECHANICAL_FAILS, BudgetRefused, NoPrice, SweepAborted, SweepOutcome, estimate_cost, pin_rule,
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
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(Script(judge_label="missing-info"))),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sweep.py -q`
Expected: FAIL with `ImportError: cannot import name 'estimate_cost'`.

- [ ] **Step 3: Implement** (append to `pith/sweep.py`; add the imports at the top of the file)

```python
import random
import time
from dataclasses import dataclass

from pith.judge import JUDGE_PROMPT_VERSION, JudgeUnavailable, judge, last_user_text
from pith.replay import Reply, call, endpoint_for
from pith.report import price_for
from pith.rewrite import RouteState, apply_profile

MECHANICAL_FAILS = ("max_tokens", "length", "max_output_tokens")
JUDGE_INPUT_OVERHEAD = 4000  # chars of request text the judge sees, used only to estimate judge cost
TRANSPORT_ABORT_FRAC = 0.2
BUDGET_CHECK_EVERY = 50


class NoPrice(Exception):
    pass


class BudgetRefused(Exception):
    def __init__(self, estimate: float, ceiling: float):
        super().__init__(f"estimated ${estimate:.4f} exceeds ceiling ${ceiling:.4f}")
        self.estimate, self.ceiling = estimate, ceiling


class SweepAborted(Exception):
    pass


@dataclass
class SweepOutcome:
    winner: str | None
    table: dict
    floor: float | None
    cost_usd: float
    sweep_id: int | None
    target_words: int
    exemplar: str


def estimate_cost(route: dict, items, profiles, trials: int, cfg: Config, *, mean_input: float, mean_output: float) -> float:
    p = price_for(route["model"], cfg.prices)
    jp = price_for(cfg.judge_model, cfg.prices)
    if not p:
        raise NoPrice(f"no price for model {route['model']!r}; add [prices.\"{route['model']}\"] to pith.toml")
    if not jp:
        raise NoPrice(f"no price for judge model {cfg.judge_model!r}")
    n = len(items)
    replays = n * trials * len(profiles) * (mean_input * p[0] + mean_output * p[1]) / 1e6
    candidates = len(profiles) - 1
    noise_pairs = 1
    judge_calls = n * trials * (candidates + noise_pairs)
    judges = judge_calls * ((JUDGE_INPUT_OVERHEAD + 2 * mean_output) * jp[0] + 8 * jp[1]) / 1e6
    return replays + judges


def pin_rule(table: dict, bar: float, floor: float | None) -> str | None:
    p0 = table["P0"]
    best = None
    for prof, row in table.items():
        if prof == "P0":
            row["qualifies"], row["reason"] = False, "baseline"
            continue
        reason = None
        if row.get("skipped"):
            reason = "skipped"
        elif row["rate"] is None or row["rate"] < bar:
            reason = "rate below bar"
        elif floor is not None and row["rate"] < floor - 0.03:
            reason = "rate below noise floor - 0.03"
        elif row["stops"] > p0["stops"]:
            reason = "more max_tokens stops than P0"
        elif (p0["mean_cache_read"] or 0) > 0 and (row["mean_cache_read"] or 0) < p0["mean_cache_read"]:
            reason = "cache reads below P0"
        elif row["cost_per_request"] >= p0["cost_per_request"]:
            reason = "not cheaper than P0"
        row["qualifies"], row["reason"] = reason is None, reason or "qualifies"
        if reason is None and (best is None or row["cost_per_request"] < table[best]["cost_per_request"]):
            best = prof
    return best


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def _mechanical_fail(r: Reply) -> bool:
    return r.status != 200 or r.usage.stop_reason in MECHANICAL_FAILS or not r.text.strip()


def run_sweep(conn, cfg: Config, route: dict, client, keys: dict, *, trials: int = 3, sample_n: int = 50,
              dry_run: bool = False, budget_usd: float | None = None, rng: random.Random | None = None,
              now: float | None = None) -> SweepOutcome:
    rng = rng or random.Random()
    now = time.time() if now is None else now
    key, provider = route["key"], route["provider"]
    pkey = keys.get(provider)
    if not pkey:
        raise SweepAborted(f"no API key for provider {provider!r}")
    profiles = PROFILES_BY_PROVIDER[provider]
    items = pick_sample(conn, key, sample_n)  # read-only until the budget gate passes
    if not items:
        raise SweepAborted("no sampleable requests")
    stats = conn.execute("SELECT AVG(input_tokens), AVG(output_tokens) FROM requests WHERE route_key=? AND profile='P0'",
                         (key,)).fetchone()
    estimate = estimate_cost(route, items, profiles, trials, cfg, mean_input=stats[0] or 0, mean_output=stats[1] or 0)
    # --budget-usd is an explicit per-run ceiling and overrides the monthly one; otherwise what is left of the month applies.
    ceiling = budget_usd if budget_usd is not None else cfg.sweep_budget_usd_month - db.month_sweep_cost(conn, now)
    if estimate > ceiling:
        raise BudgetRefused(estimate, ceiling)

    sample_id = None if dry_run else db.create_sample(conn, key, [it["id"] for it in items], now)
    sweep_id = None if dry_run else db.create_sweep(conn, key, sample_id, cfg.judge_model, JUDGE_PROMPT_VERSION, now)
    spent = {"usd": 0.0, "calls": 0}

    def account(cost: float):
        spent["usd"] += cost
        spent["calls"] += 1
        if spent["calls"] % BUDGET_CHECK_EVERY == 0:
            if sweep_id is not None:
                db.update_sweep_cost(conn, sweep_id, spent["usd"])
            if spent["usd"] > ceiling:
                raise SweepAborted(f"spent ${spent['usd']:.4f} over ceiling ${ceiling:.4f}")

    def replay_profile(profile: str, state: RouteState | None) -> tuple[dict[int, list[Reply]], bool]:
        """Returns {item_id: [reply per trial]} and whether the profile was skipped (no-op rewrite)."""
        replies: dict[int, list[Reply]] = {}
        failures = calls = 0
        for t in range(trials):
            for it in items:
                body = it["body"]
                if state is not None:
                    body = apply_profile(provider, body, state, responses_api=endpoint_for(provider, body) == "/v1/responses")
                    if body == it["body"]:
                        return {}, True
                r = call(client, cfg, provider, body, pkey, cfg.prices)
                account(r.cost_usd)
                calls += 1
                failures += r.status == 0
                replies.setdefault(it["id"], []).append(r)
        if calls and failures / calls > TRANSPORT_ABORT_FRAC:
            raise SweepAborted(f"{failures}/{calls} transport failures on {profile}")
        return replies, False

    def judged(item_id: int, profile: str, trial: int, label: str, order: str):
        if sweep_id is not None:
            db.add_judgment(conn, sweep_id, item_id, profile, trial, label, order)
        return label

    try:
        p0, _ = replay_profile("P0", None)
        questions = {it["id"]: last_user_text(provider, it["body"]) for it in items}
        baseline = {i: rs[0] for i, rs in p0.items()}
        target_words, exemplar = derive_targets([r.text for r in baseline.values() if not _mechanical_fail(r)])
        table: dict[str, dict] = {}

        def summarize(profile, replies, labels, skipped=False):
            flat = [r for rs in replies.values() for r in rs]
            judged_n = len(labels)
            errors = sum(l == "judge-error" for l in labels)
            eq = sum(l == "equivalent" for l in labels)
            rate = (eq / (judged_n - errors)) if judged_n - errors > 0 else None
            p = price_for(route["model"], cfg.prices)
            mi, mo = _mean([r.usage.input_tokens for r in flat]), _mean([r.usage.output_tokens for r in flat])
            table[profile] = {
                "n": len(replies), "skipped": skipped, "equivalent": eq, "judged": judged_n, "judge_error": errors,
                "stops": sum(r.usage.stop_reason in MECHANICAL_FAILS for r in flat),
                "mean_input": mi, "mean_output": mo, "mean_cache_read": _mean([r.usage.cache_read for r in flat]),
                "rate": rate, "cost_per_request": (mi * p[0] + mo * p[1]) / 1e6}

        noise = []
        for i, rs in p0.items():
            for t, r in enumerate(rs[1:], start=2):
                if _mechanical_fail(baseline[i]) or _mechanical_fail(r):
                    noise.append(judged(i, "P0", t, "judge-error", "n/a"))
                    continue
                label, order, cost = judge(client, cfg, keys, questions[i], baseline[i].text, r.text, rng, cfg.prices)
                account(cost)
                noise.append(judged(i, "P0", t, label, order))
        summarize("P0", p0, noise)
        floor = table["P0"]["rate"]

        for profile in profiles[1:]:
            state = RouteState(profile, route["injection_form"], target_words, exemplar)
            replies, skipped = replay_profile(profile, state)
            if skipped:
                summarize(profile, {}, [], skipped=True)
                continue
            labels = []
            for i, rs in replies.items():
                for t, r in enumerate(rs, start=1):
                    if _mechanical_fail(r):
                        labels.append(judged(i, profile, t, "format-broken", "n/a"))
                        continue
                    if _mechanical_fail(baseline[i]):
                        labels.append(judged(i, profile, t, "judge-error", "n/a"))
                        continue
                    label, order, cost = judge(client, cfg, keys, questions[i], baseline[i].text, r.text, rng, cfg.prices)
                    account(cost)
                    labels.append(judged(i, profile, t, label, order))
            summarize(profile, replies, labels)
        # Amortize the whole sweep's spend over the route's projected monthly volume, equally across candidates.
        amort = spent["usd"] / db.projected_monthly_volume(conn, key, now)
        for prof, row in table.items():
            if prof != "P0" and not row["skipped"]:
                row["cost_per_request"] += amort
    except JudgeUnavailable as exc:
        if sweep_id is not None:
            db.update_sweep_cost(conn, sweep_id, spent["usd"])
        raise SweepAborted(f"judge unavailable: {exc}") from exc
    except SweepAborted:
        if sweep_id is not None:
            db.update_sweep_cost(conn, sweep_id, spent["usd"])
        raise

    route_cfg = cfg.routes.get(key)
    bar = route_cfg.equivalence_bar if route_cfg and route_cfg.equivalence_bar is not None else cfg.equivalence_bar
    winner = pin_rule(table, bar, floor)
    result = {"table": table, "floor": floor, "bar": bar, "target_words": target_words, "exemplar": exemplar,
              "trials": trials, "sample_n": len(items)}
    if sweep_id is not None:
        db.finish_sweep(conn, sweep_id, spent["usd"], json.dumps(result), winner or "P0", now)
        if winner:
            db.set_pin(conn, key, winner)
            db.set_route_fields(conn, key, target_words=target_words, exemplar=exemplar, last_sweep_id=sweep_id)
        else:
            db.set_route_fields(conn, key, status="no-savings", last_sweep_id=sweep_id)
    return SweepOutcome(winner, table, floor, spent["usd"], sweep_id, target_words, exemplar)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_sweep.py -q -W error`
Expected: 13 passed. If `test_run_sweep_pins_cheapest_qualifying_and_records` fails on `winner`, print `out.table` — P1 (effort-down, same long text in the script) must be "not cheaper than P0" or equal; P2/P4 must qualify with `rate == 1.0`.

- [ ] **Step 5: Commit**

```bash
git add pith/sweep.py tests/test_sweep.py
git commit -m "feat: sweep runner, cost gate, pin rule"
```

---

### Task 7: `sweep.py` part 3 — `recheck`

**Files:**
- Modify: `pith/sweep.py` (append), `tests/test_sweep.py` (append)

**Interfaces:**
- Produces: `@dataclass RecheckOutcome: judged: int; rate: float | None; reverted: bool`; `RECHECK_MIN_ROWS = 20`; `recheck(conn, cfg, route: dict, client, keys: dict, *, n: int = 20, rng=None, now=None) -> RecheckOutcome`.
- Consumes: `db.recent_bodies`, `db.add_shadow`, `db.shadow_rate`, `db.set_pin`, `replay.call`, `replay.stored_response_text`, `judge.judge`, `judge.last_user_text`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sweep.py`)

```python
from pith.sweep import RECHECK_MIN_ROWS, RecheckOutcome, recheck


def _pinned_route_with_live(conn, n_live=25, live_text="short live answer"):
    db.upsert_route(conn, "k", "anthropic", "claude-opus-5-5", "h")
    db.set_pin(conn, "k", "P2")
    for i in range(n_live):
        ref = db.store_body(conn, json.dumps({"model": "claude-opus-5-5", "messages": [{"role": "user", "content": f"q{i}"}]}),
                            json.dumps({"content": [{"type": "text", "text": live_text}], "stop_reason": "end_turn", "usage": {}}), 9e9)
        db.record_request(conn, ts=time.time() - i, route_key="k", profile="P2", input_tokens=1, output_tokens=1, cache_read=0,
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
    out = recheck(conn, Config(), route, httpx.Client(transport=httpx.MockTransport(Script(judge_label="missing-info"))),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sweep.py -q -k recheck`
Expected: FAIL with `ImportError: cannot import name 'recheck'`.

- [ ] **Step 3: Implement** (append to `pith/sweep.py`; add `from pith.replay import stored_response_text` to the replay import line)

```python
RECHECK_MIN_ROWS = 20


@dataclass
class RecheckOutcome:
    judged: int
    rate: float | None
    reverted: bool


def recheck(conn, cfg: Config, route: dict, client, keys: dict, *, n: int = 20, rng: random.Random | None = None,
            now: float | None = None) -> RecheckOutcome:
    rng = rng or random.Random()
    now = time.time() if now is None else now
    key, provider, pinned = route["key"], route["provider"], route["pinned_profile"]
    if pinned == "P0":
        return RecheckOutcome(0, None, False)
    pkey = keys.get(provider)
    if not pkey:
        raise SweepAborted(f"no API key for provider {provider!r}")
    judged_n = 0
    try:
        for row in db.recent_bodies(conn, key, pinned, n):
            body = json.loads(row["request_json"])
            body.pop("stream", None)
            live = stored_response_text(provider, row["response_json"])
            p0 = call(client, cfg, provider, body, pkey, cfg.prices)
            if p0.status == 0:  # transport failure: this pair cannot be judged; write nothing (spec §11)
                continue
            if _mechanical_fail(p0) or not live.strip():
                db.add_shadow(conn, key, row["id"], "judge-error", now)
                continue
            label, _, _ = judge(client, cfg, keys, last_user_text(provider, body), p0.text, live, rng, cfg.prices)
            db.add_shadow(conn, key, row["id"], label, now)
            judged_n += 1
    except JudgeUnavailable as exc:
        raise SweepAborted(f"judge unavailable: {exc}") from exc
    rate, total = db.shadow_rate(conn, key)
    route_cfg = cfg.routes.get(key)
    bar = route_cfg.equivalence_bar if route_cfg and route_cfg.equivalence_bar is not None else cfg.equivalence_bar
    reverted = False
    if total >= RECHECK_MIN_ROWS and rate is not None and rate < bar:
        db.set_pin(conn, key, "P0", status="reverted")
        reverted = True
    return RecheckOutcome(judged_n, rate, reverted)
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add pith/sweep.py tests/test_sweep.py
git commit -m "feat: offline recheck with revert"
```

---

### Task 8: CLI subcommands

**Files:**
- Modify: `pith/__main__.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None, env: Mapping | None = None, conn=None, client=None) -> int` (exit code; `python -m pith` calls `sys.exit(main())`); `keys_from_env(env) -> dict` with only present providers; `format_table(route_key, outcome) -> str`.
- Exit codes: 0 completed; 2 budget refused / no price / no usable key; 1 aborted.
- Consumes: `sweep.eligible_routes`, `sweep.run_sweep`, `sweep.recheck`, the exception classes, `load_config`, `db.connect`, `create_app`.

- [ ] **Step 1: Write the failing tests**

`tests/test_main.py`:
```python
import json
import time

import httpx
import pytest

from pith import db
from pith.__main__ import format_table, keys_from_env, main
from pith.sweep import SweepOutcome


def test_keys_from_env():
    assert keys_from_env({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "X": "1"}) == {"anthropic": "a", "openai": "o"}
    assert keys_from_env({"OPENAI_API_KEY": ""}) == {}


def seed(conn, n=50):
    db.upsert_route(conn, "k", "anthropic", "claude-opus-5-5", "h")
    for i in range(n):
        ref = db.store_body(conn, json.dumps({"model": "claude-opus-5-5", "messages": [{"role": "user", "content": f"q{i}"}]}), "{}", 9e9)
        db.record_request(conn, ts=time.time() - i, route_key="k", profile="P0", input_tokens=10, output_tokens=10, cache_read=0,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_main.py -q`
Expected: FAIL with `ImportError: cannot import name 'format_table'`.

- [ ] **Step 3: Implement**

`pith/__main__.py` (replace the file):
```python
import argparse
import logging
import os
import sys

import httpx

from pith import db
from pith.config import load_config
from pith.sweep import BudgetRefused, NoPrice, SweepAborted, SweepOutcome, eligible_routes, recheck, run_sweep

ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def keys_from_env(env) -> dict:
    return {p: env[var] for p, var in ENV_KEYS.items() if env.get(var)}


def format_table(route_key: str, out: SweepOutcome) -> str:
    lines = [f"route {route_key}: winner: {out.winner or 'none (no-savings)'}  floor={out.floor}  cost=${out.cost_usd:.4f}  "
             f"target_words={out.target_words}"]
    lines.append(f"  {'profile':8} {'rate':>6} {'$/req':>10} {'out_tok':>8} {'stops':>5}  verdict")
    for prof, row in out.table.items():
        rate = "-" if row.get("rate") is None else f"{row['rate']:.2f}"
        lines.append(f"  {prof:8} {rate:>6} {row.get('cost_per_request', 0):>10.6f} {row.get('mean_output', 0):>8.1f} "
                     f"{row.get('stops', 0):>5}  {row.get('reason', '')}")
    return "\n".join(lines)


def _sweep(args, cfg, conn, client, env) -> int:
    keys = keys_from_env(env)
    if cfg.judge_provider not in keys:
        print(f"no judge key: set {ENV_KEYS[cfg.judge_provider]}")
        return 2
    routes = eligible_routes(conn, cfg, sample_n=args.sample, only=args.route)
    if not routes:
        print("no eligible routes" + (f" (route {args.route!r} not eligible or unknown)" if args.route else ""))
        return 0
    rc = 0
    for route in routes:
        if route["provider"] not in keys:
            print(f"route {route['key']}: skipped, set {ENV_KEYS[route['provider']]}")
            rc = 2
            continue
        try:
            out = run_sweep(conn, cfg, route, client, keys, trials=args.trials, sample_n=args.sample,
                            dry_run=args.dry_run, budget_usd=args.budget_usd)
        except BudgetRefused as e:
            print(f"route {route['key']}: refused — {e}")
            return 2
        except NoPrice as e:
            print(f"route {route['key']}: refused — {e}")
            return 2
        except SweepAborted as e:
            print(f"route {route['key']}: aborted — {e}")
            return 1
        print(format_table(route["key"], out) + ("  [dry-run: nothing written]" if args.dry_run else ""))
    return rc


def _recheck(args, cfg, conn, client, env) -> int:
    keys = keys_from_env(env)
    if cfg.judge_provider not in keys:
        print(f"no judge key: set {ENV_KEYS[cfg.judge_provider]}")
        return 2
    rows = conn.execute("SELECT key FROM routes WHERE pinned_profile != 'P0'" + (" AND key=?" if args.route else ""),
                        ((args.route,) if args.route else ())).fetchall()
    for (key,) in rows:
        route = db.get_route(conn, key)
        if route["provider"] not in keys:
            print(f"route {key}: skipped, set {ENV_KEYS[route['provider']]}")
            continue
        try:
            out = recheck(conn, cfg, route, client, keys, n=args.n)
        except SweepAborted as e:
            print(f"route {key}: aborted — {e}")
            return 1
        print(f"route {key}: judged={out.judged} rate={out.rate} reverted={out.reverted}")
    return 0


def main(argv=None, env=None, conn=None, client=None) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(prog="pith")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("serve", "sweep", "recheck"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None, help="path to pith.toml (env OPTIMIZER_* overrides it)")
        if name in ("sweep", "recheck"):
            p.add_argument("--route", default=None)
        if name == "sweep":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--budget-usd", type=float, default=None)
            p.add_argument("--trials", type=int, default=3)
            p.add_argument("--sample", type=int, default=50)
        if name == "recheck":
            p.add_argument("--n", type=int, default=20)
    args = ap.parse_args(argv)
    cmd = args.cmd or "serve"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(getattr(args, "config", None), env)
    conn = conn or db.connect(cfg.db_path)
    if cmd == "serve":
        import uvicorn
        from pith.proxy import create_app
        host, _, port = cfg.listen.rpartition(":")
        uvicorn.run(create_app(cfg, conn), host=host or "0.0.0.0", port=int(port), log_level="info")
        return 0
    client = client or httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
    return _sweep(args, cfg, conn, client, env) if cmd == "sweep" else _recheck(args, cfg, conn, client, env)


if __name__ == "__main__":
    sys.exit(main())
```

Note: `serve` with no subcommand still works (`python -m pith` → `cmd = "serve"`), but `--config` must now follow the subcommand: `python -m pith serve --config pith.toml`. Update the README's run line in Task 10.

- [ ] **Step 4: Run the tests and a smoke run**

Run: `.venv/bin/pytest tests/test_main.py -q -W error` — Expected: 5 passed.
Run: `OPTIMIZER_DB_PATH=/tmp/smoke.db .venv/bin/python -m pith sweep` — Expected: prints `no judge key: set ANTHROPIC_API_KEY`, exit code 2 (`echo $?`).

- [ ] **Step 5: Commit**

```bash
git add pith/__main__.py tests/test_main.py
git commit -m "feat: sweep and recheck CLI subcommands"
```

---

### Task 9: Report columns and sweep-audit endpoint

**Files:**
- Modify: `pith/report.py`, `pith/proxy.py`
- Test: `tests/test_report.py` (append), `tests/test_proxy.py` (append)

**Interfaces:**
- Produces: `route_rows(conn, prices=None)` rows gain `equivalence_pct`, `noise_floor_pct`, `sample_n`, `last_sweep_at`, `sweep_cost_usd`, `recheck_pct`; `sweep_rows(conn, key) -> list[dict]` with `result_json` parsed into `result`; `COLS` gains `equivalence_pct noise_floor_pct recheck_pct`; endpoint `GET /optimizer/sweeps/{route_key:path}` → `sweep_rows`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_report.py`:
```python
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
```

Append to `tests/test_proxy.py`:
```python
@pytest.mark.anyio
async def test_sweeps_endpoint_returns_audit_rows():
    app, conn, _ = make()
    db.upsert_route(conn, "anthropic:m:abc", "anthropic", "m", "h")
    sid = db.create_sample(conn, "anthropic:m:abc", [1])
    sw = db.create_sweep(conn, "anthropic:m:abc", sid, "j", "v1")
    db.finish_sweep(conn, sw, 0.1, json.dumps({"table": {}}), "P0")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/optimizer/sweeps/anthropic:m:abc")
        assert r.status_code == 200 and r.json()[0]["winner"] == "P0" and r.json()[0]["result"] == {"table": {}}
        assert (await c.get("/optimizer/sweeps/nope")).json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_report.py tests/test_proxy.py -q`
Expected: FAIL with `ImportError: cannot import name 'sweep_rows'` and `KeyError: 'equivalence_pct'`.

- [ ] **Step 3: Implement**

In `pith/report.py` replace `_usd_per_1k`, `route_rows`, `COLS` and add `sweep_rows`:
```python
import json


def _usd_per_1k(model: str, avg_in, avg_out, prices=None):
    p = price_for(model, prices)
    if not p or avg_in is None or avg_out is None:
        return None
    return (avg_in * p[0] + avg_out * p[1]) / 1e6 * 1000


def _pct(x):
    return None if x is None else 100.0 * x


def sweep_rows(conn, key) -> list[dict]:
    out = []
    for s in db.sweeps_for_route(conn, key):
        raw = s.pop("result_json", None)
        s["result"] = json.loads(raw) if raw else None
        out.append(s)
    return out


def route_rows(conn, prices=None) -> list[dict]:
    stats = {}
    for s in db.route_stats(conn):
        stats.setdefault(s["route_key"], {})[s["profile"]] = s
    rows = []
    for key in [r["key"] for r in conn.execute("SELECT key FROM routes ORDER BY last_seen DESC")]:
        route = db.get_route(conn, key)
        base = stats.get(key, {}).get("P0", {})
        pinned = stats.get(key, {}).get(route["pinned_profile"], {}) if route["pinned_profile"] != "P0" else {}
        b_usd = _usd_per_1k(route["model"], base.get("avg_input"), base.get("avg_output"), prices)
        p_usd = _usd_per_1k(route["model"], pinned.get("avg_input"), pinned.get("avg_output"), prices)
        sweep = None
        if route["last_sweep_id"]:
            row = conn.execute("SELECT * FROM sweeps WHERE id=?", (route["last_sweep_id"],)).fetchone()
            sweep = dict(row) if row else None
        result = json.loads(sweep["result_json"]) if sweep and sweep.get("result_json") else {}
        table = result.get("table") or {}
        chosen = table.get(sweep["winner"]) if sweep and sweep.get("winner") else None
        recheck_rate, _ = db.shadow_rate(conn, key)
        rows.append({
            "key": key, "provider": route["provider"], "model": route["model"], "name": route["name"],
            "status": route["status"], "pinned_profile": route["pinned_profile"],
            "baseline_n": base.get("n", 0), "baseline_avg_output": base.get("avg_output"),
            "pinned_n": pinned.get("n", 0), "pinned_avg_output": pinned.get("avg_output"),
            "baseline_usd_per_1k": b_usd, "pinned_usd_per_1k": p_usd,
            "estimated_savings_pct": (100 * (1 - p_usd / b_usd)) if b_usd and p_usd is not None else None,
            "equivalence_pct": _pct(chosen.get("rate")) if chosen else None,
            "noise_floor_pct": _pct(result.get("floor")) if result else None,
            "sample_n": result.get("sample_n") if result else None,
            "last_sweep_at": sweep.get("finished_at") if sweep else None,
            "sweep_cost_usd": sweep.get("cost_usd") if sweep else None,
            "recheck_pct": _pct(recheck_rate),
            "last_seen": route["last_seen"],
        })
    return rows


COLS = ("key", "name", "model", "status", "pinned_profile", "baseline_n", "baseline_avg_output", "pinned_n",
        "pinned_avg_output", "baseline_usd_per_1k", "pinned_usd_per_1k", "estimated_savings_pct",
        "equivalence_pct", "noise_floor_pct", "recheck_pct")
```

In `pith/proxy.py`: change the report import to `from pith.report import render_html, route_rows, sweep_rows`, pass prices in both report routes (`route_rows(conn, config.prices)`), and add before the catch-all:
```python
    @app.get("/optimizer/sweeps/{route_key:path}")
    async def sweeps(route_key: str):
        return sweep_rows(conn, route_key)
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add pith/report.py pith/proxy.py tests/test_report.py tests/test_proxy.py
git commit -m "feat: sweep columns in report and sweep audit endpoint"
```

---

### Task 10: README, live demo test

**Files:**
- Modify: `README.md`
- Create: `tests/live/test_sweep_demo.py`

**Interfaces:** none new. The live test is skipped unless `OPTIMIZER_LIVE=1` and `ANTHROPIC_API_KEY` are set.

- [ ] **Step 1: Write the live demo test**

`tests/live/test_sweep_demo.py`:
```python
"""Spec §11/§13 demo (Plan 2): a 50-item support-ticket route through the proxy, then a real sweep.

Run: OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live/test_sweep_demo.py -v -s
Cost: ~30 items × 4 profiles × 2 trials replays on claude-haiku-4-5 plus ~240 judge calls on claude-sonnet-5-5 ≈ $1–3.
"""
import json
import os
import random

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
TICKETS = [
    "Where is my order #1234? It was supposed to arrive yesterday.", "I want my money back for the broken blender.",
    "I can't log in, password reset email never arrives.", "Does the blue jacket come in XL?",
    "Your website is slow today.", "Package arrived damaged, the box was crushed.", "Charged twice for one order.",
    "How do I change my shipping address?", "Is the laptop stand compatible with a 16-inch MacBook?",
    "Cancel my subscription please.",
] * 3


@pytest.mark.anyio
async def test_demo_route_pins_a_profile_with_savings():
    conn = db.connect(":memory:")
    cfg = Config(sample_rate=1.0, sweep_budget_usd_month=5.0, equivalence_bar=0.9)
    app = create_app(cfg, conn, client=httpx.AsyncClient(timeout=120))
    hdrs = {"content-type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01", "X-Optimizer-Route": "demo-support"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=120) as c:
        for t in TICKETS:
            body = {"model": "claude-haiku-4-5", "max_tokens": 400, "system": SYSTEM,
                    "messages": [{"role": "user", "content": t}]}
            r = await c.post("/v1/messages", content=json.dumps(body).encode(), headers=hdrs)
            assert r.status_code == 200, r.text
    route = db.get_route(conn, "demo-support")
    out = run_sweep(conn, cfg, route, httpx.Client(timeout=120), {"anthropic": os.environ["ANTHROPIC_API_KEY"]},
                    trials=2, sample_n=30, rng=random.Random(0))
    print(json.dumps(out.table, indent=1, default=str))
    assert out.winner is not None, "no profile qualified — see table above"
    saved = 1 - out.table[out.winner]["mean_output"] / out.table["P0"]["mean_output"]
    assert saved >= 0.25, f"only {saved:.0%} fewer output tokens"
    assert out.table[out.winner]["rate"] >= 0.9
    assert db.get_route(conn, "demo-support")["pinned_profile"] == out.winner
```

Run: `.venv/bin/pytest tests/live -q` — Expected: 2 skipped.

- [ ] **Step 2: Update the README**

Replace the "Run" section's first command with `.venv/bin/python -m pith serve --config pith.toml`, and replace the "What you will see at first" section with:

```markdown
## Sweeps: turning observation into pins

The proxy never holds an API key, so sweeps run from the CLI with keys in its environment:

    export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY
    .venv/bin/python -m pith sweep --config pith.toml --dry-run     # spends, prints, writes nothing
    .venv/bin/python -m pith sweep --config pith.toml               # pins the cheapest profile that clears the bar

A route is swept once it has 50 sampled baseline requests (`sample_rate` controls sampling) and ≥80% text-ending
responses. The sweep replays the frozen sample under each profile, judges equivalence against the unconstrained
baseline (`judge_model`), and pins only a profile that is at least as consistent as the baseline is with itself.
`sweep_budget_usd_month = 0` (the default) refuses every sweep; set a ceiling, or pass `--budget-usd` per run.
Exit codes: 0 done, 2 refused (budget, price, or missing key), 1 aborted.

Drift: `python -m pith recheck` re-judges recent live responses on pinned routes and reverts a route to P0 when
its rolling equivalence falls below the bar. Run both from cron, e.g. a nightly `recheck` and a weekly `sweep`.

Audit: `GET /optimizer/sweeps/<route key>` returns every sweep for a route with its per-profile table.
Name a route explicitly with the request header `X-Optimizer-Route: <name>`.

The proxy has no authentication of its own — bind `listen` to a private interface. On first use of an OpenAI stream
without usage, token estimation downloads the `o200k_base` vocabulary once (set `TIKTOKEN_CACHE_DIR` to pre-seed it on
egress-filtered hosts; if the download fails the proxy falls back to a length estimate and flags the row as estimated).
```

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/pytest -q -W error -rs`
Expected: all passed, 2 skipped (both live tests).

- [ ] **Step 4: Commit**

```bash
git add README.md tests/live/test_sweep_demo.py
git commit -m "docs: sweep/recheck usage; live sweep demo test"
```
