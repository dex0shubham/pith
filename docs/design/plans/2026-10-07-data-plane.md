# Output-Token Optimizer — Plan 1: Data Plane

**Goal:** A self-hosted proxy for the Claude and OpenAI APIs that passes traffic through byte-faithfully, fingerprints routes, records usage, applies a pinned output profile per route with cache-safe rewriting, fails open on every error, and serves a per-route report. After this plan the product is a working observe-only proxy; Plan 2 (control plane: sweeps, judge, pin rule, drift) builds on the tables and interfaces defined here.

**Architecture:** One Python process: a FastAPI catch-all route forwards requests with `httpx` (streaming passthrough), a small set of pure modules (fingerprint, rewrite, usage) do the per-request logic, and stdlib `sqlite3` holds state. No ORM, no pydantic models, no background workers in this plan.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, httpx, tiktoken (OpenAI usage estimation only), stdlib `sqlite3` + `tomllib`, pytest + anyio (ships with httpx) for async tests.

**Spec:** `docs/design/specs/2026-10-07-output-token-optimizer-design.md` — read §3–§11 before starting; this plan argues from it.

## Global Constraints

- Python ≥ 3.12 (`tomllib` is stdlib). All code under `optimizer/`, tests under `tests/`.
- The proxy never edits top-level `system`, `tools`, `model`, `thinking`, `max_tokens`, `max_completion_tokens`, `max_output_tokens`, or `temperature` (spec §5, §7).
- Fail-open: any proxy-side exception before forwarding sends the customer's original bytes upstream unchanged (spec §8).
- `Authorization` and `x-api-key` headers are forwarded and never logged or stored (spec §8).
- Profiles are exactly `P0 P1 P1b P2 P3 P4` (spec §5). Statuses are exactly `observing pinned no-savings not-applicable sweeping reverted` (spec §3).
- Kill switches: request header `X-Optimizer: off`, config `routes.<key>.enabled = false`, env `OPTIMIZER_ENABLED=0` — all force P0 (spec §8). Header `X-Optimizer: bypass` means no rewrite **and** no recording (spec §6).
- Shape text is exactly: `Answer directly. No preamble, restatement, or closing summary. Target at most {n} words unless the task genuinely needs more.` with n ≥ 20 (spec §5).
- Tests: pytest only, one `test_*.py` per module, no fixture frameworks (spec §11). Every task ends with its tests green and a commit.

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | package metadata, deps, pytest config |
| `optimizer/__init__.py` | empty |
| `optimizer/config.py` | `Config` dataclass; load `optimizer.toml` + `OPTIMIZER_*` env overrides |
| `optimizer/db.py` | SQLite schema (all spec §10 tables) and the handful of query functions the proxy and report need |
| `optimizer/fingerprint.py` | route key from provider + body (+ header override) |
| `optimizer/providers.py` | provider detection by path; upstream base URL |
| `optimizer/rewrite.py` | profile definitions and cache-safe per-provider request rewriting |
| `optimizer/usage.py` | usage/stop-reason extraction from non-streaming bodies and SSE streams; tiktoken estimate |
| `optimizer/proxy.py` | FastAPI app: forward, stream, fail-open, kill switches, 4xx retry, recording, report endpoints |
| `optimizer/report.py` | per-route rows (JSON) and a static HTML table |
| `optimizer/__main__.py` | `python -m optimizer --config optimizer.toml` |
| `tests/conftest.py` | `anyio_backend` fixture (asyncio only) |
| `tests/test_<module>.py` | one per module |
| `tests/live/test_cache_safety.py` | standing live Claude test; skipped without `OPTIMIZER_LIVE=1` |
| `README.md` | install, run, config, kill switches |

---

### Task 1: Scaffold and config loader

**Files:**
- Create: `pyproject.toml`, `optimizer/__init__.py`, `optimizer/config.py`, `tests/conftest.py`, `tests/test_config.py`, `.gitignore`

**Interfaces:**
- Produces: `optimizer.config.Config` dataclass (fields below) and `load_config(path: str | None = None, env: Mapping[str, str] | None = None) -> Config`. `Config.routes` is `dict[str, RouteConfig]` with `RouteConfig(enabled: bool = True, equivalence_bar: float | None = None)`.

- [ ] **Step 1: Create the package skeleton**

`pyproject.toml`:
```toml
[project]
name = "output-optimizer"
version = "0.1.0"
description = "Self-hosted output-token control plane proxy for Claude and OpenAI APIs"
requires-python = ">=3.12"
dependencies = ["fastapi>=0.115", "uvicorn>=0.30", "httpx>=0.27", "tiktoken>=0.7"]

[project.optional-dependencies]
dev = ["pytest>=8", "anyio>=4"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["optimizer*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
.venv/
__pycache__/
*.db
*.egg-info/
```

`optimizer/__init__.py`: empty file.

`tests/conftest.py`:
```python
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
```

- [ ] **Step 2: Install into a venv**

Run:
```bash
python3 -m venv .venv && .venv/bin/pip install -q -e '.[dev]' && .venv/bin/python -c "import fastapi, httpx, tiktoken; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Write the failing config tests**

`tests/test_config.py`:
```python
from optimizer.config import Config, RouteConfig, load_config


def test_defaults_match_spec():
    c = load_config(None, env={})
    assert c.listen == "0.0.0.0:8787"
    assert c.anthropic_upstream == "https://api.anthropic.com"
    assert c.openai_upstream == "https://api.openai.com"
    assert c.db_path == "./optimizer.db"
    assert c.sample_rate == 0.05
    assert c.shadow_rate == 0.02
    assert c.retention_days == 14
    assert c.sweep_budget_usd_month == 0
    assert c.equivalence_bar == 0.95
    assert c.judge_model == "claude-sonnet-5-5"
    assert c.judge_provider == "anthropic"
    assert c.enabled is True
    assert c.routes == {}


def test_toml_and_route_overrides(tmp_path):
    p = tmp_path / "optimizer.toml"
    p.write_text(
        'listen = "127.0.0.1:9000"\nsample_rate = 0.5\n'
        '[routes."anthropic:claude-opus-5-5:abc"]\nenabled = false\nequivalence_bar = 0.97\n'
    )
    c = load_config(str(p), env={})
    assert c.listen == "127.0.0.1:9000"
    assert c.sample_rate == 0.5
    assert c.routes["anthropic:claude-opus-5-5:abc"] == RouteConfig(enabled=False, equivalence_bar=0.97)


def test_env_overrides_toml(tmp_path):
    p = tmp_path / "optimizer.toml"
    p.write_text('db_path = "/from/toml.db"\n')
    c = load_config(str(p), env={"OPTIMIZER_DB_PATH": "/from/env.db", "OPTIMIZER_ENABLED": "0",
                                 "OPTIMIZER_RETENTION_DAYS": "3"})
    assert c.db_path == "/from/env.db"
    assert c.enabled is False
    assert c.retention_days == 3


def test_unknown_toml_key_is_ignored(tmp_path):
    p = tmp_path / "optimizer.toml"
    p.write_text('not_a_field = 1\n')
    assert isinstance(load_config(str(p), env={}), Config)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'optimizer.config'`

- [ ] **Step 5: Implement config.py**

`optimizer/config.py`:
```python
"""Load optimizer.toml and OPTIMIZER_* environment overrides into a Config."""
import os
import tomllib
from dataclasses import dataclass, field, fields
from typing import Mapping


@dataclass(frozen=True)
class RouteConfig:
    enabled: bool = True
    equivalence_bar: float | None = None


@dataclass
class Config:
    listen: str = "0.0.0.0:8787"
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"
    db_path: str = "./optimizer.db"
    sample_rate: float = 0.05
    shadow_rate: float = 0.02
    retention_days: int = 14
    sweep_budget_usd_month: float = 0.0
    equivalence_bar: float = 0.95
    judge_model: str = "claude-sonnet-5-5"
    judge_provider: str = "anthropic"
    enabled: bool = True
    routes: dict[str, RouteConfig] = field(default_factory=dict)


def _coerce(kind, raw: str):
    if kind is bool:
        return raw.strip().lower() not in ("0", "false", "no", "off", "")
    return kind(raw)


def load_config(path: str | None = None, env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    data: dict = {}
    if path:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    cfg = Config()
    scalar = {f.name: f.type for f in fields(Config) if f.name != "routes"}
    for name, kind in scalar.items():
        if name in data:
            setattr(cfg, name, kind(data[name]))
        raw = env.get(f"OPTIMIZER_{name.upper()}")
        if raw is not None:
            setattr(cfg, name, _coerce(kind, raw))
    for key, rc in (data.get("routes") or {}).items():
        cfg.routes[key] = RouteConfig(enabled=bool(rc.get("enabled", True)),
                                      equivalence_bar=rc.get("equivalence_bar"))
    return cfg
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore optimizer/__init__.py optimizer/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: package scaffold and config loader"
```

---

### Task 2: SQLite schema and queries

**Files:**
- Create: `optimizer/db.py`, `tests/test_db.py`

**Interfaces:**
- Produces:
  - `connect(path: str) -> sqlite3.Connection` — creates all spec §10 tables, `row_factory = sqlite3.Row`.
  - `upsert_route(conn, key, provider, model, system_hash, name=None, now=None) -> None`
  - `get_route(conn, key) -> dict | None` — keys: `key provider model system_hash name status pinned_profile injection_form target_words exemplar eligible rejections first_seen last_seen last_sweep_id`
  - `set_pin(conn, key, profile, status="pinned") -> None`
  - `set_injection_form(conn, key, form) -> None`
  - `bump_rejection(conn, key) -> int` (returns new count)
  - `record_request(conn, *, ts, route_key, profile, input_tokens, output_tokens, cache_read, cache_create, estimated, stop_reason, latency_ms, body_ref=None) -> int`
  - `store_body(conn, request_json: str, response_json: str, expires_at: float) -> int`
  - `purge_expired(conn, now: float) -> int`
  - `route_stats(conn) -> list[dict]` — one row per (route_key, profile): `route_key profile n avg_output avg_input avg_cache_read max_tokens_stops`

- [ ] **Step 1: Write the failing tests**

`tests/test_db.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_db.py -v`
Expected: FAIL with `ImportError: cannot import name 'db'`

- [ ] **Step 3: Implement db.py**

`optimizer/db.py`:
```python
"""SQLite state: schema from spec §10 plus the few queries the proxy and report need."""
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS routes(
  key TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, system_hash TEXT NOT NULL,
  name TEXT, status TEXT NOT NULL DEFAULT 'observing', pinned_profile TEXT NOT NULL DEFAULT 'P0',
  injection_form TEXT NOT NULL DEFAULT 'system', target_words INTEGER NOT NULL DEFAULT 20,
  exemplar TEXT, eligible INTEGER, rejections INTEGER NOT NULL DEFAULT 0,
  first_seen REAL NOT NULL, last_seen REAL NOT NULL, last_sweep_id INTEGER);
CREATE TABLE IF NOT EXISTS requests(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, route_key TEXT NOT NULL, profile TEXT NOT NULL,
  input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER, cache_create INTEGER,
  estimated INTEGER NOT NULL DEFAULT 0, stop_reason TEXT, latency_ms INTEGER, body_ref INTEGER);
CREATE INDEX IF NOT EXISTS requests_route ON requests(route_key, profile);
CREATE TABLE IF NOT EXISTS bodies(
  id INTEGER PRIMARY KEY, request_json TEXT NOT NULL, response_json TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS samples(
  id INTEGER PRIMARY KEY, route_key TEXT NOT NULL, created_at REAL NOT NULL, item_ids_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sweeps(
  id INTEGER PRIMARY KEY, route_key TEXT NOT NULL, sample_id INTEGER NOT NULL, judge_model TEXT,
  judge_prompt_version TEXT, started_at REAL, finished_at REAL, cost_usd REAL, result_json TEXT, winner TEXT);
CREATE TABLE IF NOT EXISTS judgments(
  id INTEGER PRIMARY KEY, sweep_id INTEGER NOT NULL, item_id INTEGER NOT NULL, profile TEXT NOT NULL,
  trial INTEGER NOT NULL, label TEXT NOT NULL, order_ab TEXT);
CREATE TABLE IF NOT EXISTS shadow(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, route_key TEXT NOT NULL, request_id INTEGER, label TEXT NOT NULL);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_route(conn, key, provider, model, system_hash, name=None, now=None):
    now = time.time() if now is None else now
    conn.execute(
        "INSERT INTO routes(key, provider, model, system_hash, name, first_seen, last_seen) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET last_seen=excluded.last_seen",
        (key, provider, model, system_hash, name, now, now))
    conn.commit()


def get_route(conn, key):
    row = conn.execute("SELECT * FROM routes WHERE key=?", (key,)).fetchone()
    return dict(row) if row else None


def set_pin(conn, key, profile, status="pinned"):
    conn.execute("UPDATE routes SET pinned_profile=?, status=?, rejections=0 WHERE key=?", (profile, status, key))
    conn.commit()


def set_injection_form(conn, key, form):
    conn.execute("UPDATE routes SET injection_form=? WHERE key=?", (form, key))
    conn.commit()


def bump_rejection(conn, key) -> int:
    conn.execute("UPDATE routes SET rejections=rejections+1 WHERE key=?", (key,))
    conn.commit()
    return conn.execute("SELECT rejections FROM routes WHERE key=?", (key,)).fetchone()[0]


def record_request(conn, *, ts, route_key, profile, input_tokens, output_tokens, cache_read, cache_create,
                   estimated, stop_reason, latency_ms, body_ref=None) -> int:
    cur = conn.execute(
        "INSERT INTO requests(ts, route_key, profile, input_tokens, output_tokens, cache_read, cache_create, "
        "estimated, stop_reason, latency_ms, body_ref) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (ts, route_key, profile, input_tokens, output_tokens, cache_read, cache_create, int(bool(estimated)),
         stop_reason, latency_ms, body_ref))
    conn.commit()
    return cur.lastrowid


def store_body(conn, request_json: str, response_json: str, expires_at: float) -> int:
    cur = conn.execute("INSERT INTO bodies(request_json, response_json, expires_at) VALUES(?,?,?)",
                       (request_json, response_json, expires_at))
    conn.commit()
    return cur.lastrowid


def purge_expired(conn, now: float) -> int:
    cur = conn.execute("DELETE FROM bodies WHERE expires_at <= ?", (now,))
    conn.commit()
    return cur.rowcount


def route_stats(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT route_key, profile, COUNT(*) AS n, AVG(output_tokens) AS avg_output, AVG(input_tokens) AS avg_input, "
        "AVG(cache_read) AS avg_cache_read, "
        "SUM(CASE WHEN stop_reason IN ('max_tokens','length','max_output_tokens') THEN 1 ELSE 0 END) AS max_tokens_stops "
        "FROM requests GROUP BY route_key, profile").fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_db.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add optimizer/db.py tests/test_db.py
git commit -m "feat: sqlite schema and queries"
```

---

### Task 3: Route fingerprinting

**Files:**
- Create: `optimizer/fingerprint.py`, `tests/test_fingerprint.py`

**Interfaces:**
- Produces: `Fingerprint(key: str, model: str, system_hash: str)` NamedTuple; `fingerprint(provider: str, body: dict, override: str | None = None) -> Fingerprint`. Key format: `f"{provider}:{model}:{system_hash[:16]}"` or the override verbatim. `system_hash` = sha256 hex of `normalize(system_text) + "\x00" + tool_signature`.

- [ ] **Step 1: Write the failing tests**

`tests/test_fingerprint.py`:
```python
from optimizer.fingerprint import fingerprint, normalize, system_text, tool_signature


def test_normalize_collapses_whitespace_only():
    assert normalize("  a \n\n b\t c ") == "a b c"
    assert normalize("A b") != normalize("a b")


def test_anthropic_system_string_and_blocks_are_equal():
    s1 = {"model": "claude-opus-5-5", "system": "You are   terse.", "messages": []}
    s2 = {"model": "claude-opus-5-5", "system": [{"type": "text", "text": "You are"}, {"type": "text", "text": "terse."}],
          "messages": []}
    assert fingerprint("anthropic", s1) == fingerprint("anthropic", s2)
    assert fingerprint("anthropic", s1).key.startswith("anthropic:claude-opus-5-5:")


def test_openai_chat_uses_leading_system_and_developer_messages_only():
    body = {"model": "gpt-5", "messages": [{"role": "system", "content": "A"}, {"role": "developer", "content": "B"},
                                          {"role": "user", "content": "hi"}, {"role": "system", "content": "late"}]}
    assert system_text("openai", body) == "A\nB"


def test_openai_responses_uses_instructions_and_leading_developer_items():
    body = {"model": "gpt-5", "instructions": "I", "input": [{"role": "developer", "content": "D"},
                                                              {"role": "user", "content": "u"}]}
    assert system_text("openai", body) == "I\nD"
    assert system_text("openai", {"model": "gpt-5", "input": "plain string"}) == ""


def test_tools_sorted_by_name_and_schema_included():
    a = {"model": "m", "messages": [], "tools": [{"name": "b", "input_schema": {"x": 1}}, {"name": "a", "input_schema": {}}]}
    b = {"model": "m", "messages": [], "tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {"x": 1}}]}
    c = {"model": "m", "messages": [], "tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {"x": 2}}]}
    assert fingerprint("anthropic", a) == fingerprint("anthropic", b)
    assert fingerprint("anthropic", a) != fingerprint("anthropic", c)
    assert tool_signature("openai", {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}).startswith("f:")


def test_override_and_model_change_routes():
    body = {"model": "claude-opus-5-5", "system": "s", "messages": []}
    assert fingerprint("anthropic", body, override="billing").key == "billing"
    other = dict(body, model="claude-sonnet-5-5")
    assert fingerprint("anthropic", body).key != fingerprint("anthropic", other).key
    assert fingerprint("anthropic", body).system_hash == fingerprint("anthropic", other).system_hash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_fingerprint.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement fingerprint.py**

`optimizer/fingerprint.py`:
```python
"""Route key = provider + model + hash(normalized system prompt + sorted tool signatures). Spec §4."""
import hashlib
import json
from typing import NamedTuple


class Fingerprint(NamedTuple):
    key: str
    model: str
    system_hash: str


def normalize(text: str) -> str:
    return " ".join(text.split())


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type", "text") in ("text", "input_text"))
    return ""


def system_text(provider: str, body: dict) -> str:
    if provider == "anthropic":
        return _text_of(body.get("system", ""))
    parts = []
    if "messages" in body:  # chat completions
        for m in body.get("messages") or []:
            if m.get("role") not in ("system", "developer"):
                break
            parts.append(_text_of(m.get("content", "")))
    else:  # responses
        if body.get("instructions"):
            parts.append(str(body["instructions"]))
        inp = body.get("input")
        if isinstance(inp, list):
            for item in inp:
                if not isinstance(item, dict) or item.get("role") not in ("system", "developer"):
                    break
                parts.append(_text_of(item.get("content", "")))
    return "\n".join(parts)


def tool_signature(provider: str, body: dict) -> str:
    sigs = []
    for t in body.get("tools") or []:
        name = t.get("name") or (t.get("function") or {}).get("name") or t.get("type", "")
        sigs.append(f"{name}:{json.dumps(t, sort_keys=True, separators=(',', ':'))}")
    return "|".join(sorted(sigs))


def fingerprint(provider: str, body: dict, override: str | None = None) -> Fingerprint:
    model = str(body.get("model", ""))
    raw = normalize(system_text(provider, body)) + "\x00" + tool_signature(provider, body)
    system_hash = hashlib.sha256(raw.encode()).hexdigest()
    key = override if override else f"{provider}:{model}:{system_hash[:16]}"
    return Fingerprint(key, model, system_hash)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_fingerprint.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add optimizer/fingerprint.py tests/test_fingerprint.py
git commit -m "feat: route fingerprinting"
```

---

### Task 4: Provider detection

**Files:**
- Create: `optimizer/providers.py`, `tests/test_providers.py`

**Interfaces:**
- Produces: `detect_provider(path: str) -> str | None` (`"anthropic"`, `"openai"`, or `None`); `is_responses_api(path: str) -> bool`; `upstream(provider: str, config: Config) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_providers.py`:
```python
from optimizer.config import Config
from optimizer.providers import detect_provider, is_responses_api, upstream


def test_detects_by_path():
    assert detect_provider("/v1/messages") == "anthropic"
    assert detect_provider("/v1/chat/completions") == "openai"
    assert detect_provider("/v1/responses") == "openai"
    assert detect_provider("/v1/messages/count_tokens") is None
    assert detect_provider("/v1/models") is None
    assert is_responses_api("/v1/responses") and not is_responses_api("/v1/chat/completions")


def test_upstream_from_config():
    c = Config(anthropic_upstream="http://a", openai_upstream="http://o")
    assert upstream("anthropic", c) == "http://a" and upstream("openai", c) == "http://o"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement providers.py**

`optimizer/providers.py`:
```python
"""Which upstream a path belongs to. Unknown paths are forwarded without inspection (spec §7)."""
from optimizer.config import Config

_PATHS = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai", "/v1/responses": "openai"}


def detect_provider(path: str) -> str | None:
    return _PATHS.get(path.rstrip("/"))


def is_responses_api(path: str) -> bool:
    return path.rstrip("/") == "/v1/responses"


def upstream(provider: str, config: Config) -> str:
    return config.anthropic_upstream if provider == "anthropic" else config.openai_upstream
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add optimizer/providers.py tests/test_providers.py
git commit -m "feat: provider detection by path"
```

---

### Task 5: Profiles and cache-safe rewriting

**Files:**
- Create: `optimizer/rewrite.py`, `tests/test_rewrite.py`

**Interfaces:**
- Produces:
  - `PROFILES = ("P0", "P1", "P1b", "P2", "P3", "P4")`
  - `SHAPE_TEXT` constant (exact spec wording with `{n}` placeholder)
  - `RouteState(profile: str, injection_form: str = "system", target_words: int = 20, exemplar: str | None = None)` dataclass
  - `apply_profile(provider: str, body: dict, state: RouteState, responses_api: bool = False) -> dict` — returns a **new** dict; P0 or an inapplicable profile returns a deep copy equal to the input.
  - `is_system_role_rejection(status: int, text: str) -> bool`
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

`tests/test_rewrite.py`:
```python
import copy

from optimizer.rewrite import PROFILES, SHAPE_TEXT, RouteState, apply_profile, is_system_role_rejection

ANTH = {"model": "claude-opus-5-5", "max_tokens": 1024, "system": "S", "tools": [{"name": "t", "input_schema": {}}],
        "messages": [{"role": "user", "content": "q"}]}
CHAT = {"model": "gpt-5", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]}
RESP = {"model": "gpt-5", "instructions": "S", "input": "q"}
NEVER = ("system", "tools", "model", "thinking", "max_tokens", "max_completion_tokens", "max_output_tokens", "temperature")


def untouched(before, after):
    return all(before.get(k) == after.get(k) for k in NEVER)


def test_profiles_constant():
    assert PROFILES == ("P0", "P1", "P1b", "P2", "P3", "P4")
    assert SHAPE_TEXT == ("Answer directly. No preamble, restatement, or closing summary. "
                          "Target at most {n} words unless the task genuinely needs more.")


def test_p0_is_identity_and_does_not_mutate_input():
    body = copy.deepcopy(ANTH)
    out = apply_profile("anthropic", body, RouteState("P0"))
    assert out == ANTH and body == ANTH and out is not body


def test_anthropic_effort_down_from_model_default_and_explicit():
    out = apply_profile("anthropic", ANTH, RouteState("P1"))
    assert out["output_config"]["effort"] == "low"  # opus-5-5 default medium -> low
    body = dict(ANTH, model="claude-sonnet-5-5")
    assert apply_profile("anthropic", body, RouteState("P1"))["output_config"]["effort"] == "medium"  # default high
    body = dict(ANTH, output_config={"effort": "low"})
    assert apply_profile("anthropic", body, RouteState("P1")) == body  # already lowest: no-op
    body = dict(ANTH, model="claude-haiku-4-5")
    assert apply_profile("anthropic", body, RouteState("P1")) == body  # no effort on haiku: no-op
    assert untouched(ANTH, out)


def test_anthropic_shape_appends_mid_conversation_system_message():
    out = apply_profile("anthropic", ANTH, RouteState("P2", target_words=30))
    assert out["messages"][-1] == {"role": "system", "content": SHAPE_TEXT.format(n=30)}
    assert out["messages"][:-1] == ANTH["messages"] and untouched(ANTH, out)


def test_anthropic_shape_user_text_fallback_keeps_cache_control_block_first():
    body = dict(ANTH, messages=[{"role": "user", "content": [
        {"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}}]}])
    out = apply_profile("anthropic", body, RouteState("P2", injection_form="user_text", target_words=20))
    blocks = out["messages"][-1]["content"]
    assert blocks[0] == body["messages"][0]["content"][0]
    assert blocks[1] == {"type": "text", "text": SHAPE_TEXT.format(n=20)}
    out = apply_profile("anthropic", ANTH, RouteState("P2", injection_form="user_text", target_words=20))
    assert out["messages"][-1]["content"] == [{"type": "text", "text": "q"}, {"type": "text", "text": SHAPE_TEXT.format(n=20)}]


def test_anthropic_shape_falls_back_to_user_text_when_last_message_is_assistant():
    body = dict(ANTH, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    out = apply_profile("anthropic", body, RouteState("P2"))
    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][0]["content"][-1]["text"] == SHAPE_TEXT.format(n=20)


def test_p3_adds_exemplar_and_p4_combines():
    out = apply_profile("anthropic", ANTH, RouteState("P3", target_words=20, exemplar="Yes."))
    assert out["messages"][-1]["content"] == SHAPE_TEXT.format(n=20) + "\n\nExample of the expected length:\nYes."
    out = apply_profile("anthropic", ANTH, RouteState("P4", target_words=25))
    assert out["output_config"]["effort"] == "low" and out["messages"][-1]["role"] == "system"


def test_openai_chat_profiles():
    assert apply_profile("openai", CHAT, RouteState("P1"))["reasoning_effort"] == "low"
    body = dict(CHAT, reasoning_effort="high")
    assert apply_profile("openai", body, RouteState("P1"))["reasoning_effort"] == "medium"
    assert apply_profile("openai", CHAT, RouteState("P1b"))["verbosity"] == "low"
    out = apply_profile("openai", CHAT, RouteState("P2", target_words=40))
    assert out["messages"][-1] == {"role": "developer", "content": SHAPE_TEXT.format(n=40)}


def test_openai_responses_profiles():
    assert apply_profile("openai", RESP, RouteState("P1"), responses_api=True)["reasoning"] == {"effort": "low"}
    assert apply_profile("openai", RESP, RouteState("P1b"), responses_api=True)["text"] == {"verbosity": "low"}
    out = apply_profile("openai", RESP, RouteState("P2", target_words=20), responses_api=True)
    assert out["input"] == [{"role": "user", "content": "q"}, {"role": "developer", "content": SHAPE_TEXT.format(n=20)}]
    body = dict(RESP, input=[{"role": "user", "content": "q"}])
    out = apply_profile("openai", body, RouteState("P2", target_words=20), responses_api=True)
    assert out["input"][-1]["role"] == "developer" and len(out["input"]) == 2


def test_rejection_detector():
    assert is_system_role_rejection(400, '{"error":{"message":"role \'system\' is not supported on this model"}}')
    assert not is_system_role_rejection(400, "other")
    assert not is_system_role_rejection(500, "role 'system' is not supported")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rewrite.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement rewrite.py**

`optimizer/rewrite.py`:
```python
"""Output profiles (spec §5) and cache-safe request rewriting (spec §7).

Never touches: system, tools, model, thinking, max_tokens/max_completion_tokens/max_output_tokens, temperature.
"""
import copy
from dataclasses import dataclass

PROFILES = ("P0", "P1", "P1b", "P2", "P3", "P4")
SHAPE_TEXT = ("Answer directly. No preamble, restatement, or closing summary. "
              "Target at most {n} words unless the task genuinely needs more.")
EXEMPLAR_PREFIX = "\n\nExample of the expected length:\n"

EFFORT_LADDER = ("low", "medium", "high", "xhigh", "max")
# Models whose default effort is not "high" (Anthropic pricing docs, 2026-09). Everything else defaults to "high".
ANTHROPIC_DEFAULT_EFFORT = {"claude-opus-5-5": "medium"}
# Models with no effort parameter: P1 is a no-op on these.
ANTHROPIC_NO_EFFORT = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-haiku-3", "claude-3-")


@dataclass
class RouteState:
    profile: str
    injection_form: str = "system"  # "system" | "user_text"
    target_words: int = 20
    exemplar: str | None = None


def _shape(state: RouteState) -> str:
    text = SHAPE_TEXT.format(n=max(20, int(state.target_words)))
    if state.profile == "P3" and state.exemplar:
        text += EXEMPLAR_PREFIX + state.exemplar
    return text


def _step_down(current: str) -> str:
    i = EFFORT_LADDER.index(current) if current in EFFORT_LADDER else EFFORT_LADDER.index("medium")
    return EFFORT_LADDER[max(0, i - 1)]


def _anthropic_effort(body: dict) -> None:
    model = body.get("model", "")
    if any(model.startswith(p) for p in ANTHROPIC_NO_EFFORT):
        return
    oc = body.get("output_config") or {}
    current = oc.get("effort") or ANTHROPIC_DEFAULT_EFFORT.get(model, "high")
    new = _step_down(current)
    if new != current:
        body["output_config"] = dict(oc, effort=new)


def _append_user_text(msg: dict, text: str) -> None:
    content = msg.get("content", "")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    msg["content"] = list(content) + [{"type": "text", "text": text}]


def _anthropic_shape(body: dict, state: RouteState) -> None:
    msgs = body.setdefault("messages", [])
    text = _shape(state)
    if state.injection_form == "system" and msgs and msgs[-1].get("role") == "user":
        msgs.append({"role": "system", "content": text})
        return
    for m in reversed(msgs):
        if m.get("role") == "user":
            _append_user_text(m, text)
            return


def _openai_shape(body: dict, state: RouteState, responses_api: bool) -> None:
    text = _shape(state)
    if responses_api:
        inp = body.get("input", "")
        if isinstance(inp, str):
            inp = [{"role": "user", "content": inp}]
        body["input"] = list(inp) + [{"role": "developer", "content": text}]
    else:
        body["messages"] = list(body.get("messages") or []) + [{"role": "developer", "content": text}]


def apply_profile(provider: str, body: dict, state: RouteState, responses_api: bool = False) -> dict:
    out = copy.deepcopy(body)
    p = state.profile
    if p == "P0" or p not in PROFILES:
        return out
    if provider == "anthropic":
        if p in ("P1", "P4"):
            _anthropic_effort(out)
        if p in ("P2", "P3", "P4"):
            _anthropic_shape(out, state)
        return out
    if p in ("P1", "P4"):
        if responses_api:
            r = out.get("reasoning") or {}
            out["reasoning"] = dict(r, effort=_step_down(r.get("effort", "medium")))
        else:
            out["reasoning_effort"] = _step_down(out.get("reasoning_effort", "medium"))
    if p == "P1b":
        if responses_api:
            out["text"] = dict(out.get("text") or {}, verbosity="low")
        else:
            out["verbosity"] = "low"
    if p in ("P2", "P3", "P4"):
        _openai_shape(out, state, responses_api)
    return out


def is_system_role_rejection(status: int, text: str) -> bool:
    return status == 400 and "role 'system' is not supported" in text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_rewrite.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add optimizer/rewrite.py tests/test_rewrite.py
git commit -m "feat: output profiles and cache-safe rewriting"
```

---

### Task 6: Usage extraction (non-streaming and SSE)

**Files:**
- Create: `optimizer/usage.py`, `tests/test_usage.py`

**Interfaces:**
- Produces:
  - `Usage` dataclass: `input_tokens: int | None, output_tokens: int | None, cache_read: int | None, cache_create: int | None, stop_reason: str | None, estimated: bool = False`
  - `usage_from_body(provider: str, body: dict) -> Usage`
  - `class StreamUsage(provider: str)` with `.feed(chunk: bytes) -> None` and `.result() -> Usage`; for OpenAI streams with no usage chunk, `output_tokens` is estimated from accumulated text and `estimated=True`.
  - `estimate_tokens(text: str) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_usage.py`:
```python
from optimizer.usage import StreamUsage, Usage, estimate_tokens, usage_from_body


def test_anthropic_body():
    u = usage_from_body("anthropic", {"stop_reason": "end_turn", "usage": {
        "input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 1}})
    assert u == Usage(10, 20, 5, 1, "end_turn", False)


def test_openai_chat_and_responses_bodies():
    u = usage_from_body("openai", {"choices": [{"finish_reason": "length"}], "usage": {
        "prompt_tokens": 7, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 2}}})
    assert u == Usage(7, 3, 2, None, "length", False)
    u = usage_from_body("openai", {"object": "response", "status": "incomplete",
                                   "incomplete_details": {"reason": "max_output_tokens"},
                                   "usage": {"input_tokens": 4, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 0}}})
    assert u == Usage(4, 9, 0, None, "max_output_tokens", False)
    assert usage_from_body("openai", {"object": "response", "status": "completed", "usage": {}}).stop_reason == "completed"


def test_missing_usage_is_none_not_error():
    assert usage_from_body("anthropic", {}) == Usage(None, None, None, None, None, False)


def test_anthropic_stream():
    s = StreamUsage("anthropic")
    s.feed(b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":11,"cache_read_input_tokens":3,"cache_creation_input_tokens":0}}}\n\n')
    s.feed(b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n')
    s.feed(b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}\n\n')
    assert s.result() == Usage(11, 6, 3, 0, "end_turn", False)


def test_openai_chat_stream_with_and_without_usage():
    s = StreamUsage("openai")
    s.feed(b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n')
    s.feed(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\ndata: [DONE]\n\n')
    assert s.result() == Usage(5, 2, None, None, "stop", False)
    s = StreamUsage("openai")
    s.feed(b'data: {"choices":[{"delta":{"content":"hello world"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    u = s.result()
    assert u.estimated and u.output_tokens == estimate_tokens("hello world") and u.stop_reason == "stop"


def test_openai_responses_stream():
    s = StreamUsage("openai")
    s.feed(b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":2}}}\n\n')
    assert s.result() == Usage(1, 2, None, None, "completed", False)


def test_split_chunks_are_reassembled():
    s = StreamUsage("anthropic")
    line = b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}\n\n'
    s.feed(line[:20]); s.feed(line[20:])
    assert s.result().output_tokens == 6


def test_estimate_is_positive():
    assert estimate_tokens("The quick brown fox") >= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_usage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement usage.py**

`optimizer/usage.py`:
```python
"""Usage + stop reason from provider responses, non-streaming and SSE (spec §8)."""
import json
from dataclasses import dataclass

_enc = None


def estimate_tokens(text: str) -> int:
    global _enc
    try:
        if _enc is None:
            import tiktoken
            _enc = tiktoken.get_encoding("o200k_base")
        return len(_enc.encode(text))
    except Exception:  # tiktoken missing or encoding download blocked
        return max(1, len(text) // 4)


@dataclass
class Usage:
    input_tokens: int | None
    output_tokens: int | None
    cache_read: int | None
    cache_create: int | None
    stop_reason: str | None
    estimated: bool = False


def usage_from_body(provider: str, body: dict) -> Usage:
    u = body.get("usage") or {}
    if provider == "anthropic":
        return Usage(u.get("input_tokens"), u.get("output_tokens"), u.get("cache_read_input_tokens"),
                     u.get("cache_creation_input_tokens"), body.get("stop_reason"))
    if body.get("object") == "response" or "input_tokens" in u:
        stop = (body.get("incomplete_details") or {}).get("reason") or body.get("status")
        return Usage(u.get("input_tokens"), u.get("output_tokens"),
                     (u.get("input_tokens_details") or {}).get("cached_tokens"), None, stop)
    choices = body.get("choices") or [{}]
    return Usage(u.get("prompt_tokens"), u.get("completion_tokens"),
                 (u.get("prompt_tokens_details") or {}).get("cached_tokens"), None, choices[0].get("finish_reason"))


class StreamUsage:
    def __init__(self, provider: str):
        self.provider = provider
        self._buf = b""
        self._u = Usage(None, None, None, None, None)
        self._text = []

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                self._event(json.loads(payload))
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

    def _event(self, ev: dict) -> None:
        u = self._u
        if self.provider == "anthropic":
            if ev.get("type") == "message_start":
                mu = (ev.get("message") or {}).get("usage") or {}
                u.input_tokens, u.cache_read, u.cache_create = mu.get("input_tokens"), mu.get("cache_read_input_tokens"), mu.get("cache_creation_input_tokens")
            elif ev.get("type") == "message_delta":
                u.output_tokens = (ev.get("usage") or {}).get("output_tokens", u.output_tokens)
                u.stop_reason = (ev.get("delta") or {}).get("stop_reason", u.stop_reason)
            return
        if ev.get("type") == "response.completed":
            self._u = usage_from_body("openai", ev.get("response") or {})
            return
        for c in ev.get("choices") or []:
            txt = (c.get("delta") or {}).get("content")
            if txt:
                self._text.append(txt)
            if c.get("finish_reason"):
                u.stop_reason = c["finish_reason"]
        if ev.get("usage"):
            got = usage_from_body("openai", {"usage": ev["usage"], "choices": [{"finish_reason": u.stop_reason}]})
            got.stop_reason = u.stop_reason
            self._u = got

    def result(self) -> Usage:
        u = self._u
        if self.provider == "openai" and u.output_tokens is None and self._text:
            u.output_tokens, u.estimated = estimate_tokens("".join(self._text)), True
        return u
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_usage.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add optimizer/usage.py tests/test_usage.py
git commit -m "feat: usage extraction for bodies and SSE streams"
```

---

### Task 7: Proxy — non-streaming passthrough, fail-open, kill switches, recording

**Files:**
- Create: `optimizer/proxy.py`, `tests/test_proxy.py`

**Interfaces:**
- Produces: `create_app(config: Config, conn: sqlite3.Connection, client: httpx.AsyncClient | None = None) -> FastAPI`. Internal helpers used by Tasks 8–9: `_decide(config, conn, path, headers, raw) -> Decision` where `Decision(provider, fp, route, profile, body_bytes, record: bool)`; `_forward(client, method, url, headers, content, stream: bool) -> httpx.Response`; `HOP_HEADERS` set.
- Consumes: `load_config/Config` (T1), `db.*` (T2), `fingerprint` (T3), `detect_provider/is_responses_api/upstream` (T4), `apply_profile/RouteState` (T5), `usage_from_body` (T6).

- [ ] **Step 1: Write the failing tests**

`tests/test_proxy.py`:
```python
import json

import httpx
import pytest

from optimizer import db
from optimizer.config import Config, RouteConfig
from optimizer.proxy import create_app

ANTH_REQ = {"model": "claude-opus-5-5", "max_tokens": 50, "system": "S", "messages": [{"role": "user", "content": "q"}]}
ANTH_RESP = {"id": "m1", "type": "message", "stop_reason": "end_turn",
             "content": [{"type": "text", "text": "A"}],
             "usage": {"input_tokens": 9, "output_tokens": 4, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}


def make(config=None, handler=None, seen=None):
    seen = [] if seen is None else seen

    def default_handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, json=ANTH_RESP, headers={"x-upstream": "1"})

    conn = db.connect(":memory:")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler or default_handler))
    app = create_app(config or Config(sample_rate=1.0), conn, client=client)
    return app, conn, seen


async def post(app, path, body, headers=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, content=json.dumps(body).encode(), headers={"content-type": "application/json",
                                                                               "x-api-key": "sk-secret", **(headers or {})})


@pytest.mark.anyio
async def test_passthrough_is_byte_identical_and_forwards_auth_only_upstream():
    app, conn, seen = make()
    r = await post(app, "/v1/messages", ANTH_REQ)
    assert r.status_code == 200 and r.json() == ANTH_RESP and r.headers["x-upstream"] == "1"
    assert json.loads(seen[0].content) == ANTH_REQ
    assert seen[0].headers["x-api-key"] == "sk-secret"
    assert str(seen[0].url) == "https://api.anthropic.com/v1/messages"


@pytest.mark.anyio
async def test_records_route_and_usage_and_samples_body():
    app, conn, seen = make()
    await post(app, "/v1/messages", ANTH_REQ)
    route = conn.execute("SELECT * FROM routes").fetchone()
    assert route["provider"] == "anthropic" and route["model"] == "claude-opus-5-5" and route["status"] == "observing"
    req = conn.execute("SELECT * FROM requests").fetchone()
    assert (req["profile"], req["input_tokens"], req["output_tokens"], req["stop_reason"]) == ("P0", 9, 4, "end_turn")
    body = conn.execute("SELECT * FROM bodies").fetchone()
    assert "sk-secret" not in body["request_json"] and json.loads(body["request_json"]) == ANTH_REQ


@pytest.mark.anyio
async def test_route_header_override_names_route():
    app, conn, _ = make()
    await post(app, "/v1/messages", ANTH_REQ, {"X-Optimizer-Route": "billing"})
    assert conn.execute("SELECT key FROM routes").fetchone()["key"] == "billing"


@pytest.mark.anyio
async def test_unknown_path_and_malformed_json_fail_open():
    app, conn, seen = make()
    r = await post(app, "/v1/models", {"x": 1})
    assert r.status_code == 200 and json.loads(seen[0].content) == {"x": 1}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/messages", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 200 and seen[1].content == b"{not json"
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


@pytest.mark.anyio
async def test_upstream_error_status_is_passed_through():
    def h(req):
        return httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "7"})
    app, conn, _ = make(handler=h)
    r = await post(app, "/v1/messages", ANTH_REQ)
    assert r.status_code == 429 and r.headers["retry-after"] == "7" and r.json() == {"error": "slow down"}


@pytest.mark.anyio
async def test_kill_switches_force_p0_and_bypass_skips_recording():
    app, conn, seen = make()
    await post(app, "/v1/messages", ANTH_REQ)
    key = conn.execute("SELECT key FROM routes").fetchone()["key"]
    db.set_pin(conn, key, "P2")
    await post(app, "/v1/messages", ANTH_REQ, {"X-Optimizer": "off"})
    assert json.loads(seen[-1].content) == ANTH_REQ
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P0"
    n = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    await post(app, "/v1/messages", ANTH_REQ, {"X-Optimizer": "bypass"})
    assert json.loads(seen[-1].content) == ANTH_REQ
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == n
    app2, conn2, seen2 = make(Config(enabled=False, sample_rate=0))
    await post(app2, "/v1/messages", ANTH_REQ)
    key2 = conn2.execute("SELECT key FROM routes").fetchone()["key"]
    db.set_pin(conn2, key2, "P2")
    await post(app2, "/v1/messages", ANTH_REQ)
    assert json.loads(seen2[-1].content) == ANTH_REQ
    app3, conn3, seen3 = make(Config(sample_rate=0, routes={"billing": RouteConfig(enabled=False)}))
    await post(app3, "/v1/messages", ANTH_REQ, {"X-Optimizer-Route": "billing"})
    db.set_pin(conn3, "billing", "P2")
    await post(app3, "/v1/messages", ANTH_REQ, {"X-Optimizer-Route": "billing"})
    assert json.loads(seen3[-1].content) == ANTH_REQ


@pytest.mark.anyio
async def test_health():
    app, _, _ = make()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/optimizer/health")).json() == {"ok": True}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_proxy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'optimizer.proxy'`

- [ ] **Step 3: Implement proxy.py (non-streaming)**

`optimizer/proxy.py`:
```python
"""The data plane: forward, record, apply pinned profile, fail open. Spec §3, §7, §8."""
import json
import logging
import random
import time
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, Response

from optimizer import db
from optimizer.config import Config
from optimizer.fingerprint import Fingerprint, fingerprint
from optimizer.providers import detect_provider, is_responses_api, upstream
from optimizer.rewrite import RouteState, apply_profile
from optimizer.usage import usage_from_body

log = logging.getLogger("optimizer")
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}
RESP_DROP = {"content-length", "content-encoding", "transfer-encoding", "connection"}


@dataclass
class Decision:
    provider: str | None
    fp: Fingerprint | None
    route: dict | None
    profile: str
    body_bytes: bytes
    record: bool


def _decide(config: Config, conn, path: str, headers, raw: bytes) -> Decision:
    """Everything before forwarding. Any exception here is caught by the caller -> fail open."""
    provider = detect_provider(path)
    if provider is None:
        return Decision(None, None, None, "P0", raw, False)
    mode = (headers.get("x-optimizer") or "").lower()
    if mode == "bypass":
        return Decision(provider, None, None, "P0", raw, False)
    body = json.loads(raw)
    fp = fingerprint(provider, body, headers.get("x-optimizer-route"))
    db.upsert_route(conn, fp.key, provider, fp.model, fp.system_hash, headers.get("x-optimizer-route"))
    route = db.get_route(conn, fp.key)
    route_cfg = config.routes.get(fp.key)
    enabled = config.enabled and mode != "off" and (route_cfg is None or route_cfg.enabled)
    profile = route["pinned_profile"] if enabled else "P0"
    if profile == "P0":
        return Decision(provider, fp, route, "P0", raw, True)
    state = RouteState(profile, route["injection_form"], route["target_words"], route["exemplar"])
    new_body = apply_profile(provider, body, state, responses_api=is_responses_api(path))
    return Decision(provider, fp, route, profile, json.dumps(new_body).encode(), True)


def _upstream_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS}


async def _forward(client: httpx.AsyncClient, method: str, url: str, headers: dict, content: bytes, stream: bool):
    req = client.build_request(method, url, headers=headers, content=content)
    return await client.send(req, stream=stream)


def _record(config: Config, conn, d: Decision, usage, latency_ms: int, request_raw: bytes, response_text: str,
            profile: str):
    body_ref = None
    if random.random() < config.sample_rate:
        body_ref = db.store_body(conn, request_raw.decode("utf-8", "replace"), response_text,
                                 time.time() + config.retention_days * 86400)
    db.record_request(conn, ts=time.time(), route_key=d.fp.key, profile=profile, input_tokens=usage.input_tokens,
                      output_tokens=usage.output_tokens, cache_read=usage.cache_read, cache_create=usage.cache_create,
                      estimated=usage.estimated, stop_reason=usage.stop_reason, latency_ms=latency_ms, body_ref=body_ref)


def create_app(config: Config, conn, client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI()
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    @app.get("/optimizer/health")
    async def health():
        return {"ok": True}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request):
        raw = await request.body()
        path = "/" + path
        try:
            d = _decide(config, conn, path, request.headers, raw) if request.method == "POST" else \
                Decision(detect_provider(path), None, None, "P0", raw, False)
        except Exception:  # fail open: forward the customer's original bytes
            log.exception("decide failed; forwarding original request")
            d = Decision(detect_provider(path), None, None, "P0", raw, False)
        # Unknown paths still need an upstream: Anthropic for /v1/messages/*, OpenAI otherwise.
        base = upstream(d.provider, config) if d.provider else (
            config.anthropic_upstream if path.startswith("/v1/messages") else config.openai_upstream)
        url = base + path + (f"?{request.url.query}" if request.url.query else "")
        headers = _upstream_headers(request.headers)
        t0 = time.monotonic()
        resp = await _forward(client, request.method, url, headers, d.body_bytes, stream=False)
        latency = int((time.monotonic() - t0) * 1000)
        content = await resp.aread()
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESP_DROP}
        if d.record and resp.status_code < 400:
            try:
                usage = usage_from_body(d.provider, json.loads(content))
                _record(config, conn, d, usage, latency, raw, content.decode("utf-8", "replace"), d.profile)
            except Exception:
                log.exception("record failed")
        return Response(content=content, status_code=resp.status_code, headers=out_headers)

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_proxy.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add optimizer/proxy.py tests/test_proxy.py
git commit -m "feat: proxy passthrough with fail-open, kill switches and recording"
```

---

### Task 8: Proxy — streaming passthrough

**Files:**
- Modify: `optimizer/proxy.py` (the `proxy` handler), `tests/test_proxy.py` (append)

**Interfaces:**
- Consumes: `StreamUsage` (T6).
- Produces: streaming responses are relayed chunk-by-chunk with the upstream status and headers; usage is recorded after the stream ends.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_proxy.py`)

```python
SSE = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":11,"cache_read_input_tokens":3,"cache_creation_input_tokens":0}}}\n\n'
       b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
       b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}\n\n')


@pytest.mark.anyio
async def test_streaming_passthrough_relays_chunks_and_records_after_end():
    async def gen():
        for i in range(0, len(SSE), 37):
            yield SSE[i:i + 37]

    def h(req):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())
    app, conn, _ = make(handler=h)
    body = dict(ANTH_REQ, stream=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/messages", content=json.dumps(body).encode(),
                            headers={"content-type": "application/json"}) as r:
            assert r.status_code == 200 and r.headers["content-type"] == "text/event-stream"
            got = b"".join([chunk async for chunk in r.aiter_raw()])
    assert got == SSE
    req = conn.execute("SELECT * FROM requests").fetchone()
    assert (req["input_tokens"], req["output_tokens"], req["cache_read"], req["stop_reason"]) == (11, 6, 3, "end_turn")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_proxy.py::test_streaming_passthrough_relays_chunks_and_records_after_end -v`
Expected: FAIL — the non-streaming handler buffers (`got == SSE` may pass) but `requests` has no row because `usage_from_body` cannot parse SSE → `req is None` → `TypeError`.

- [ ] **Step 3: Add streaming to the handler**

In `optimizer/proxy.py`, add the import and replace the body of `proxy` from `t0 = time.monotonic()` onward:

```python
from fastapi.responses import StreamingResponse
from optimizer.usage import StreamUsage, usage_from_body
```

```python
        t0 = time.monotonic()
        resp = await _forward(client, request.method, url, headers, d.body_bytes, stream=True)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESP_DROP}
        if "text/event-stream" in resp.headers.get("content-type", ""):
            su = StreamUsage(d.provider or "openai")
            collected = []

            async def relay():
                try:
                    async for chunk in resp.aiter_raw():
                        su.feed(chunk)
                        if d.record:
                            collected.append(chunk)
                        yield chunk
                finally:
                    await resp.aclose()
                    if d.record and resp.status_code < 400:
                        try:
                            _record(config, conn, d, su.result(), int((time.monotonic() - t0) * 1000), raw,
                                    b"".join(collected).decode("utf-8", "replace"), d.profile)
                        except Exception:
                            log.exception("record failed")

            return StreamingResponse(relay(), status_code=resp.status_code, headers=out_headers)
        content = await resp.aread()
        await resp.aclose()
        latency = int((time.monotonic() - t0) * 1000)
        if d.record and resp.status_code < 400:
            try:
                usage = usage_from_body(d.provider, json.loads(content))
                _record(config, conn, d, usage, latency, raw, content.decode("utf-8", "replace"), d.profile)
            except Exception:
                log.exception("record failed")
        return Response(content=content, status_code=resp.status_code, headers=out_headers)
```

- [ ] **Step 4: Run all proxy tests to verify they pass**

Run: `.venv/bin/pytest tests/test_proxy.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add optimizer/proxy.py tests/test_proxy.py
git commit -m "feat: streaming passthrough with post-stream usage recording"
```

---

### Task 9: Proxy — apply pinned profile, retry-with-original on 4xx, auto-unpin

**Files:**
- Modify: `optimizer/proxy.py`, `tests/test_proxy.py` (append)

**Interfaces:**
- Consumes: `is_system_role_rejection` (T5), `db.bump_rejection / set_injection_form / set_pin` (T2).
- Produces: behaviour per spec §7–§8: rewritten request → provider 4xx → one retry with the original bytes; a `role 'system'` rejection switches the route to `user_text` injection without counting; other rejections count, and the third sets `pinned_profile=P0, status=reverted`. The retried request is recorded as `P0`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_proxy.py`)

```python
from optimizer.rewrite import SHAPE_TEXT


@pytest.mark.anyio
async def test_pinned_profile_rewrites_request():
    app, conn, seen = make(Config(sample_rate=0))
    await post(app, "/v1/messages", ANTH_REQ)
    key = conn.execute("SELECT key FROM routes").fetchone()["key"]
    db.set_pin(conn, key, "P4")
    await post(app, "/v1/messages", ANTH_REQ)
    sent = json.loads(seen[-1].content)
    assert sent["output_config"] == {"effort": "low"}
    assert sent["messages"][-1] == {"role": "system", "content": SHAPE_TEXT.format(n=20)}
    assert sent["system"] == "S" and sent["max_tokens"] == 50
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P4"


@pytest.mark.anyio
async def test_system_role_rejection_switches_to_user_text_and_retries():
    calls = []

    def h(req):
        calls.append(json.loads(req.content))
        if any(m.get("role") == "system" for m in calls[-1]["messages"]):
            return httpx.Response(400, json={"error": {"message": "role 'system' is not supported on this model"}})
        return httpx.Response(200, json=ANTH_RESP)
    app, conn, _ = make(Config(sample_rate=0), handler=h)
    await post(app, "/v1/messages", ANTH_REQ)
    key = conn.execute("SELECT key FROM routes").fetchone()["key"]
    db.set_pin(conn, key, "P2")
    r = await post(app, "/v1/messages", ANTH_REQ)
    assert r.status_code == 200 and calls[-1] == ANTH_REQ  # retried with original bytes
    route = db.get_route(conn, key)
    assert route["injection_form"] == "user_text" and route["rejections"] == 0 and route["pinned_profile"] == "P2"
    await post(app, "/v1/messages", ANTH_REQ)
    assert calls[-1]["messages"][-1]["content"][-1]["text"] == SHAPE_TEXT.format(n=20)
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"


@pytest.mark.anyio
async def test_three_other_rejections_unpin_to_p0_reverted():
    def h(req):
        body = json.loads(req.content)
        if "output_config" in body:
            return httpx.Response(400, json={"error": {"message": "output_config.effort: unsupported"}})
        return httpx.Response(200, json=ANTH_RESP)
    app, conn, _ = make(Config(sample_rate=0), handler=h)
    await post(app, "/v1/messages", ANTH_REQ)
    key = conn.execute("SELECT key FROM routes").fetchone()["key"]
    db.set_pin(conn, key, "P1")
    for i in range(3):
        r = await post(app, "/v1/messages", ANTH_REQ)
        assert r.status_code == 200
        assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P0"
    route = db.get_route(conn, key)
    assert route["pinned_profile"] == "P0" and route["status"] == "reverted"


@pytest.mark.anyio
async def test_4xx_on_unrewritten_request_is_not_retried():
    n = []

    def h(req):
        n.append(1)
        return httpx.Response(400, json={"error": "bad"})
    app, conn, _ = make(Config(sample_rate=0), handler=h)
    r = await post(app, "/v1/messages", ANTH_REQ)
    assert r.status_code == 400 and len(n) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_proxy.py -v -k "pinned or rejection or unpin or not_retried"`
Expected: `test_pinned_profile_rewrites_request` PASS (rewrite already wired), the two rejection tests FAIL (retry not implemented; 400 returned), `not_retried` PASS.

- [ ] **Step 3: Add the retry path**

In `optimizer/proxy.py`, import `is_system_role_rejection` from `optimizer.rewrite`. Then, in the `proxy` handler, **replace** the single line `out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESP_DROP}` that follows the first `_forward(...)` call with this block (it ends with that same line):

```python
        profile_used = d.profile
        if d.profile != "P0" and 400 <= resp.status_code < 500:
            err_text = (await resp.aread()).decode("utf-8", "replace")
            await resp.aclose()
            try:
                if is_system_role_rejection(resp.status_code, err_text):
                    db.set_injection_form(conn, d.fp.key, "user_text")
                elif db.bump_rejection(conn, d.fp.key) >= 3:
                    db.set_pin(conn, d.fp.key, "P0", status="reverted")
                    log.warning("route %s reverted to P0 after 3 provider rejections", d.fp.key)
            except Exception:
                log.exception("rejection bookkeeping failed")
            log.info("provider rejected profile %s on %s (%s); retrying original", d.profile, d.fp.key, resp.status_code)
            t0 = time.monotonic()
            resp = await _forward(client, request.method, url, headers, raw, stream=True)
            profile_used = "P0"
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESP_DROP}
```

and replace both `d.profile` arguments to `_record(...)` with `profile_used`.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all passed (config 4, db 3, fingerprint 6, providers 2, rewrite 10, usage 8, proxy 12)

- [ ] **Step 5: Commit**

```bash
git add optimizer/proxy.py tests/test_proxy.py
git commit -m "feat: retry rewritten requests with original on 4xx; auto-unpin after 3 rejections"
```

---

### Task 10: Report endpoints

**Files:**
- Create: `optimizer/report.py`, `tests/test_report.py`
- Modify: `optimizer/proxy.py` (two routes)

**Interfaces:**
- Produces: `route_rows(conn) -> list[dict]` with keys `key provider model name status pinned_profile baseline_n baseline_avg_output pinned_n pinned_avg_output baseline_usd_per_1k pinned_usd_per_1k estimated_savings_pct last_seen`; `render_html(rows: list[dict]) -> str`; `PRICES: dict[str, tuple[float, float]]` ($/M input, $/M output). Endpoints `GET /optimizer/report` (JSON list) and `GET /optimizer/report.html`.
- Consumes: `db.route_stats`, `db.get_route` (T2).

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
import httpx
import pytest

from optimizer import db
from optimizer.config import Config
from optimizer.proxy import create_app
from optimizer.report import PRICES, render_html, route_rows


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'optimizer.report'`

- [ ] **Step 3: Implement report.py and wire the routes**

`optimizer/report.py`:
```python
"""Per-route report: baseline vs pinned profile, $/1k requests, savings. Spec §3."""
import html

from optimizer import db

# $/M tokens (input, output). Claude from the Anthropic pricing docs (2026-09); OpenAI entries added as customers need them.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0), "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _usd_per_1k(model: str, avg_in, avg_out):
    p = PRICES.get(model)
    if not p or avg_in is None or avg_out is None:
        return None
    return (avg_in * p[0] + avg_out * p[1]) / 1e6 * 1000


def route_rows(conn) -> list[dict]:
    stats = {}
    for s in db.route_stats(conn):
        stats.setdefault(s["route_key"], {})[s["profile"]] = s
    rows = []
    for key in [r["key"] for r in conn.execute("SELECT key FROM routes ORDER BY last_seen DESC")]:
        route = db.get_route(conn, key)
        base = stats.get(key, {}).get("P0", {})
        pinned = stats.get(key, {}).get(route["pinned_profile"], {}) if route["pinned_profile"] != "P0" else {}
        b_usd = _usd_per_1k(route["model"], base.get("avg_input"), base.get("avg_output"))
        p_usd = _usd_per_1k(route["model"], pinned.get("avg_input"), pinned.get("avg_output"))
        rows.append({
            "key": key, "provider": route["provider"], "model": route["model"], "name": route["name"],
            "status": route["status"], "pinned_profile": route["pinned_profile"],
            "baseline_n": base.get("n", 0), "baseline_avg_output": base.get("avg_output"),
            "pinned_n": pinned.get("n", 0), "pinned_avg_output": pinned.get("avg_output"),
            "baseline_usd_per_1k": b_usd, "pinned_usd_per_1k": p_usd,
            "estimated_savings_pct": (100 * (1 - p_usd / b_usd)) if b_usd and p_usd else None,
            "last_seen": route["last_seen"],
        })
    return rows


COLS = ("key", "name", "model", "status", "pinned_profile", "baseline_n", "baseline_avg_output", "pinned_n",
        "pinned_avg_output", "baseline_usd_per_1k", "pinned_usd_per_1k", "estimated_savings_pct")


def render_html(rows: list[dict]) -> str:
    def cell(v):
        return "" if v is None else (f"{v:.2f}" if isinstance(v, float) else html.escape(str(v)))
    head = "".join(f"<th>{c}</th>" for c in COLS)
    body = "".join("<tr>" + "".join(f"<td>{cell(r[c])}</td>" for c in COLS) + "</tr>" for r in rows)
    return (f"<!doctype html><title>Output optimizer report</title>"
            f"<style>body{{font-family:system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}</style>"
            f"<h1>Routes</h1><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
```

In `optimizer/proxy.py`, add `from fastapi.responses import HTMLResponse, StreamingResponse` and `from optimizer.report import render_html, route_rows`, then add inside `create_app` **before** the catch-all route:

```python
    @app.get("/optimizer/report")
    async def report():
        return route_rows(conn)

    @app.get("/optimizer/report.html")
    async def report_html():
        return HTMLResponse(render_html(route_rows(conn)))
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all passed (previous 45 + report 4)

- [ ] **Step 5: Commit**

```bash
git add optimizer/report.py optimizer/proxy.py tests/test_report.py
git commit -m "feat: per-route JSON and HTML report"
```

---

### Task 11: Entrypoint, retention purge, README, live cache-safety test

**Files:**
- Create: `optimizer/__main__.py`, `tests/live/__init__.py`, `tests/live/test_cache_safety.py`, `README.md`, `optimizer.example.toml`
- Modify: `optimizer/proxy.py` (purge expired bodies on startup and every 1000 requests)

**Interfaces:**
- Produces: `python -m optimizer [--config optimizer.toml]` serves `Config.listen`. `tests/live/test_cache_safety.py` is skipped unless `OPTIMIZER_LIVE=1` and `ANTHROPIC_API_KEY` are set.

- [ ] **Step 1: Add the retention purge to the proxy**

In `create_app`, after `client = ...`:
```python
    counter = {"n": 0}
    db.purge_expired(conn, time.time())
```
and at the top of the `proxy` handler:
```python
        counter["n"] += 1
        if counter["n"] % 1000 == 0:
            db.purge_expired(conn, time.time())
```

Run: `.venv/bin/pytest -q` — Expected: all passed (no behaviour change for tests).

- [ ] **Step 2: Write the entrypoint**

`optimizer/__main__.py`:
```python
import argparse
import logging

import uvicorn

from optimizer import db
from optimizer.config import load_config
from optimizer.proxy import create_app


def main():
    ap = argparse.ArgumentParser(prog="optimizer")
    ap.add_argument("--config", default=None, help="path to optimizer.toml (env OPTIMIZER_* overrides it)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(args.config)
    host, _, port = cfg.listen.rpartition(":")
    uvicorn.run(create_app(cfg, db.connect(cfg.db_path)), host=host or "0.0.0.0", port=int(port), log_level="info")


if __name__ == "__main__":
    main()
```

`optimizer.example.toml` — copy the TOML block from spec §9 verbatim.

- [ ] **Step 3: Smoke-run the server**

Run (background, then curl, then stop):
```bash
(.venv/bin/python -m optimizer & echo $! > /tmp/opt.pid; sleep 2; curl -s localhost:8787/optimizer/health; kill $(cat /tmp/opt.pid))
```
Expected: `{"ok":true}`

- [ ] **Step 4: Write the live cache-safety test**

`tests/live/__init__.py`: empty.

`tests/live/test_cache_safety.py`:
```python
"""Standing live test (spec §11): a route pinned to P2 must still hit the customer's prompt cache.

Run: OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live -v
Costs ~2 small Opus 5.5 requests with a ~1.5k-token cached system prompt.
"""
import json
import os

import httpx
import pytest

from optimizer import db
from optimizer.config import Config
from optimizer.proxy import create_app

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_LIVE") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
                                reason="set OPTIMIZER_LIVE=1 and ANTHROPIC_API_KEY to run")

SYSTEM = ("You are a support classifier for an online store. " * 150)  # ~1.5k tokens, above the 512-token cache minimum
REQ = {"model": "claude-opus-5-5", "max_tokens": 64,
       "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
       "messages": [{"role": "user", "content": "Classify: 'Where is my order #123?' Reply with one word."}]}


@pytest.mark.anyio
async def test_p2_pinned_route_keeps_cache_reads():
    conn = db.connect(":memory:")
    app = create_app(Config(sample_rate=0), conn, client=httpx.AsyncClient(timeout=120))
    hdrs = {"content-type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01", "X-Optimizer-Route": "live-cache-test"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=120) as c:
        r1 = await c.post("/v1/messages", content=json.dumps(REQ).encode(), headers=hdrs)
        assert r1.status_code == 200, r1.text
        db.set_pin(conn, "live-cache-test", "P2")
        r2 = await c.post("/v1/messages", content=json.dumps(REQ).encode(), headers=hdrs)
        assert r2.status_code == 200, r2.text
    u2 = r2.json()["usage"]
    assert u2["cache_read_input_tokens"] > 0, u2
    assert db.get_route(conn, "live-cache-test")["injection_form"] == "system"
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
```

Run: `.venv/bin/pytest tests/live -v` — Expected: 1 skipped (no key). If you have a key: `OPTIMIZER_LIVE=1 .venv/bin/pytest tests/live -v` — Expected: 1 passed.

- [ ] **Step 5: Write the README**

`README.md`:
```markdown
# Output-Token Optimizer (data plane)

Self-hosted drop-in proxy for the Claude and OpenAI APIs. Passes traffic through byte-for-byte, fingerprints routes,
records usage, and — once a route has a pinned output profile — rewrites requests cache-safely to shorten outputs.
Design: `docs/design/specs/2026-10-07-output-token-optimizer-design.md`.

## Run

    python3 -m venv .venv && .venv/bin/pip install -e .
    cp optimizer.example.toml optimizer.toml
    .venv/bin/python -m optimizer --config optimizer.toml

Point your client at it and keep your own API key:

    ANTHROPIC_BASE_URL=http://localhost:8787   # Anthropic SDKs
    OPENAI_BASE_URL=http://localhost:8787/v1   # OpenAI SDKs

## Kill switches

- Per request: header `X-Optimizer: off` (forces P0). `X-Optimizer: bypass` also skips recording.
- Per route: `[routes."<key>"] enabled = false` in `optimizer.toml`.
- Global: `OPTIMIZER_ENABLED=0`.

Any proxy-side failure forwards your original request unchanged.

## Report

`GET /optimizer/report` (JSON) · `GET /optimizer/report.html`

## Tests

    .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest
    OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live   # live cache-safety check
```

- [ ] **Step 6: Run the full suite one last time**

Run: `.venv/bin/pytest -v`
Expected: 49 passed, 1 skipped

- [ ] **Step 7: Commit**

```bash
git add optimizer/__main__.py optimizer/proxy.py optimizer.example.toml README.md tests/live
git commit -m "feat: entrypoint, retention purge, README, live cache-safety test"
```

---

## Deferred to Plan 2 (control plane)

Sampling into `samples`, sweep runner and cost gate, judge prompt and noise floor, pin rule, shadow drift, `target_words`/`exemplar` computation, route eligibility (`eligible` column), `sweep_budget_usd_month` enforcement, and the `sweeps`/`judgments`/`shadow` tables' writers. The tables exist (Task 2) so Plan 2 never migrates the schema.
