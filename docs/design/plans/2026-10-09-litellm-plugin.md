# Plan 3: LiteLLM Guardrail Plugin Implementation Plan

**Goal:** A LiteLLM proxy guardrail that applies pith's pinned shape profiles and records usage inside LiteLLM, sharing the SQLite with the existing sweep CLI, which gains a `litellm` provider kind so it can tune those routes by replaying through the LiteLLM proxy.

**Architecture:** One new module, `pith/guardrail.py`: a framework-free `PithHooks` class with LiteLLM's four hook methods (pre-call rewrite + stash, post-call record, streaming tee + record, failure bookkeeping), and `PithGuardrail(PithHooks, CustomGuardrail)` defined only when LiteLLM is importable. The proxy's decision and recording cores are extracted into `proxy.choose` and `proxy.record` so both entry points share them. The sweep CLI learns a third provider, `litellm`, whose replays go to `litellm_upstream` with an `X-Optimizer: bypass` header and whose profiles are `P0 P2 P3`.

**Tech Stack:** Python 3.12, stdlib `sqlite3`/`json`/`logging`/`time`, httpx (`MockTransport` in tests), pytest + anyio (already dev deps). LiteLLM ≥ 1.104 is never a dependency of pith; it is only present where the guardrail is registered.

**Spec:** `docs/design/specs/2026-10-09-litellm-plugin-design.md` (authoritative; §2 lists the LiteLLM facts verified by spike). Parent specs `docs/design/specs/2026-10-07-output-token-optimizer-design.md` and `docs/design/specs/2026-10-07-control-plane-design.md`. Read all three before starting.

## Global Constraints

- Python ≥ 3.12; code under `pith/`, tests under `tests/`; pytest only, one `test_*.py` per module, no fixture frameworks beyond `tests/conftest.py`'s `anyio_backend`; the suite must stay warning-free under `.venv/bin/pytest -q -W error`. Run everything with `.venv/bin/pytest` / `.venv/bin/python`.
- **No new dependency.** `pith/guardrail.py` must import when `litellm` is absent; `pyproject.toml` is not changed.
- **Fail open at every hook.** Every hook body is wrapped in `try/except Exception`; on failure log `type(exc).__name__` only (never the exception text, which can embed header values) and return the input unchanged.
- **Shape-only through LiteLLM:** `PROFILES_BY_PROVIDER["litellm"] == ("P0", "P2", "P3")`; `apply_profile("litellm", …)` touches only the last user message and only for P2/P3; every other profile returns the body unchanged.
- **Replays bypass the guardrail:** every `replay.call` to provider `litellm` carries header `x-optimizer: bypass`.
- Provider string is exactly `"litellm"`; config field `litellm_upstream` (default `"http://localhost:4000"`); env key `LITELLM_API_KEY`; env `OPTIMIZER_CONFIG` names the toml for the guardrail.
- Commit messages: plain conventional commits (`feat:`, `test:`, `refactor:`, `docs:`), no trailers, no co-author lines, no tool names anywhere in the repo.
- Work on branch `litellm-plugin` (already exists with the spec committed).

## File structure

| File | Responsibility |
|---|---|
| `pith/config.py` (modify) | `litellm_upstream` field |
| `pith/providers.py` (modify) | `upstream("litellm")` |
| `pith/__main__.py` (modify) | `ENV_KEYS["litellm"]` |
| `pith/sweep.py` (modify) | `PROFILES_BY_PROVIDER["litellm"]` |
| `pith/rewrite.py` (modify) | `litellm` branch of `apply_profile` |
| `pith/replay.py` (modify) | bypass header for `litellm` |
| `pith/proxy.py` (modify) | extract `choose` and `record`; `_decide` and the two record call sites use them |
| `pith/guardrail.py` (new) | `PithHooks` (four hooks), `PithGuardrail` |
| `tests/test_config.py`, `test_providers.py`, `test_main.py`, `test_sweep.py`, `test_rewrite.py`, `test_replay.py`, `test_proxy.py` (modify); `tests/test_guardrail.py`, `tests/live/test_litellm_mock.py` (new) | tests |
| `README.md`, `pith.example.toml`, spec §3 (modify) | docs |

---

### Task 1: Provider kind `litellm` in config, providers, CLI keys, profile list

**Files:**
- Modify: `pith/config.py` (the `Config` dataclass), `pith/providers.py:15-16`, `pith/__main__.py:12`, `pith/sweep.py:18`, `pith.example.toml`
- Test: `tests/test_config.py`, `tests/test_providers.py`, `tests/test_main.py`, `tests/test_sweep.py`

**Interfaces:**
- Produces: `Config.litellm_upstream: str = "http://localhost:4000"`; `providers.upstream("litellm", cfg) -> cfg.litellm_upstream`; `__main__.ENV_KEYS["litellm"] == "LITELLM_API_KEY"`; `sweep.PROFILES_BY_PROVIDER["litellm"] == ("P0", "P2", "P3")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_litellm_upstream_default_and_env_override():
    assert Config().litellm_upstream == "http://localhost:4000"
    assert load_config(None, env={"OPTIMIZER_LITELLM_UPSTREAM": "http://l:4000"}).litellm_upstream == "http://l:4000"
```

In `tests/test_providers.py` replace `test_upstream_from_config` with:

```python
def test_upstream_from_config():
    c = Config(anthropic_upstream="http://a", openai_upstream="http://o", litellm_upstream="http://l")
    assert upstream("anthropic", c) == "http://a" and upstream("openai", c) == "http://o"
    assert upstream("litellm", c) == "http://l"
```

In `tests/test_main.py` replace `test_keys_from_env` with:

```python
def test_keys_from_env():
    assert keys_from_env({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "LITELLM_API_KEY": "l", "X": "1"}) == \
        {"anthropic": "a", "openai": "o", "litellm": "l"}
    assert keys_from_env({"OPENAI_API_KEY": ""}) == {}
```

In `tests/test_sweep.py` replace `test_profiles_constant` with:

```python
def test_profiles_constant():
    assert PROFILES_BY_PROVIDER == {"anthropic": ("P0", "P1", "P2", "P3", "P4"),
                                    "openai": ("P0", "P1", "P1b", "P2", "P3", "P4"),
                                    "litellm": ("P0", "P2", "P3")}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py tests/test_providers.py tests/test_main.py::test_keys_from_env tests/test_sweep.py::test_profiles_constant -q`
Expected: 4 failures (`TypeError: unexpected keyword 'litellm_upstream'`, `AttributeError`, two `AssertionError`s).

- [ ] **Step 3: Implement**

`pith/config.py` — in `Config`, after `openai_upstream`, add:

```python
    litellm_upstream: str = "http://localhost:4000"  # a LiteLLM proxy; sweeps replay routes recorded by pith.guardrail
```

`pith/providers.py` — replace `upstream`:

```python
def upstream(provider: str, config: Config) -> str:
    if provider == "anthropic":
        return config.anthropic_upstream
    return config.litellm_upstream if provider == "litellm" else config.openai_upstream
```

`pith/__main__.py:12`:

```python
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "litellm": "LITELLM_API_KEY"}
```

`pith/sweep.py:18`:

```python
PROFILES_BY_PROVIDER = {"anthropic": ("P0", "P1", "P2", "P3", "P4"), "openai": ("P0", "P1", "P1b", "P2", "P3", "P4"),
                        "litellm": ("P0", "P2", "P3")}  # through LiteLLM only user-text shape is cache-safe (plan 3 spec §1)
```

`pith.example.toml` — after the `openai_upstream` line add:

```toml
litellm_upstream = "http://localhost:4000"   # only used by sweeps of routes recorded by the LiteLLM guardrail
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pith/config.py pith/providers.py pith/__main__.py pith/sweep.py pith.example.toml tests/test_config.py tests/test_providers.py tests/test_main.py tests/test_sweep.py
git commit -m "feat: litellm provider kind for config, upstream, keys and profiles"
```

---

### Task 2: `apply_profile("litellm")` and the replay bypass header

**Files:**
- Modify: `pith/rewrite.py:83-107` (`apply_profile`), `pith/replay.py:102-106` (`call`)
- Test: `tests/test_rewrite.py`, `tests/test_replay.py`

**Interfaces:**
- Consumes: `Config.litellm_upstream` (Task 1).
- Produces: `apply_profile("litellm", body, state, responses_api=False) -> dict` — P2/P3 append `{"type": "text", "text": shape}` to the last user message's content (string content becomes a list of text parts), ignoring `state.injection_form`; every other profile returns a deep copy equal to `body`. `replay.call(client, cfg, "litellm", body, key, prices)` posts to `cfg.litellm_upstream + "/v1/chat/completions"` with `authorization: Bearer <key>` and `x-optimizer: bypass`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rewrite.py`:

```python
def test_litellm_profiles_are_user_text_shape_only():
    out = apply_profile("litellm", CHAT, RouteState("P2", target_words=30))
    assert out["messages"][-1] == {"role": "user", "content": [{"type": "text", "text": "q"},
                                                               {"type": "text", "text": SHAPE_TEXT.format(n=30)}]}
    assert out["messages"][:-1] == CHAT["messages"][:-1] and untouched(CHAT, out)
    out = apply_profile("litellm", CHAT, RouteState("P2", injection_form="system", target_words=30))
    assert out["messages"][-1]["role"] == "user" and len(out["messages"]) == len(CHAT["messages"])  # form column ignored
    out = apply_profile("litellm", CHAT, RouteState("P3", target_words=20, exemplar="Yes."))
    assert out["messages"][-1]["content"][-1]["text"] == SHAPE_TEXT.format(n=20) + "\n\nExample of the expected length:\nYes."
    for p in ("P1", "P1b", "P4"):
        assert apply_profile("litellm", CHAT, RouteState(p)) == CHAT
```

In `tests/test_replay.py` replace `test_endpoint_and_headers` with:

```python
def test_endpoint_and_headers():
    assert endpoint_for("anthropic", {"messages": []}) == "/v1/messages"
    assert endpoint_for("openai", {"messages": []}) == "/v1/chat/completions"
    assert endpoint_for("openai", {"input": "q"}) == "/v1/responses"
    assert endpoint_for("litellm", {"messages": []}) == "/v1/chat/completions"
    assert auth_headers("anthropic", "k") == {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    assert auth_headers("openai", "k") == {"authorization": "Bearer k"}
    assert auth_headers("litellm", "k") == {"authorization": "Bearer k"}
```

and append:

```python
def test_call_through_litellm_uses_chat_endpoint_and_bypass_header():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json=CHAT)
    client = httpx.Client(transport=httpx.MockTransport(h))
    r = call(client, Config(litellm_upstream="http://l:4000"), "litellm", {"model": "mock", "messages": []}, "k", {"mock": (1.0, 2.0)})
    assert r.status == 200 and r.text == "hi" and r.cost_usd == (5 * 1.0 + 1 * 2.0) / 1e6
    assert str(seen[0].url) == "http://l:4000/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer k" and seen[0].headers["x-optimizer"] == "bypass"
    call(client, Config(), "openai", {"model": "m", "messages": []}, "k")
    assert "x-optimizer" not in seen[1].headers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rewrite.py::test_litellm_profiles_are_user_text_shape_only tests/test_replay.py -q`
Expected: the rewrite test fails (P2 on `litellm` currently appends a `developer` message); the bypass-header test fails with `KeyError: 'x-optimizer'`.

- [ ] **Step 3: Implement**

`pith/rewrite.py` — in `apply_profile`, immediately after the `if provider == "anthropic": … return out` block, add:

```python
    if provider == "litellm":
        # LiteLLM folds system/developer messages into the provider's system prompt (cache-breaking) and a guardrail
        # cannot retry a rejected request: only the user-text shape, nothing else.
        if p in ("P2", "P3"):
            _anthropic_shape(out, RouteState(p, "user_text", state.target_words, state.exemplar))
        return out
```

`pith/replay.py` — in `call`, right after `headers = {**auth_headers(provider, key), "content-type": "application/json"}` add:

```python
    if provider == "litellm":
        headers["x-optimizer"] = "bypass"  # the pith guardrail inside LiteLLM must neither rewrite nor record replays
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pith/rewrite.py pith/replay.py tests/test_rewrite.py tests/test_replay.py
git commit -m "feat: user-text shape profiles and bypass header for the litellm provider"
```

---

### Task 3: Extract `choose` and `record` from the proxy

**Files:**
- Modify: `pith/proxy.py:39-58` (`_decide`), `pith/proxy.py:83-91` (`_record`), the two `_record(` call sites in `create_app`
- Test: `tests/test_proxy.py`

**Interfaces:**
- Produces: `proxy.choose(config, conn, provider: str, body: dict, mode: str, route_name: str | None, responses_api: bool = False) -> tuple[Fingerprint, dict, str, dict]` returning `(fp, route_row, profile, body_to_send)`, where `body_to_send is body` at P0 and a rewritten deep copy otherwise; `proxy.record(config, conn, route_key: str, profile: str, usage: Usage, latency_ms: int, request_json: str, response_json: str) -> None`.
- Every existing test in `tests/test_proxy.py` must pass unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy.py` (add `choose, record` to the existing `from pith.proxy import create_app` line and `from pith.usage import Usage` to the imports):

```python
def test_choose_is_the_shared_decision_core():
    conn = db.connect(":memory:")
    cfg = Config()
    fp, route, profile, body = choose(cfg, conn, "anthropic", ANTH_REQ, "", None)
    assert profile == "P0" and body is ANTH_REQ and route["key"] == fp.key and fp.model == "claude-opus-5-5"
    db.set_pin(conn, fp.key, "P2")
    fp2, route, profile, body = choose(cfg, conn, "anthropic", ANTH_REQ, "", None)
    assert fp2 == fp and profile == "P2" and body["messages"][-1]["role"] == "system" and body is not ANTH_REQ
    assert choose(cfg, conn, "anthropic", ANTH_REQ, "off", None)[2] == "P0"
    assert choose(cfg, conn, "anthropic", ANTH_REQ, "", "named")[0].key == "named"
    assert choose(Config(enabled=False), conn, "anthropic", ANTH_REQ, "", None)[2] == "P0"


def test_record_samples_by_rate_and_stores_strings():
    conn = db.connect(":memory:")
    db.upsert_route(conn, "k", "anthropic", "m", "h")
    record(Config(sample_rate=1.0), conn, "k", "P2", Usage(1, 2, 0, 0, "end_turn"), 7, '{"a":1}', '{"b":2}')
    record(Config(sample_rate=0.0), conn, "k", "P0", Usage(1, 2, 0, 0, "end_turn", estimated=True), 8, '{"a":1}', '{"b":2}')
    rows = conn.execute("SELECT profile, latency_ms, body_ref, estimated FROM requests ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("P2", 7, 1, 0), ("P0", 8, None, 1)]
    assert conn.execute("SELECT request_json, response_json FROM bodies").fetchone()[:] == ('{"a":1}', '{"b":2}')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_proxy.py -q`
Expected: `ImportError: cannot import name 'choose'`.

- [ ] **Step 3: Implement**

`pith/proxy.py` — replace `_decide` with:

```python
def choose(config: Config, conn, provider: str, body: dict, mode: str, route_name: str | None,
           responses_api: bool = False) -> tuple[Fingerprint, dict, str, dict]:
    """Fingerprint, register and pick the profile for one request: (fp, route, profile, body_to_send).
    body_to_send is `body` itself at P0. Shared by the proxy and the LiteLLM guardrail."""
    fp = fingerprint(provider, body, route_name)
    db.upsert_route(conn, fp.key, provider, fp.model, fp.system_hash, route_name)
    route = db.get_route(conn, fp.key)
    route_cfg = config.routes.get(fp.key)
    enabled = config.enabled and mode != "off" and (route_cfg is None or route_cfg.enabled)
    profile = route["pinned_profile"] if enabled else "P0"
    if profile == "P0":
        return fp, route, "P0", body
    state = RouteState(profile, route["injection_form"], route["target_words"], route["exemplar"])
    return fp, route, profile, apply_profile(provider, body, state, responses_api=responses_api)


def _decide(config: Config, conn, path: str, headers, raw: bytes) -> Decision:
    """Everything before forwarding. Any exception here is caught by the caller -> fail open."""
    provider = detect_provider(path)
    if provider is None:
        return Decision(None, None, None, "P0", raw, False)
    mode = (headers.get("x-optimizer") or "").lower()
    if mode == "bypass":
        return Decision(provider, None, None, "P0", raw, False)
    fp, route, profile, body = choose(config, conn, provider, json.loads(raw), mode, headers.get("x-optimizer-route"),
                                      responses_api=is_responses_api(path))
    return Decision(provider, fp, route, profile, raw if profile == "P0" else json.dumps(body).encode(), True)
```

Replace `_record` with:

```python
def record(config: Config, conn, route_key: str, profile: str, usage, latency_ms: int, request_json: str,
           response_json: str) -> None:
    body_ref = None
    if random.random() < config.sample_rate:
        body_ref = db.store_body(conn, request_json, response_json, time.time() + config.retention_days * 86400)
    db.record_request(conn, ts=time.time(), route_key=route_key, profile=profile, input_tokens=usage.input_tokens,
                      output_tokens=usage.output_tokens, cache_read=usage.cache_read, cache_create=usage.cache_create,
                      estimated=usage.estimated, stop_reason=usage.stop_reason, latency_ms=latency_ms, body_ref=body_ref)
```

In `create_app`, the streaming call site becomes:

```python
                            record(config, conn, d.fp.key, profile_used, su.result(), int((time.monotonic() - t0) * 1000),
                                   raw.decode("utf-8", "replace"), b"".join(collected).decode("utf-8", "replace"))
```

and the non-streaming one:

```python
                record(config, conn, d.fp.key, profile_used, usage, latency, raw.decode("utf-8", "replace"),
                       content.decode("utf-8", "replace"))
```

Remove the old `_record` entirely; `grep -n "_record" pith/` must print nothing.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass, including every pre-existing `tests/test_proxy.py` test.

- [ ] **Step 5: Commit**

```bash
git add pith/proxy.py tests/test_proxy.py
git commit -m "refactor: extract choose and record from the proxy for reuse"
```

---

### Task 4: `pith/guardrail.py` — `PithHooks` pre-call hook

**Files:**
- Create: `pith/guardrail.py`
- Test: `tests/test_guardrail.py`

**Interfaces:**
- Consumes: `proxy.choose` (Task 3), `apply_profile("litellm", …)` (Task 2), `load_config`, `db.connect`.
- Produces: `PithHooks(config: Config | None = None, conn=None, env: Mapping | None = None)`; `async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type: str) -> dict`; module constants `PROVIDER = "litellm"`, `CHAT_CALLS = ("completion", "acompletion")`, `LITELLM_KEYS`; helper `_stash(data) -> dict | None` returning `data["metadata"]["pith"]`. The stash is `{"route": str, "profile": str, "body": dict, "t0": float}` and Task 5 reads it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_guardrail.py`:

```python
import json
import logging

import pytest

from pith import db
from pith.config import Config
from pith.guardrail import PithHooks

BODY = {"model": "mock", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], "max_tokens": 50,
        "metadata": {"pith_route": "ignored"}}
CLEAN = {k: v for k, v in BODY.items() if k != "metadata"}


def req(body=BODY, headers=None):
    """What LiteLLM hands async_pre_call_hook on /v1/chat/completions (spec §2)."""
    hdrs = {"content-type": "application/json", **{k.lower(): v for k, v in (headers or {}).items()}}
    return {**{k: v for k, v in body.items() if k != "metadata"}, "litellm_call_id": "c1", "litellm_logging_obj": object(),
            "secret_fields": {}, "metadata": {**body.get("metadata", {}), "headers": hdrs, "user_api_key_hash": "h"},
            "proxy_server_request": {"url": "http://l/v1/chat/completions", "method": "POST", "headers": hdrs, "body": dict(body)}}


def hooks(config=None):
    conn = db.connect(":memory:")
    return PithHooks(config or Config(sample_rate=1.0), conn), conn


class Boom:
    def execute(self, *a, **k):
        raise RuntimeError("x-api-key: sk-secret")


@pytest.mark.anyio
async def test_pre_call_registers_route_and_stashes_original_body_at_p0():
    h, conn = hooks()
    data = req()
    out = await h.async_pre_call_hook({}, None, data, "acompletion")
    assert out is data
    route = conn.execute("SELECT * FROM routes").fetchone()
    assert route["provider"] == "litellm" and route["model"] == "mock" and route["key"].startswith("litellm:mock:")
    stash = data["metadata"]["pith"]
    assert stash["route"] == route["key"] and stash["profile"] == "P0" and stash["body"] == CLEAN
    assert {k: data[k] for k in CLEAN} == CLEAN and data["metadata"]["pith_route"] == "ignored"


@pytest.mark.anyio
async def test_pre_call_ignores_non_chat_calls_and_bypass():
    h, conn = hooks()
    data = req()
    await h.async_pre_call_hook({}, None, data, "aembedding")
    assert "pith" not in data["metadata"] and conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0
    data = req(headers={"X-Optimizer": "bypass"})
    await h.async_pre_call_hook({}, None, data, "acompletion")
    assert "pith" not in data["metadata"] and conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


@pytest.mark.anyio
async def test_pre_call_applies_p2_pin_as_user_text_and_off_forces_p0():
    h, conn = hooks()
    await h.async_pre_call_hook({}, None, req(headers={"X-Optimizer-Route": "billing"}), "acompletion")
    assert db.get_route(conn, "billing")["provider"] == "litellm"
    db.set_pin(conn, "billing", "P2")
    data = req(headers={"X-Optimizer-Route": "billing"})
    await h.async_pre_call_hook({}, None, data, "acompletion")
    last = data["messages"][-1]
    assert last["role"] == "user" and last["content"][0] == {"type": "text", "text": "q"}
    assert last["content"][1]["type"] == "text" and last["content"][1]["text"].startswith("Answer directly.")
    assert data["messages"][0] == BODY["messages"][0] and data["model"] == "mock" and data["max_tokens"] == 50
    assert data["metadata"]["pith"]["profile"] == "P2" and data["metadata"]["pith"]["body"] == CLEAN
    data = req(headers={"X-Optimizer-Route": "billing", "X-Optimizer": "off"})
    await h.async_pre_call_hook({}, None, data, "acompletion")
    assert data["messages"] == BODY["messages"] and data["metadata"]["pith"]["profile"] == "P0"


@pytest.mark.anyio
async def test_pre_call_without_proxy_snapshot_uses_data_minus_litellm_keys():
    h, conn = hooks()
    data = req()
    del data["proxy_server_request"]
    await h.async_pre_call_hook({}, None, data, "acompletion")
    assert data["metadata"]["pith"]["body"] == CLEAN


@pytest.mark.anyio
async def test_pre_call_fails_open_and_logs_only_class_name(caplog):
    h = PithHooks(Config(), Boom())
    data = req()
    with caplog.at_level(logging.WARNING, logger="pith.guardrail"):
        out = await h.async_pre_call_hook({}, None, data, "acompletion")
    assert out is data and data["messages"] == BODY["messages"] and "pith" not in data["metadata"]
    assert "sk-secret" not in caplog.text and "RuntimeError" in caplog.text


def test_lazy_config_and_connection_from_env(tmp_path):
    h = PithHooks(env={"OPTIMIZER_DB_PATH": str(tmp_path / "g.db"), "OPTIMIZER_SAMPLE_RATE": "1"})
    cfg, conn = h._ready()
    assert cfg.db_path == str(tmp_path / "g.db") and cfg.sample_rate == 1.0 and (tmp_path / "g.db").exists()
    assert h._ready() == (cfg, conn)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_guardrail.py -q`
Expected: `ModuleNotFoundError: No module named 'pith.guardrail'`.

- [ ] **Step 3: Implement**

Create `pith/guardrail.py`:

```python
"""LiteLLM guardrail: the data plane inside a LiteLLM proxy. Spec: docs/design/specs/2026-10-09-litellm-plugin-design.md.

`PithHooks` is framework-free and fully testable; `PithGuardrail` (defined only where litellm is importable) inherits
it next to litellm's CustomGuardrail. Every hook fails open: a guardrail exception would reject the customer's request.
"""
import json
import logging
import os
import time
from typing import Mapping

from pith import db
from pith.config import Config, load_config
from pith.proxy import PURGE_EVERY, choose, record
from pith.usage import estimate_tokens, usage_from_body

log = logging.getLogger("pith.guardrail")
PROVIDER = "litellm"
CHAT_CALLS = ("completion", "acompletion")
LITELLM_KEYS = {"litellm_call_id", "litellm_logging_obj", "metadata", "litellm_metadata", "proxy_server_request",
                "secret_fields"}


def _stash(data) -> dict | None:
    return ((data or {}).get("metadata") or {}).get("pith")


class PithHooks:
    def __init__(self, config: Config | None = None, conn=None, env: Mapping[str, str] | None = None):
        self._config, self._conn = config, conn
        self._env = os.environ if env is None else env
        self._n = 0

    def _ready(self):
        if self._config is None:
            self._config = load_config(self._env.get("OPTIMIZER_CONFIG"), self._env)
        if self._conn is None:
            self._conn = db.connect(self._config.db_path)  # created on the event-loop thread, on first use
        return self._config, self._conn

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            if call_type not in CHAT_CALLS or not isinstance(data, dict) or "messages" not in data:
                return data
            cfg, conn = self._ready()
            snapshot = (data.get("proxy_server_request") or {}).get("body")
            body = dict(snapshot) if snapshot is not None else {k: v for k, v in data.items() if k not in LITELLM_KEYS}
            body.pop("metadata", None)  # LiteLLM's merged metadata must never be replayed
            headers = {k.lower(): v for k, v in (((data.get("metadata") or {}).get("headers")) or {}).items()}
            mode = (headers.get("x-optimizer") or "").lower()
            if mode == "bypass":
                return data
            fp, route, profile, new_body = choose(cfg, conn, PROVIDER, body, mode, headers.get("x-optimizer-route"))
            if profile != "P0":
                data["messages"] = new_body["messages"]
            data.setdefault("metadata", {})["pith"] = {"route": fp.key, "profile": profile, "body": body,
                                                       "t0": time.monotonic()}
        except Exception as exc:  # fail open; never log exc text (it can embed header values)
            log.warning("pre_call failed (%s); request forwarded unchanged", type(exc).__name__)
        return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_guardrail.py -q -W error`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add pith/guardrail.py tests/test_guardrail.py
git commit -m "feat: litellm guardrail pre-call hook applies pins and stashes state"
```

---

### Task 5: Post-call, streaming and failure hooks

**Files:**
- Modify: `pith/guardrail.py` (add three methods and `_record` to `PithHooks`)
- Test: `tests/test_guardrail.py`

**Interfaces:**
- Consumes: the stash from Task 4; `proxy.record`, `usage_from_body("openai", …)`, `estimate_tokens`, `db.bump_rejection`, `db.set_pin`, `db.purge_expired`.
- Produces: `async_post_call_success_hook(self, data, user_api_key_dict, response) -> response`; `async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data)` (async generator yielding each chunk unchanged); `async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict, traceback_str=None) -> None`. Stored streaming responses are chat-completion dicts readable by `replay.stored_response_text("openai", …)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_guardrail.py` (add `from pith.replay import stored_response_text` to the imports):

```python
class FakeResponse:
    def __init__(self, d):
        self._d = d

    def model_dump(self):
        return dict(self._d)


CHAT_RESP = {"id": "x", "object": "chat.completion", "model": "gpt-4o",
             "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 20}}


async def chunks(items):
    for it in items:
        yield it


def chunk(text=None, finish=None, usage=None):
    return FakeResponse({"object": "chat.completion.chunk", "model": "gpt-4o",
                         "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}],
                         **({"usage": usage} if usage else {})})


class Rejected(Exception):
    def __init__(self, status):
        super().__init__("provider said no: x-api-key sk-secret")
        self.status_code = status


@pytest.mark.anyio
async def test_post_call_records_usage_profile_and_samples_body():
    h, conn = hooks()
    data = req()
    await h.async_pre_call_hook({}, None, data, "acompletion")
    resp = FakeResponse(CHAT_RESP)
    assert await h.async_post_call_success_hook(data, {}, resp) is resp
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["profile"], row["input_tokens"], row["output_tokens"], row["stop_reason"], row["estimated"]) == ("P0", 10, 20, "stop", 0)
    assert row["latency_ms"] >= 0
    body = conn.execute("SELECT * FROM bodies").fetchone()
    assert json.loads(body["request_json"]) == CLEAN and json.loads(body["response_json"]) == CHAT_RESP
    assert stored_response_text("openai", body["response_json"]) == "Hello"


@pytest.mark.anyio
async def test_post_call_without_stash_or_model_dump_records_nothing_and_fails_open():
    h, conn = hooks()
    await h.async_post_call_success_hook(req(), {}, FakeResponse(CHAT_RESP))
    await h.async_post_call_success_hook(req(), {}, {"not": "a model"})
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    data = req()
    data["metadata"]["pith"] = {"route": "r", "profile": "P0", "body": CLEAN, "t0": 0.0}
    resp = FakeResponse(CHAT_RESP)
    assert await PithHooks(Config(), Boom()).async_post_call_success_hook(data, {}, resp) is resp


@pytest.mark.anyio
async def test_streaming_passes_chunks_through_and_records_usage_from_last_chunk():
    h, conn = hooks()
    data = req(dict(BODY, stream=True))
    await h.async_pre_call_hook({}, None, data, "acompletion")
    src = [chunk("Hel"), chunk("lo"), chunk(finish="stop"), chunk(usage={"prompt_tokens": 14, "completion_tokens": 3})]
    got = [c async for c in h.async_post_call_streaming_iterator_hook({}, chunks(src), data)]
    assert got == src
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["profile"], row["input_tokens"], row["output_tokens"], row["stop_reason"], row["estimated"]) == ("P0", 14, 3, "stop", 0)
    stored = conn.execute("SELECT response_json FROM bodies").fetchone()[0]
    assert stored_response_text("openai", stored) == "Hello" and json.loads(stored)["choices"][0]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_streaming_without_usage_estimates_and_early_stop_records_nothing():
    h, conn = hooks()
    data = req(dict(BODY, stream=True))
    await h.async_pre_call_hook({}, None, data, "acompletion")
    src = [chunk("Hello world"), chunk(finish="stop")]
    [c async for c in h.async_post_call_streaming_iterator_hook({}, chunks(src), data)]
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert row["output_tokens"] >= 1 and row["estimated"] == 1 and row["input_tokens"] is None and row["stop_reason"] == "stop"
    data = req(dict(BODY, stream=True))
    await h.async_pre_call_hook({}, None, data, "acompletion")
    agen = h.async_post_call_streaming_iterator_hook({}, chunks(src), data)
    assert await agen.__anext__() is src[0]
    await agen.aclose()
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
    # garbage chunks and a broken connection never break the customer's stream
    bad = [FakeResponse({"choices": "nope"}), object(), chunk(finish="stop")]
    data = req(dict(BODY, stream=True))
    data["metadata"]["pith"] = {"route": "r", "profile": "P0", "body": CLEAN, "t0": 0.0}
    assert [c async for c in PithHooks(Config(), Boom()).async_post_call_streaming_iterator_hook({}, chunks(bad), data)] == bad


@pytest.mark.anyio
async def test_failure_hook_counts_once_per_request_and_reverts_after_three():
    h, conn = hooks()
    await h.async_pre_call_hook({}, None, req(headers={"X-Optimizer-Route": "r"}), "acompletion")
    db.set_pin(conn, "r", "P2")
    for i in range(3):
        data = req(headers={"X-Optimizer-Route": "r"})
        await h.async_pre_call_hook({}, None, data, "acompletion")
        assert data["metadata"]["pith"]["profile"] == "P2"
        await h.async_post_call_failure_hook(data, Rejected(400), {})
        await h.async_post_call_failure_hook(data, Rejected(400), {})  # LiteLLM fires it twice per failure
        assert db.get_route(conn, "r")["rejections"] == (i + 1 if i < 2 else 0)
    r = db.get_route(conn, "r")
    assert r["pinned_profile"] == "P0" and r["status"] == "reverted"
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


@pytest.mark.anyio
async def test_failure_hook_ignores_p0_non_4xx_missing_stash_and_fails_open(caplog):
    h, conn = hooks()
    await h.async_pre_call_hook({}, None, req(headers={"X-Optimizer-Route": "r"}), "acompletion")
    db.set_pin(conn, "r", "P2")
    data = req(headers={"X-Optimizer-Route": "r"})
    await h.async_pre_call_hook({}, None, data, "acompletion")
    await h.async_post_call_failure_hook(data, Rejected(500), {})
    await h.async_post_call_failure_hook(data, RuntimeError("boom"), {})
    await h.async_post_call_failure_hook(req(), Rejected(400), {})
    p0 = req(headers={"X-Optimizer-Route": "r", "X-Optimizer": "off"})
    await h.async_pre_call_hook({}, None, p0, "acompletion")
    await h.async_post_call_failure_hook(p0, Rejected(400), {})
    assert db.get_route(conn, "r")["rejections"] == 0 and db.get_route(conn, "r")["pinned_profile"] == "P2"
    data["metadata"]["pith"]["failed"] = False
    with caplog.at_level(logging.WARNING, logger="pith.guardrail"):
        await PithHooks(Config(), Boom()).async_post_call_failure_hook(data, Rejected(400), {})
    assert "sk-secret" not in caplog.text and "RuntimeError" in caplog.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_guardrail.py -q`
Expected: the six new tests fail with `AttributeError: 'PithHooks' object has no attribute 'async_post_call_success_hook'` (and the streaming/failure equivalents).

- [ ] **Step 3: Implement**

Append to `class PithHooks` in `pith/guardrail.py`:

```python
    def _record(self, stash: dict, resp: dict, usage) -> None:
        cfg, conn = self._ready()
        self._n += 1
        if self._n % PURGE_EVERY == 0:
            db.purge_expired(conn, time.time())
        record(cfg, conn, stash["route"], stash["profile"], usage, int((time.monotonic() - stash["t0"]) * 1000),
               json.dumps(stash["body"]), json.dumps(resp, default=str))

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            stash = _stash(data)
            if stash and hasattr(response, "model_dump"):
                resp = response.model_dump()
                self._record(stash, resp, usage_from_body("openai", resp))
        except Exception as exc:
            log.warning("post_call record failed (%s)", type(exc).__name__)
        return response

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        text, finish, usage, model = [], None, None, None
        async for chunk in response:
            try:  # read-only tee on the customer's stream: a bad chunk is passed on, not parsed
                d = chunk.model_dump()
                model = d.get("model") or model
                for c in d.get("choices") or []:
                    t = (c.get("delta") or {}).get("content")
                    if t:
                        text.append(t)
                    finish = c.get("finish_reason") or finish
                usage = d.get("usage") or usage
            except Exception:
                pass
            yield chunk
        try:  # reached only when the stream ended; a consumer that stops early closes the generator at `yield`
            stash = _stash(request_data)
            if stash:
                resp = {"object": "chat.completion", "model": model, "usage": usage or {},
                        "choices": [{"index": 0, "finish_reason": finish,
                                     "message": {"role": "assistant", "content": "".join(text)}}]}
                u = usage_from_body("openai", resp)
                if u.output_tokens is None and text:
                    u.output_tokens, u.estimated = estimate_tokens("".join(text)), True
                self._record(stash, resp, u)
        except Exception as exc:
            log.warning("stream record failed (%s)", type(exc).__name__)

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict, traceback_str=None):
        try:
            stash = _stash(request_data)
            if not stash or stash["profile"] == "P0" or stash.get("failed") or \
                    getattr(original_exception, "status_code", None) not in (400, 422):
                return
            stash["failed"] = True  # LiteLLM fires this hook twice per failure
            cfg, conn = self._ready()
            if db.bump_rejection(conn, stash["route"]) >= 3:
                db.set_pin(conn, stash["route"], "P0", status="reverted")
                log.warning("route %s reverted to P0 after 3 provider rejections", stash["route"])
        except Exception as exc:
            log.warning("failure bookkeeping failed (%s)", type(exc).__name__)
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pith/guardrail.py tests/test_guardrail.py
git commit -m "feat: guardrail records responses, tees streams and reverts rejected pins"
```

---

### Task 6: `PithGuardrail`, import-without-LiteLLM guarantee, live mock test

**Files:**
- Modify: `pith/guardrail.py` (module tail), `docs/design/specs/2026-10-09-litellm-plugin-design.md` §3 (one line)
- Create: `tests/live/test_litellm_mock.py`
- Test: `tests/test_guardrail.py`

**Interfaces:**
- Produces: `pith.guardrail.PithGuardrail(PithHooks, CustomGuardrail)` present iff `litellm` imports; `pith.guardrail.CustomGuardrail` is `None` otherwise. `PithHooks` must precede `CustomGuardrail` in the MRO because `CustomGuardrail` defines default implementations of all four hooks.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_guardrail.py`:

```python
def test_module_imports_without_litellm_and_defines_guardrail_only_with_it():
    import importlib.util

    import pith.guardrail as g
    has = importlib.util.find_spec("litellm") is not None
    assert hasattr(g, "PithGuardrail") == has
    if has:
        mro = g.PithGuardrail.__mro__
        assert mro.index(PithHooks) < mro.index(g.CustomGuardrail)
        assert isinstance(g.PithGuardrail(guardrail_name="pith"), PithHooks)
    else:
        assert g.CustomGuardrail is None
```

Create `tests/live/test_litellm_mock.py`:

```python
"""Boots a real LiteLLM proxy with a mock model and the pith guardrail: no API key, no network.

Run: .venv/bin/pip install 'litellm[proxy]' && OPTIMIZER_LITELLM_LIVE=1 .venv/bin/pytest tests/live/test_litellm_mock.py -v
LiteLLM is not a pith dependency and is not installed in CI, so this file is skipped unless the flag is set.
"""
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from pith import db

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_LITELLM_LIVE") != "1", reason="set OPTIMIZER_LITELLM_LIVE=1")

CONFIG = """
model_list:
  - model_name: mock
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-fake
      mock_response: "Hello from mock"
guardrails:
  - guardrail_name: pith
    litellm_params:
      guardrail: pith.guardrail.PithGuardrail
      mode: [pre_call, post_call]
      default_on: true
general_settings:
  master_key: sk-test
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def proxy(tmp_path):
    pytest.importorskip("litellm")
    (tmp_path / "config.yaml").write_text(CONFIG)
    port = free_port()
    env = {**os.environ, "OPTIMIZER_DB_PATH": str(tmp_path / "pith.db"), "OPTIMIZER_SAMPLE_RATE": "1"}
    exe = os.path.join(os.path.dirname(sys.executable), "litellm")
    p = subprocess.Popen([exe, "--config", str(tmp_path / "config.yaml"), "--port", str(port), "--host", "127.0.0.1"],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(base + "/health/liveliness", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("litellm proxy did not start")
        yield base, str(tmp_path / "pith.db")
    finally:
        p.terminate()
        p.wait(timeout=10)


def post(base, body, headers=None):
    return httpx.post(base + "/v1/chat/completions", json=body, timeout=30,
                      headers={"authorization": "Bearer sk-test", **(headers or {})})


def test_guardrail_records_applies_pins_and_honours_bypass(proxy):
    base, db_path = proxy
    body = {"model": "mock", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]}
    hdr = {"X-Optimizer-Route": "mock-route"}
    assert post(base, body, hdr).status_code == 200
    r = post(base, dict(body, stream=True, stream_options={"include_usage": True}), hdr)
    assert r.status_code == 200 and "Hello from mock" in r.text
    assert post(base, body, {**hdr, "X-Optimizer": "bypass"}).status_code == 200
    conn = db.connect(db_path)
    route = db.get_route(conn, "mock-route")
    assert route["provider"] == "litellm" and route["model"] == "mock"
    rows = conn.execute("SELECT profile, output_tokens, stop_reason FROM requests ORDER BY id").fetchall()
    assert len(rows) == 2 and all(r["profile"] == "P0" and r["output_tokens"] and r["stop_reason"] == "stop" for r in rows)
    db.set_pin(conn, "mock-route", "P2")
    assert post(base, body, hdr).status_code == 200
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
    assert conn.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_guardrail.py::test_module_imports_without_litellm_and_defines_guardrail_only_with_it -q`
Expected: FAIL (`AttributeError: module 'pith.guardrail' has no attribute 'CustomGuardrail'`).

- [ ] **Step 3: Implement**

Append to `pith/guardrail.py`:

```python
try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:  # pith stays importable without LiteLLM; the guardrail class exists only where it can be registered
    CustomGuardrail = None

if CustomGuardrail is not None:
    class PithGuardrail(PithHooks, CustomGuardrail):  # PithHooks first: CustomGuardrail defines no-op defaults of every hook
        """config.yaml: guardrail: pith.guardrail.PithGuardrail, mode: [pre_call, post_call], default_on: true."""

        def __init__(self, **kwargs):
            CustomGuardrail.__init__(self, **kwargs)
            PithHooks.__init__(self)
```

In the spec `docs/design/specs/2026-10-09-litellm-plugin-design.md` §3, change
`PithGuardrail(litellm.integrations.custom_guardrail.CustomGuardrail, PithHooks)` to
`PithGuardrail(PithHooks, litellm.integrations.custom_guardrail.CustomGuardrail)` and append to that bullet:
"`PithHooks` comes first in the bases because `CustomGuardrail` defines default implementations of all four hooks."

- [ ] **Step 4: Run the suite, then the live mock test**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass; `tests/live/test_litellm_mock.py` reports skipped.

Then install LiteLLM into the project venv and run the live test (a few minutes for the install; the test itself takes ~30 s):

```bash
.venv/bin/pip install -q 'litellm[proxy]' && OPTIMIZER_LITELLM_LIVE=1 .venv/bin/pytest tests/live/test_litellm_mock.py tests/test_guardrail.py -v
```

Expected: both pass (with LiteLLM installed, the MRO test now exercises `PithGuardrail`). Run `.venv/bin/pytest -q -W error` once more with LiteLLM installed to confirm the suite stays warning-free (verified on LiteLLM 1.104.2 during the spike). LiteLLM may stay installed in `.venv`; it is not in `pyproject.toml`.

- [ ] **Step 5: Commit**

```bash
git add pith/guardrail.py tests/test_guardrail.py tests/live/test_litellm_mock.py docs/design/specs/2026-10-09-litellm-plugin-design.md
git commit -m "feat: PithGuardrail for LiteLLM with a mock-backed live test"
```

---

### Task 7: README section

**Files:**
- Modify: `README.md` (new section after "## Kill switches", before "## Sweeps: turning observation into pins"; one line in "## Tests")

- [ ] **Step 1: Insert the section**

After the "## Kill switches" section's last paragraph (`Any proxy-side failure forwards your original request unchanged.`) insert:

````markdown
## LiteLLM plugin

Already running a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)? Register pith as a guardrail instead of
adding a second hop. Install pith into the LiteLLM proxy's environment and add to its `config.yaml`:

    guardrails:
      - guardrail_name: pith
        litellm_params:
          guardrail: pith.guardrail.PithGuardrail
          mode: [pre_call, post_call]
          default_on: true

    # in the proxy's environment
    OPTIMIZER_CONFIG=/path/to/pith.toml      # optional; every setting is also an OPTIMIZER_<FIELD> variable
    pip install git+https://github.com/dex0shubham/pith

The guardrail fingerprints every `/v1/chat/completions` request, records usage into the same SQLite the CLI reads, and
applies a pinned profile by appending the shape text to the last user message. Only P2 and P3 apply through LiteLLM:
LiteLLM folds system messages into the provider's system prompt (cache-breaking), and a guardrail cannot retry a
rejected request, so effort profiles are left to the standalone proxy. The kill switches above still work: the
guardrail reads `X-Optimizer` and `X-Optimizer-Route` from the request headers LiteLLM records. Any pith error inside a
hook is logged and the request proceeds unchanged.

Sweep those routes from the same host, through the LiteLLM proxy (replays carry `X-Optimizer: bypass`, so the guardrail
ignores them):

    litellm_upstream = "http://localhost:4000"   # pith.toml
    [prices."<model_name as clients send it>"]   # required: LiteLLM aliases are not in the built-in price table
    input = 1.25
    output = 10.0

    export LITELLM_API_KEY=sk-...                 # a LiteLLM virtual key or the master key
    .venv/bin/python -m pith sweep --config pith.toml

Set `judge_provider = "litellm"` and a `judge_model` LiteLLM serves to run the judge through it as well.
````

In "## Tests" append after the `.env` paragraph:

```markdown
`OPTIMIZER_LITELLM_LIVE=1 .venv/bin/pytest tests/live/test_litellm_mock.py` boots a real LiteLLM proxy with a mock
model and the guardrail (needs `pip install 'litellm[proxy]'`, no API key).
```

- [ ] **Step 2: Check and commit**

Run: `.venv/bin/pytest -q -W error` (unchanged, all pass) and `grep -n "Claude\|Co-Authored\|Generated with" README.md` (must print nothing).

```bash
git add README.md
git commit -m "docs: LiteLLM plugin section"
```

---

## Self-review against the spec

- §1 decisions: 1 → Tasks 4–6; 2 → Tasks 1–2; 3 → Tasks 1–2 (`PROFILES_BY_PROVIDER`, `apply_profile`); 4 → Tasks 4–5 (every hook wrapped, class-name logging, tested with `Boom`); 5 → Task 6 (`try/except ImportError`, no pyproject change).
- §3 hooks: pre-call steps 1–6 → Task 4; post-call, streaming (early-stop not recorded, estimate without usage), failure (idempotent, 3 → revert, P0 ignored) → Task 5; purge every `PURGE_EVERY` → Task 5 `_record`.
- §4 provider kind: `litellm_upstream`, `upstream`, `ENV_KEYS`, bypass header, profiles → Tasks 1–2; prices via `[prices]` only → README (Task 7), code unchanged.
- §5 proxy extractions → Task 3. §6 config/docs → Tasks 1 and 7. §7 tests → every task; live mock test → Task 6.
- Names used consistently: `PithHooks`, `PithGuardrail`, `CustomGuardrail`, `_stash`, `_ready`, `_record`, `choose`, `record`, `PROVIDER`, `CHAT_CALLS`, `LITELLM_KEYS`, `OPTIMIZER_LITELLM_LIVE`.
