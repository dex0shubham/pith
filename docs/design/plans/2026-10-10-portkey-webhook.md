# Plan 5: Portkey Webhook Implementation Plan

**Goal:** Give Portkey gateway users pith's output-profile pins with no gateway code: one webhook endpoint in pith's own app that Portkey's built-in `default.webhook` check calls before and after each request, plus a `portkey` provider kind so the sweep CLI can tune those routes through the gateway.

**Architecture:** `pith/portkey.py` exposes `handle(cfg, conn, payload) -> dict`, a pure function over the webhook payload: on `beforeRequestHook` it runs the shared `proxy.choose` and returns `transformedData.request.json` for a P2/P3 pin; on `afterRequestHook` it strips pith's own shape suffix from the delivered request (`rewrite.strip_shape`) to recover the original, and records usage from `response.json` (non-streaming only). `proxy.create_app` mounts it at `POST /optimizer/portkey` behind an optional bearer token. The `portkey` provider kind reuses the OpenAI replay path with Portkey headers and a bypass flag in `x-portkey-metadata`.

**Tech Stack:** Python 3.12, FastAPI (already a dependency), httpx `MockTransport`/`ASGITransport` in tests, pytest + anyio. The Portkey gateway (`npx @portkey-ai/gateway@1.15.2`, Node) is used only by an opt-in live test.

**Spec:** `docs/design/specs/2026-10-09-portkey-webhook-design.md` (authoritative; §2 holds the spike-verified Portkey facts). Conventions follow the LiteLLM and Headroom adapters (`pith/guardrail.py`, `pith/headroom.py`). Read the spec before starting.

## Global Constraints

- Python ≥ 3.12; code under `pith/`, tests under `tests/`; pytest only; the suite must stay warning-free under `.venv/bin/pytest -q -W error`. Run everything with `.venv/bin/pytest` / `.venv/bin/python`.
- **No new dependency**; `pyproject.toml` unchanged.
- **Never raise from the handler:** every path returns `{"verdict": True}` (plus `transformedData` only for a transformed before-hook); any exception is caught, logged as `type(exc).__name__` only (never exception text), and answered `{"verdict": True}`.
- **Scope:** only `requestType == "chatComplete"` payloads whose `request.json` is a dict with `messages`; everything else returns `{"verdict": True}` untouched. Streams (`response.json` null) are never recorded.
- **Metadata keys** (from Portkey's `x-portkey-metadata`): `pith_bypass` (truthy → untouched, unrecorded), `pith_route` (route name), `pith: "off"` (force P0).
- **Provider kind** `"portkey"`: `Config.portkey_upstream = "http://localhost:8787"`, `Config.webhook_token = ""` (empty = no auth), `Config.portkey_headers: dict[str, str]` from a `[portkey_headers]` toml table; `ENV_KEYS["portkey"] = "PORTKEY_API_KEY"`; `PROFILES_BY_PROVIDER["portkey"] = ("P0", "P2", "P3")`; replays send `x-portkey-api-key`, merge `portkey_headers`, and set `x-portkey-metadata: {"pith_bypass": true}`.
- **Recorded latency is 0** (the webhook has no timing); stored request body is the original (shape suffix stripped); profile recorded is the route's `pinned_profile` when a suffix was present, else P0.
- `webhook_token` is typed `str` (not `str | None`): `load_config` calls the field type on toml/env values, and a union type is not callable.
- Commit messages: plain conventional commits (`feat:`, `docs:`, `test:`), no trailers, no co-author lines, no tool or model names anywhere in the repo.
- Branch `portkey-webhook` (already created from main after PR #9).

## File structure

| File | Responsibility |
|---|---|
| `pith/config.py` (modify) | `portkey_upstream`, `webhook_token`, `portkey_headers` |
| `pith/providers.py` (modify) | `upstream("portkey")` |
| `pith/__main__.py`, `pith/sweep.py` (modify) | key map, profile list |
| `pith/replay.py` (modify) | Portkey auth header, replay headers, bypass metadata |
| `pith/rewrite.py` (modify) | `portkey` in the user-text branch; `strip_shape` |
| `pith/portkey.py` (new) | `handle(cfg, conn, payload)` |
| `pith/proxy.py` (modify) | `POST /optimizer/portkey` route |
| `tests/test_config.py`, `test_providers.py`, `test_main.py`, `test_sweep.py`, `test_replay.py`, `test_rewrite.py` (modify); `tests/test_portkey.py`, `tests/live/test_portkey_mock.py` (new) | tests |
| `pith.example.toml`, `README.md` (modify) | docs |

---

### Task 1: Provider kind `portkey` (config, upstream, keys, profiles, replay)

**Files:**
- Modify: `pith/config.py`, `pith/providers.py`, `pith/__main__.py:12`, `pith/sweep.py:18-19`, `pith/replay.py` (`auth_headers`, `call`), `pith.example.toml`
- Test: `tests/test_config.py`, `tests/test_providers.py`, `tests/test_main.py`, `tests/test_sweep.py`, `tests/test_replay.py`

**Interfaces:**
- Produces: `Config.portkey_upstream: str = "http://localhost:8787"`, `Config.webhook_token: str = ""`, `Config.portkey_headers: dict[str, str]`; `upstream("portkey", cfg)`; `auth_headers("portkey", key) == {"x-portkey-api-key": key}`; `call(client, cfg, "portkey", body, key, prices)` posts to `cfg.portkey_upstream + "/v1/chat/completions"` with `cfg.portkey_headers` merged and `x-portkey-metadata` set to `{"pith_bypass": true}`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, inside `test_defaults_match_spec` after the `assert c.routes == {}` line, add:

```python
    assert c.portkey_upstream == "http://localhost:8787" and c.webhook_token == "" and c.portkey_headers == {}
```

and append:

```python
def test_portkey_headers_table_token_and_env_override(tmp_path):
    p = tmp_path / "pith.toml"
    p.write_text('webhook_token = "t0k"\n[portkey_headers]\n"x-portkey-provider" = "openai"\n"x-portkey-config" = "pc-1"\n')
    c = load_config(str(p), env={"OPTIMIZER_PORTKEY_UPSTREAM": "http://gw:8787"})
    assert c.webhook_token == "t0k" and c.portkey_upstream == "http://gw:8787"
    assert c.portkey_headers == {"x-portkey-provider": "openai", "x-portkey-config": "pc-1"}
    assert load_config(None, env={"OPTIMIZER_WEBHOOK_TOKEN": "env"}).webhook_token == "env"
```

In `tests/test_providers.py` replace `test_upstream_from_config` with:

```python
def test_upstream_from_config():
    c = Config(anthropic_upstream="http://a", openai_upstream="http://o", litellm_upstream="http://l", portkey_upstream="http://p")
    assert upstream("anthropic", c) == "http://a" and upstream("openai", c) == "http://o"
    assert upstream("litellm", c) == "http://l" and upstream("portkey", c) == "http://p"
    assert upstream("unknown", c) == "http://o"
```

In `tests/test_main.py` replace `test_keys_from_env` with:

```python
def test_keys_from_env():
    assert keys_from_env({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "LITELLM_API_KEY": "l", "PORTKEY_API_KEY": "p", "X": "1"}) == \
        {"anthropic": "a", "openai": "o", "litellm": "l", "portkey": "p"}
    assert keys_from_env({"OPENAI_API_KEY": ""}) == {}
```

In `tests/test_sweep.py` replace `test_profiles_constant` with:

```python
def test_profiles_constant():
    assert PROFILES_BY_PROVIDER == {"anthropic": ("P0", "P1", "P2", "P3", "P4"),
                                    "openai": ("P0", "P1", "P1b", "P2", "P3", "P4"),
                                    "litellm": ("P0", "P2", "P3"), "portkey": ("P0", "P2", "P3")}
```

In `tests/test_replay.py`, add to `test_endpoint_and_headers`:

```python
    assert endpoint_for("portkey", {"messages": []}) == "/v1/chat/completions"
    assert auth_headers("portkey", "k") == {"x-portkey-api-key": "k"}
```

and append:

```python
def test_call_through_portkey_adds_bypass_metadata_and_config_headers():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json=CHAT)
    client = httpx.Client(transport=httpx.MockTransport(h))
    cfg = Config(portkey_upstream="http://gw:8787", portkey_headers={"x-portkey-provider": "openai"})
    r = call(client, cfg, "portkey", {"model": "mock", "messages": [], "stream": True}, "k", {"mock": (1.0, 2.0)})
    assert r.status == 200 and r.text == "hi" and r.cost_usd == (5 * 1.0 + 1 * 2.0) / 1e6
    assert str(seen[0].url) == "http://gw:8787/v1/chat/completions" and "stream" not in json.loads(seen[0].content)
    assert seen[0].headers["x-portkey-api-key"] == "k" and seen[0].headers["x-portkey-provider"] == "openai"
    assert json.loads(seen[0].headers["x-portkey-metadata"]) == {"pith_bypass": True}
    assert "authorization" not in seen[0].headers and "x-optimizer" not in seen[0].headers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py tests/test_providers.py tests/test_main.py::test_keys_from_env tests/test_sweep.py::test_profiles_constant tests/test_replay.py -q`
Expected: failures on the new fields (`TypeError`/`AttributeError`/`AssertionError`/`KeyError`).

- [ ] **Step 3: Implement**

`pith/config.py` — in `Config`, after `litellm_upstream` add:

```python
    portkey_upstream: str = "http://localhost:8787"  # a Portkey gateway; sweeps replay routes recorded by the pith webhook
    webhook_token: str = ""  # when set, POST /optimizer/portkey requires "Authorization: Bearer <token>"
```

after `prices` add:

```python
    portkey_headers: dict[str, str] = field(default_factory=dict)  # extra headers on replays through Portkey
```

In `load_config`, change the scalar exclusion to `if f.name not in ("routes", "prices", "portkey_headers")` and, before `return cfg`, add:

```python
    cfg.portkey_headers = {str(k): str(v) for k, v in (data.get("portkey_headers") or {}).items()}
```

`pith/providers.py` — replace `upstream`:

```python
def upstream(provider: str, config: Config) -> str:
    return {"anthropic": config.anthropic_upstream, "litellm": config.litellm_upstream,
            "portkey": config.portkey_upstream}.get(provider, config.openai_upstream)
```

`pith/__main__.py:12`:

```python
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "litellm": "LITELLM_API_KEY",
            "portkey": "PORTKEY_API_KEY"}
```

`pith/sweep.py` — `PROFILES_BY_PROVIDER` gains `"portkey": ("P0", "P2", "P3")` (same comment as litellm: only user-text shape through a gateway hook).

`pith/replay.py` — replace `auth_headers`:

```python
def auth_headers(provider: str, key: str) -> dict:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if provider == "portkey":
        return {"x-portkey-api-key": key}
    return {"authorization": f"Bearer {key}"}
```

and in `call`, after the `litellm` bypass line add:

```python
    if provider == "portkey":  # the pith webhook behind Portkey must neither rewrite nor record replays
        headers.update(cfg.portkey_headers)
        headers["x-portkey-metadata"] = json.dumps({"pith_bypass": True})
```

`pith.example.toml` — after the `litellm_upstream` line add:

```toml
portkey_upstream = "http://localhost:8787"   # only used by sweeps of routes recorded by the Portkey webhook
webhook_token = ""                           # set to require "Authorization: Bearer <token>" on POST /optimizer/portkey
```

and at the end:

```toml
[portkey_headers]           # sent on every replay through Portkey (a saved config id, provider, virtual key, ...)
"x-portkey-provider" = "openai"
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pith/config.py pith/providers.py pith/__main__.py pith/sweep.py pith/replay.py pith.example.toml tests/test_config.py tests/test_providers.py tests/test_main.py tests/test_sweep.py tests/test_replay.py
git commit -m "feat: portkey provider kind for config, replays, keys and profiles"
```

---

### Task 2: `apply_profile("portkey")` and `strip_shape`

**Files:**
- Modify: `pith/rewrite.py` (`apply_profile` branch condition; add `SHAPE_PREFIX`, `strip_shape`)
- Test: `tests/test_rewrite.py`

**Interfaces:**
- Produces: `apply_profile("portkey", body, state)` behaves exactly as the `litellm` branch (P2/P3 → user-text append, else an equal copy); `SHAPE_PREFIX: str` (the shape text up to `{n}`); `strip_shape(body: dict) -> tuple[dict, bool]` returning a deep copy with pith's shape text part removed from the last user message (a lone remaining plain text part becomes string content again) and `True`, or an untouched copy and `False`.

- [ ] **Step 1: Write the failing tests**

Add `strip_shape` to the `pith.rewrite` import line of `tests/test_rewrite.py` and append:

```python
def test_portkey_profiles_match_the_litellm_user_text_shape():
    out = apply_profile("portkey", CHAT, RouteState("P2", target_words=30))
    assert out["messages"][-1]["content"][-1] == {"type": "text", "text": SHAPE_TEXT.format(n=30)}
    assert out["messages"][:-1] == CHAT["messages"][:-1] and untouched(CHAT, out)
    for p in ("P1", "P1b", "P4"):
        assert apply_profile("portkey", CHAT, RouteState(p)) == CHAT


def test_strip_shape_round_trips_append_shape():
    shaped = apply_profile("portkey", CHAT, RouteState("P3", target_words=20, exemplar="Yes."))
    back, stripped = strip_shape(shaped)
    assert stripped and back == CHAT and shaped["messages"][-1]["content"][-1]["text"].startswith("Answer directly.")
    body = dict(CHAT, messages=[{"role": "user", "content": [{"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}}]}])
    back, stripped = strip_shape(apply_profile("portkey", body, RouteState("P2")))
    assert stripped and back == body  # a part carrying cache_control is never flattened back to a string
    back, stripped = strip_shape(CHAT)
    assert not stripped and back == CHAT and back is not CHAT
    plain = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Answer directly please"}]}]}
    assert strip_shape(plain) == (plain, False)  # only pith's exact prefix counts
    assert strip_shape({"messages": [{"role": "assistant", "content": "a"}]})[1] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rewrite.py -q`
Expected: `ImportError: cannot import name 'strip_shape'`.

- [ ] **Step 3: Implement**

In `pith/rewrite.py`, after `EXEMPLAR_PREFIX` add:

```python
SHAPE_PREFIX = SHAPE_TEXT.split("{n}")[0]  # how strip_shape recognises pith's own appended text part
```

Change the `apply_profile` branch `if provider == "litellm":` to `if provider in ("litellm", "portkey"):` (comment: "a gateway hook cannot retry a rejected request and folds system messages: only the user-text shape").

After `append_shape` add:

```python
def strip_shape(body: dict) -> tuple[dict, bool]:
    """Undo append_shape on a deep copy: (body, True) when the last user message ends with pith's shape text part
    (removed; a lone remaining plain text part becomes string content again), else (copy, False)."""
    out = copy.deepcopy(body)
    for m in reversed(out.get("messages") or []):
        if m.get("role") != "user":
            continue
        parts = m.get("content")
        if isinstance(parts, list) and parts and isinstance(parts[-1], dict) and parts[-1].get("type") == "text" \
                and str(parts[-1].get("text", "")).startswith(SHAPE_PREFIX):
            rest = parts[:-1]
            m["content"] = rest[0]["text"] if len(rest) == 1 and set(rest[0]) == {"type", "text"} else rest
            return out, True
        return out, False
    return out, False
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pith/rewrite.py tests/test_rewrite.py
git commit -m "feat: portkey shape profile and strip_shape to recover the original request"
```

---

### Task 3: `pith/portkey.py` and the webhook route

**Files:**
- Create: `pith/portkey.py`
- Modify: `pith/proxy.py` (`create_app`: new route before the catch-all)
- Test: `tests/test_portkey.py`

**Interfaces:**
- Consumes: `proxy.choose`, `proxy.record`, `proxy.PURGE_EVERY`, `rewrite.strip_shape` (Task 2), `usage.usage_from_body`, `Config.webhook_token` (Task 1).
- Produces: `portkey.handle(cfg: Config, conn, payload) -> dict`; `POST /optimizer/portkey` on the pith app (401 when `webhook_token` is set and the bearer does not match; `{"verdict": true}` for malformed JSON).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_portkey.py`:

```python
import json
import logging

import httpx
import pytest

from pith import db
from pith.config import Config
from pith.portkey import handle
from pith.proxy import create_app

BODY = {"model": "gpt-4o", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], "max_tokens": 50}
RESP = {"id": "c", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5}}


def payload(event, body=BODY, resp=None, meta=None, request_type="chatComplete", provider="openai"):
    """What Portkey's default.webhook POSTs (spec §2)."""
    after = event == "afterRequestHook"
    return {"eventType": event, "provider": provider, "requestType": request_type, "metadata": meta or {},
            "request": {"json": body, "text": "q", "isStreamingRequest": bool(body.get("stream")), "isTransformed": False},
            "response": {"json": resp if after else {}, "text": "", "statusCode": 200 if after else None, "isTransformed": False}}


def setup():
    return Config(sample_rate=1.0), db.connect(":memory:")


class Boom:
    def execute(self, *a, **k):
        raise RuntimeError("x-portkey-api-key: sk-secret")


def test_before_hook_registers_the_route_and_leaves_p0_untouched():
    cfg, conn = setup()
    assert handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"})) == {"verdict": True}
    route = db.get_route(conn, "r")
    assert route["provider"] == "portkey" and route["model"] == "gpt-4o"
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    key = handle(cfg, conn, payload("beforeRequestHook")) and conn.execute("SELECT key FROM routes WHERE key != 'r'").fetchone()[0]
    assert key.startswith("portkey:gpt-4o:")


def test_before_hook_transforms_a_pinned_route_and_honours_off_and_bypass():
    cfg, conn = setup()
    handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"}))
    db.set_pin(conn, "r", "P2")
    out = handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"}))
    new = out["transformedData"]["request"]["json"]
    assert out["verdict"] is True and new["messages"][-1]["content"][0] == {"type": "text", "text": "q"}
    assert new["messages"][-1]["content"][1]["text"].startswith("Answer directly.")
    assert new["messages"][0] == BODY["messages"][0] and new["max_tokens"] == 50 and new["model"] == "gpt-4o"
    assert handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r", "pith": "off"})) == {"verdict": True}
    assert handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r", "pith_bypass": True})) == {"verdict": True}


def test_non_chat_and_malformed_payloads_are_untouched():
    cfg, conn = setup()
    assert handle(cfg, conn, payload("beforeRequestHook", request_type="messages")) == {"verdict": True}
    assert handle(cfg, conn, payload("beforeRequestHook", body={"input": "q"})) == {"verdict": True}
    assert handle(cfg, conn, None) == {"verdict": True}
    assert handle(cfg, conn, {"eventType": "beforeRequestHook", "requestType": "chatComplete"}) == {"verdict": True}
    assert conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


def test_after_hook_records_p0_and_strips_the_shape_for_a_pinned_route():
    cfg, conn = setup()
    assert handle(cfg, conn, payload("afterRequestHook", resp=RESP, meta={"pith_route": "r"})) == {"verdict": True}
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["route_key"], row["profile"], row["input_tokens"], row["output_tokens"], row["stop_reason"], row["latency_ms"]) == \
        ("r", "P0", 12, 5, "stop", 0)
    assert json.loads(conn.execute("SELECT request_json FROM bodies").fetchone()[0]) == BODY
    db.set_pin(conn, "r", "P2")
    shaped = handle(cfg, conn, payload("beforeRequestHook", meta={"pith_route": "r"}))["transformedData"]["request"]["json"]
    handle(cfg, conn, payload("afterRequestHook", body=shaped, resp=RESP, meta={"pith_route": "r"}))
    row = conn.execute("SELECT * FROM requests ORDER BY id DESC").fetchone()
    assert row["profile"] == "P2"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == BODY
    assert json.loads(conn.execute("SELECT response_json FROM bodies ORDER BY id DESC").fetchone()[0]) == RESP
    assert db.get_route(conn, "r")["pinned_profile"] == "P2"


def test_after_hook_ignores_streams_bypass_and_failures(caplog):
    cfg, conn = setup()
    assert handle(cfg, conn, payload("afterRequestHook", body=dict(BODY, stream=True), resp=None)) == {"verdict": True}
    assert handle(cfg, conn, payload("afterRequestHook", resp=RESP, meta={"pith_bypass": True})) == {"verdict": True}
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    with caplog.at_level(logging.WARNING, logger="pith.portkey"):
        assert handle(cfg, Boom(), payload("afterRequestHook", resp=RESP)) == {"verdict": True}
        assert handle(cfg, Boom(), payload("beforeRequestHook")) == {"verdict": True}
    assert "sk-secret" not in caplog.text and "RuntimeError" in caplog.text


def make_app(config):
    return create_app(config, db.connect(":memory:"),
                      client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))))


async def post(app, body, headers=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.post("/optimizer/portkey", content=body, headers={"content-type": "application/json", **(headers or {})})


@pytest.mark.anyio
async def test_route_answers_hooks_requires_the_token_when_set_and_tolerates_bad_json():
    app = make_app(Config(sample_rate=1.0))
    r = await post(app, json.dumps(payload("beforeRequestHook", meta={"pith_route": "r"})))
    assert r.status_code == 200 and r.json() == {"verdict": True}
    r = await post(app, b"{not json")
    assert r.status_code == 200 and r.json() == {"verdict": True}
    app = make_app(Config(webhook_token="t0k"))
    assert (await post(app, json.dumps(payload("beforeRequestHook")))).status_code == 401
    assert (await post(app, json.dumps(payload("beforeRequestHook")), {"authorization": "Bearer nope"})).status_code == 401
    assert (await post(app, json.dumps(payload("beforeRequestHook")), {"authorization": "Bearer t0k"})).status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_portkey.py -q`
Expected: `ModuleNotFoundError: No module named 'pith.portkey'`.

- [ ] **Step 3: Implement**

Create `pith/portkey.py`:

```python
"""Portkey webhook: pith's data plane behind Portkey's built-in `default.webhook` check.
Spec: docs/design/specs/2026-10-09-portkey-webhook-design.md.

Portkey POSTs the hook context here before and after each request. The handler never raises: a pith failure answers
{"verdict": true} with no transform, so Portkey forwards the customer's request unchanged.
"""
import json
import logging
import time

from pith import db
from pith.proxy import PURGE_EVERY, choose, record
from pith.rewrite import strip_shape
from pith.usage import usage_from_body

log = logging.getLogger("pith.portkey")
PROVIDER = "portkey"
_counter = {"n": 0}


def handle(cfg, conn, payload) -> dict:
    try:
        if not isinstance(payload, dict) or payload.get("requestType") != "chatComplete":
            return {"verdict": True}
        req = (payload.get("request") or {}).get("json")
        if not isinstance(req, dict) or "messages" not in req:
            return {"verdict": True}
        meta = payload.get("metadata") or {}
        if meta.get("pith_bypass"):
            return {"verdict": True}
        mode = "off" if str(meta.get("pith", "")).lower() == "off" else ""
        route_name = meta.get("pith_route") or None
        event = payload.get("eventType")
        if event == "beforeRequestHook":
            fp, route, profile, body = choose(cfg, conn, PROVIDER, req, mode, route_name)
            if profile == "P0":
                return {"verdict": True}
            return {"verdict": True, "transformedData": {"request": {"json": body}}}
        if event == "afterRequestHook":
            resp = (payload.get("response") or {}).get("json")
            if not isinstance(resp, dict) or not resp:
                return {"verdict": True}  # streams deliver null here; nothing to record
            original, stripped = strip_shape(req)
            fp, route, _, _ = choose(cfg, conn, PROVIDER, original, "off", route_name)  # register/refresh, never rewrite
            profile = route["pinned_profile"] if stripped else "P0"
            _counter["n"] += 1
            if _counter["n"] % PURGE_EVERY == 0:
                db.purge_expired(conn, time.time())
            record(cfg, conn, fp.key, profile, usage_from_body("openai", resp), 0, json.dumps(original), json.dumps(resp))
        return {"verdict": True}
    except Exception as exc:  # never raise into Portkey; never log exc text
        log.warning("portkey hook failed (%s); request left unchanged", type(exc).__name__)
        return {"verdict": True}
```

In `pith/proxy.py`, inside `create_app`, immediately before the `@app.api_route("/{path:path}", ...)` catch-all, add:

```python
    from pith.portkey import handle as portkey_handle  # function-level: pith.portkey imports this module

    @app.post("/optimizer/portkey")
    async def portkey_hook(request: Request):
        if config.webhook_token and request.headers.get("authorization") != f"Bearer {config.webhook_token}":
            return Response(status_code=401)
        try:
            payload = await request.json()
        except Exception:  # malformed JSON: nothing to decide on
            payload = None
        return portkey_handle(config, conn, payload)
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass (the proxy's existing tests are unaffected: the new route is registered before the catch-all and only matches `POST /optimizer/portkey`).

- [ ] **Step 5: Commit**

```bash
git add pith/portkey.py pith/proxy.py tests/test_portkey.py
git commit -m "feat: Portkey webhook endpoint applies pins before the call and records after it"
```

---

### Task 4: README, live test against the open-source gateway

**Files:**
- Modify: `README.md` (new section after "## Headroom plugin", before "## Sweeps: turning observation into pins"; one line in "## Tests")
- Create: `tests/live/test_portkey_mock.py`

- [ ] **Step 1: Create the live test**

Create `tests/live/test_portkey_mock.py`:

```python
"""Runs the open-source Portkey gateway with default.webhook hooks pointing at pith, against a mock upstream.

Run: OPTIMIZER_PORTKEY_LIVE=1 .venv/bin/pytest tests/live/test_portkey_mock.py -v
Needs Node (`npx`) and a free port 8787: the gateway's CLI always binds 8787 and ignores PORT/--port. Skipped otherwise.
No API key and no network beyond the npm download of @portkey-ai/gateway.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from pith import db
from pith.config import Config
from pith.proxy import create_app

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_PORTKEY_LIVE") != "1" or shutil.which("npx") is None,
                                reason="set OPTIMIZER_PORTKEY_LIVE=1 and install Node")
GATEWAY = "@portkey-ai/gateway@1.15.2"
GATEWAY_PORT = 8787
SEEN = []


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append((self.path, body))
        out = {"id": "c", "object": "chat.completion", "model": body["model"],
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 12, "completion_tokens": 5}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_busy(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def stack(tmp_path):
    if port_busy(GATEWAY_PORT):
        pytest.skip(f"port {GATEWAY_PORT} is busy and the Portkey gateway CLI cannot be moved")
    import uvicorn

    up = free_port()
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", up), Upstream).serve_forever, daemon=True).start()
    db_path = str(tmp_path / "pith.db")
    pith_port = free_port()
    app = create_app(Config(db_path=db_path, sample_rate=1.0, listen=f"127.0.0.1:{pith_port}"), db.connect(db_path))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=pith_port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    gw = subprocess.Popen(["npx", "-y", GATEWAY], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(240):  # the first run downloads the package; allow two minutes
            try:
                if httpx.get(f"http://127.0.0.1:{GATEWAY_PORT}/", timeout=1).status_code < 500:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("portkey gateway did not start")
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{pith_port}/optimizer/health", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        hook = f"http://127.0.0.1:{pith_port}/optimizer/portkey"
        config = {"provider": "openai", "api_key": "sk-fake", "custom_host": f"http://127.0.0.1:{up}",
                  "before_request_hooks": [{"type": "mutator", "id": "pith-before",
                                            "checks": [{"id": "default.webhook", "parameters": {"webhookURL": hook}}]}],
                  "after_request_hooks": [{"type": "guardrail", "id": "pith-after", "deny": False,
                                           "checks": [{"id": "default.webhook", "parameters": {"webhookURL": hook}}]}]}
        yield f"http://127.0.0.1:{GATEWAY_PORT}", json.dumps(config), db_path
    finally:
        os.killpg(os.getpgid(gw.pid), signal.SIGTERM)
        try:
            gw.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(gw.pid), signal.SIGKILL)
        server.should_exit = True


def test_portkey_hooks_record_and_apply_pins(stack):
    base, config, db_path = stack
    body = {"model": "gpt-4o", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Where is my order?"}], "max_tokens": 50}
    hdrs = {"x-portkey-config": config, "x-portkey-metadata": json.dumps({"pith_route": "pk"}), "content-type": "application/json"}
    r = httpx.post(base + "/v1/chat/completions", json=body, headers=hdrs, timeout=30)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Hello", r.text
    assert r.json()["hook_results"]["before_request_hooks"][0]["verdict"] is True
    time.sleep(0.5)
    conn = db.connect(db_path)
    assert db.get_route(conn, "pk")["provider"] == "portkey"
    row = conn.execute("SELECT profile, input_tokens, output_tokens, stop_reason FROM requests").fetchone()
    assert tuple(row) == ("P0", 12, 5, "stop")
    assert SEEN[-1][1]["messages"][-1]["content"] == "Where is my order?"
    db.set_pin(conn, "pk", "P2")
    conn.close()
    r = httpx.post(base + "/v1/chat/completions", json=body, headers=hdrs, timeout=30)
    assert r.status_code == 200 and r.json()["hook_results"]["before_request_hooks"][0]["transformed"] is True
    time.sleep(0.5)
    sent = SEEN[-1][1]["messages"][-1]["content"]
    assert sent[0] == {"type": "text", "text": "Where is my order?"} and sent[1]["text"].startswith("Answer directly.")
    conn = db.connect(db_path)
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == body
```

- [ ] **Step 2: Insert the README section**

Immediately before the line `## Sweeps: turning observation into pins` insert:

````markdown
## Portkey plugin

Behind a [Portkey](https://github.com/Portkey-AI/gateway) gateway, pith needs no gateway code: Portkey's built-in
`default.webhook` check calls pith before and after each request. Run `pith serve` where the gateway can reach it and
add two hooks to the Portkey config (`x-portkey-config` header or a saved config):

    {
      "before_request_hooks": [{"type": "mutator", "id": "pith-before",
        "checks": [{"id": "default.webhook", "parameters": {"webhookURL": "http://pith:8787/optimizer/portkey",
                                                            "headers": {"authorization": "Bearer <webhook_token>"}}}]}],
      "after_request_hooks":  [{"type": "guardrail", "id": "pith-after", "deny": false,
        "checks": [{"id": "default.webhook", "parameters": {"webhookURL": "http://pith:8787/optimizer/portkey",
                                                            "headers": {"authorization": "Bearer <webhook_token>"}}}]}]
    }

The before hook fingerprints the request and, for a pinned route, returns it with the shape text appended to the last
user message; the after hook records usage into the same SQLite the CLI reads. `x-portkey-metadata` keys: `pith_route`
(name the route), `pith_bypass` (skip pith entirely), `pith: "off"` (force P0). Set `webhook_token` in `pith.toml`
when the endpoint is reachable beyond the gateway. Limits: only OpenAI-format (`chatComplete`) requests are handled;
streaming responses are seen but not recorded (Portkey delivers no body for them); provider rejections are invisible
to after hooks, so `recheck` is the drift guard; Portkey appends `hook_results` to responses whenever hooks run. Only
P2/P3 apply through Portkey. Note the port clash: Portkey's gateway and pith both default to 8787, so move one.

Sweep those routes through the gateway: set `portkey_upstream`, put a `[prices."<model>"]` entry for each model name
the clients send, add any routing headers under `[portkey_headers]` (a saved config id, provider, virtual key), export
`PORTKEY_API_KEY`, and run `python -m pith sweep`. Replays and the judge carry `x-portkey-metadata: {"pith_bypass": true}`
so the webhook ignores them; `judge_provider = "portkey"` runs the judge through the gateway too.

````

In "## Tests", after the Headroom live-test paragraph, append:

```markdown
`OPTIMIZER_PORTKEY_LIVE=1 .venv/bin/pytest tests/live/test_portkey_mock.py` runs the open-source Portkey gateway via
`npx` (needs Node and a free port 8787) with the webhook hooks against a mock upstream; no API key.
```

- [ ] **Step 3: Run the suite, then the live test**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass; `tests/live/test_portkey_mock.py` skipped.

Then (Node 22 is installed; the first run downloads the gateway package):

```bash
OPTIMIZER_PORTKEY_LIVE=1 .venv/bin/pytest tests/live/test_portkey_mock.py -v
```

Expected: 1 passed. If the gateway never answers on 8787, check `lsof -i :8787` for a leftover process. Afterwards confirm `lsof -i :8787` shows nothing (the fixture kills the gateway's process group).

- [ ] **Step 4: Commit**

```bash
git add README.md tests/live/test_portkey_mock.py
git commit -m "docs: Portkey plugin section and live gateway test"
```

---

## Self-review against the spec

- §1 decisions: 1 (endpoint, config-only wiring) → Tasks 3–4; 2 (chatComplete only) → Task 3; 3 (shape-only, P0/P2/P3, no revert) → Tasks 1–3; 4 (streams unrecorded) → Task 3; 5 (never raise) → Task 3 (`Boom` tests).
- §2 facts: payload shape → the `payload()` test helper and the live test; `transformedData.request.json` → Task 3; `hook_results` → README and live test; after hook on 200 only, `response.json` null for streams → Task 3.
- §3 handler contract → Task 3 (`handle`, route, 401, bad JSON). §4 rewrite → Task 2. §5 provider kind → Task 1 (`webhook_token` typed `str`, empty = unset, because `load_config` calls the field type). §6 docs → Task 4. §7 tests → every task; live test → Task 4.
- Names consistent: `handle`, `PROVIDER`, `strip_shape`, `SHAPE_PREFIX`, `portkey_upstream`, `webhook_token`, `portkey_headers`, `OPTIMIZER_PORTKEY_LIVE`.
