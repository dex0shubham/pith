# Plan 4: Headroom Adapter Implementation Plan

**Goal:** Run pith's data plane inside a Headroom proxy: a middleware that fingerprints, decides and records on the raw request and response, plus a pipeline extension that appends the pinned shape text after Headroom's compression, so Headroom users get output-profile pins without a second hop.

**Architecture:** One new module `pith/headroom.py` with two Headroom entry points: `install(app, config)` (group `headroom.proxy_extension`) adds `PithMiddleware`, a pure-ASGI middleware that buffers the body, calls the shared `proxy.choose`, rewrites effort params in the raw body for P1/P1b/P4, publishes the decision in a `ContextVar`, tees the response bytes and records usage with pith's own parsers; `PithPipeline` (group `headroom.pipeline_extension`) reads that decision at `PRE_SEND` and appends the shape as user text for P2/P3/P4. `rewrite.apply_profile` is split into `apply_effort` and `append_shape` so the halves can be used separately. Routes are ordinary `anthropic`/`openai` routes swept with the existing CLI.

**Tech Stack:** Python 3.12, stdlib `contextvars`/`json`/`sqlite3`, pytest + anyio (dev deps). Headroom (`headroom-ai` on PyPI) is never a dependency and never imported by pith; it is installed into `.venv` only for the opt-in live test.

**Spec:** `docs/design/specs/2026-10-09-headroom-adapter-design.md` (authoritative; §2 holds the spike-verified Headroom facts). Conventions follow `docs/design/specs/2026-10-09-litellm-plugin-design.md` and `pith/guardrail.py`. Read the spec before starting.

## Global Constraints

- Python ≥ 3.12; code under `pith/`, tests under `tests/`; pytest only; the suite must stay warning-free under `.venv/bin/pytest -q -W error`. Run everything with `.venv/bin/pytest` / `.venv/bin/python`.
- **No new dependency.** `pith/headroom.py` must import with Headroom absent; `pyproject.toml` changes only by the two entry-point tables.
- **Fail open.** Every pith step in the middleware and the pipeline extension is wrapped; on failure the original request is forwarded (or recording is skipped) and the log carries `type(exc).__name__` only, never exception text. An exception raised by the downstream (Headroom) app propagates unchanged.
- **Middleware scope:** `POST` to `/v1/messages` (provider `anthropic`) or `/v1/chat/completions` (provider `openai`) only; everything else passes through untouched and unrecorded.
- **Profiles:** P1/P1b/P4 rewrite params in the raw body (`apply_effort`); P2/P3/P4 append the shape as user text at `PRE_SEND` (`append_shape`); a shape profile whose `PRE_SEND` never ran is recorded as P0; the middleware never appends the shape itself.
- **Recording:** usage from the teed response via `usage_from_body` when the bytes parse as a JSON object, else `StreamUsage.result()`; request body stored is the client's original; the route's `injection_form` is set to `user_text`; 400/422 after a rewrite bumps rejections and reverts at 3 exactly as the proxy; only `status < 400` records.
- **Headers:** `X-Optimizer: off|bypass` and `X-Optimizer-Route` from the inbound ASGI headers, lower-cased.
- **Context variable:** module-level `DECISION: ContextVar` holding `{"route", "profile", "applied", "body", "t0", "target_words", "exemplar"}` or `None`.
- Commit messages: plain conventional commits (`feat:`, `test:`, `refactor:`, `docs:`), no trailers, no co-author lines, no tool or model names anywhere in the repo.
- Branch `headroom-adapter` (already exists with the spec committed).

## File structure

| File | Responsibility |
|---|---|
| `pith/rewrite.py` (modify) | `apply_effort`, `append_shape`; `apply_profile` composes them |
| `pith/headroom.py` (new) | `Runtime`, `DECISION`, `PithMiddleware`, `PithPipeline`, `install` |
| `pyproject.toml` (modify) | two entry-point tables |
| `tests/test_rewrite.py` (modify), `tests/test_headroom.py` (new), `tests/live/test_headroom_mock.py` (new) | tests |
| `README.md` (modify) | "Headroom plugin" section, Tests line |

---

### Task 1: Split `apply_profile` into `apply_effort` and `append_shape`

**Files:**
- Modify: `pith/rewrite.py` (`_anthropic_shape`, `apply_profile`; add two functions)
- Test: `tests/test_rewrite.py`

**Interfaces:**
- Produces: `apply_effort(provider: str, body: dict, profile: str, responses_api: bool = False) -> dict` (deep copy with the P1/P4 effort step-down and the P1b verbosity flag applied; every other profile returns an equal copy); `append_shape(messages: list[dict], state: RouteState) -> None` (in place: appends `{"type": "text", "text": shape}` to the last user message, converting string content to a text-part list; no user message → no-op). `apply_profile` keeps its signature and every existing behaviour.

- [ ] **Step 1: Write the failing tests**

Change the import line of `tests/test_rewrite.py` to:

```python
from pith.rewrite import PROFILES, SHAPE_TEXT, RouteState, append_shape, apply_effort, apply_profile, is_system_role_rejection
```

Append:

```python
def test_apply_effort_is_the_parameter_half():
    assert apply_effort("anthropic", ANTH, "P1")["output_config"]["effort"] == "low"
    assert apply_effort("anthropic", ANTH, "P2") == ANTH and apply_effort("anthropic", ANTH, "P2") is not ANTH
    out = apply_effort("openai", CHAT, "P4")
    assert out["reasoning_effort"] == "low" and out["messages"] == CHAT["messages"]  # no shape in the parameter half
    assert apply_effort("openai", CHAT, "P1b")["verbosity"] == "low"
    out = apply_effort("openai", RESP, "P1", responses_api=True)
    assert out["reasoning"] == {"effort": "low"} and untouched(RESP, out)
    assert apply_effort("openai", CHAT, "P0") == CHAT


def test_append_shape_targets_the_last_user_message_in_place():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    append_shape(msgs, RouteState("P2", target_words=30))
    assert msgs[0]["content"] == [{"type": "text", "text": "q"}, {"type": "text", "text": SHAPE_TEXT.format(n=30)}]
    assert msgs[1] == {"role": "assistant", "content": "a"}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}}]}]
    append_shape(msgs, RouteState("P3", target_words=20, exemplar="Yes."))
    assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert msgs[0]["content"][1]["text"] == SHAPE_TEXT.format(n=20) + "\n\nExample of the expected length:\nYes."
    empty = []
    append_shape(empty, RouteState("P2"))
    assert empty == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rewrite.py -q`
Expected: `ImportError: cannot import name 'append_shape'`.

- [ ] **Step 3: Implement**

In `pith/rewrite.py`, replace `_anthropic_shape` and `apply_profile` with:

```python
def append_shape(messages: list, state: RouteState) -> None:
    """In place: the shape text as a text part of the last user message (the cache-safe user-text form)."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            _append_user_text(m, _shape(state))
            return


def _anthropic_shape(body: dict, state: RouteState) -> None:
    msgs = body.setdefault("messages", [])
    if state.injection_form == "system" and msgs and msgs[-1].get("role") == "user":
        msgs.append({"role": "system", "content": _shape(state)})
        return
    append_shape(msgs, state)


def apply_effort(provider: str, body: dict, profile: str, responses_api: bool = False) -> dict:
    """The parameter half of a profile: effort step-down (P1, P4) and the OpenAI verbosity flag (P1b). Deep copy."""
    out = copy.deepcopy(body)
    if provider == "anthropic":
        if profile in ("P1", "P4"):
            _anthropic_effort(out)
        return out
    if profile in ("P1", "P4"):
        if responses_api:
            r = out.get("reasoning") or {}
            out["reasoning"] = dict(r, effort=_step_down(r.get("effort", "medium")))
        else:
            out["reasoning_effort"] = _step_down(out.get("reasoning_effort", "medium"))
    if profile == "P1b":
        if responses_api:
            out["text"] = dict(out.get("text") or {}, verbosity="low")
        else:
            out["verbosity"] = "low"
    return out


def apply_profile(provider: str, body: dict, state: RouteState, responses_api: bool = False) -> dict:
    p = state.profile
    if p == "P0" or p not in PROFILES:
        return copy.deepcopy(body)
    if provider == "litellm":
        # LiteLLM folds system/developer messages into the provider's system prompt (cache-breaking) and a guardrail
        # cannot retry a rejected request: only the user-text shape, nothing else.
        out = copy.deepcopy(body)
        if p in ("P2", "P3"):
            append_shape(out.setdefault("messages", []), state)
        return out
    out = apply_effort(provider, body, p, responses_api)
    if p in ("P2", "P3", "P4"):
        if provider == "anthropic":
            _anthropic_shape(out, state)
        else:
            _openai_shape(out, state, responses_api)
    return out
```

(`_openai_shape`, `_anthropic_effort`, `_step_down`, `_append_user_text`, `_shape` stay as they are.)

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass; every pre-existing `tests/test_rewrite.py` test unchanged and green.

- [ ] **Step 5: Commit**

```bash
git add pith/rewrite.py tests/test_rewrite.py
git commit -m "refactor: split apply_profile into apply_effort and append_shape"
```

---

### Task 2: `pith/headroom.py` — `Runtime`, `DECISION`, `PithMiddleware`

**Files:**
- Create: `pith/headroom.py`
- Test: `tests/test_headroom.py`

**Interfaces:**
- Consumes: `proxy.choose(config, conn, provider, body, mode, route_name, responses_api=False) -> (fp, route, profile, body)`, `proxy.record(config, conn, route_key, profile, usage, latency_ms, request_json, response_json)`, `proxy.PURGE_EVERY`, `rewrite.apply_effort` (Task 1), `usage.StreamUsage(provider).feed(bytes)/.result()`, `usage.usage_from_body(provider, dict)`, `db.set_injection_form`, `db.bump_rejection`, `db.set_pin`, `db.purge_expired`, `config.load_config`, `db.connect`.
- Produces: `Runtime(config=None, conn=None, env=None)` with `.ready() -> (Config, conn)` and `.n`; module constant `RUNTIME = Runtime()`; `DECISION: ContextVar`; `PATHS = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai"}`; `PithMiddleware(app, runtime: Runtime | None = None)` (pure ASGI). Task 3 adds `PithPipeline` and `install` to this module and reads `DECISION`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_headroom.py`:

```python
import json
import logging

import pytest

from pith import db
from pith.config import Config
from pith.headroom import DECISION, PithMiddleware, Runtime

ANTH = {"model": "claude-opus-5-5", "max_tokens": 50, "system": "S", "messages": [{"role": "user", "content": "q"}]}
CHAT = {"model": "gpt-5", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]}
ANTH_RESP = {"id": "m", "type": "message", "stop_reason": "end_turn", "content": [{"type": "text", "text": "A"}],
             "usage": {"input_tokens": 9, "output_tokens": 4}}
CHAT_SSE = (b'data: {"choices":[{"delta":{"content":"A"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\ndata: [DONE]\n\n')


class Downstream:
    """Stands in for Headroom's app: records the body it receives, lets a test hook act on it (Headroom's pipeline
    runs between the two), and answers with a canned response in several chunks."""

    def __init__(self, response=b"", status=200, sse=False, raise_exc=None):
        self.response, self.status, self.sse, self.raise_exc = response, status, sse, raise_exc
        self.seen, self.decision_seen, self.hook = [], None, None

    async def __call__(self, scope, receive, send):
        body, more = b"", True
        while more:
            m = await receive()
            body += m.get("body", b"")
            more = m.get("more_body", False)
        parsed = json.loads(body) if body else None
        self.seen.append(parsed)
        self.decision_seen = DECISION.get()
        if self.hook and parsed:
            self.hook(parsed)
        if self.raise_exc:
            raise self.raise_exc
        ctype = b"text/event-stream" if self.sse else b"application/json"
        await send({"type": "http.response.start", "status": self.status, "headers": [(b"content-type", ctype)]})
        for i in range(0, len(self.response), 7):
            await send({"type": "http.response.body", "body": self.response[i:i + 7], "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def call(mw, path, body, headers=None, method="POST"):
    scope = {"type": "http", "method": method, "path": path,
             "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]}
    raw = json.dumps(body).encode()
    inbox = [{"type": "http.request", "body": raw[:5], "more_body": True},
             {"type": "http.request", "body": raw[5:], "more_body": False}]
    delegated = []

    async def receive():
        if inbox:
            return inbox.pop(0)
        delegated.append(1)
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent, delegated


def make(response=json.dumps(ANTH_RESP).encode(), **kw):
    conn = db.connect(":memory:")
    down = Downstream(response, **kw)
    return PithMiddleware(down, Runtime(Config(sample_rate=1.0), conn)), down, conn


def relayed(sent):
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


class Boom:
    def execute(self, *a, **k):
        raise RuntimeError("x-api-key: sk-secret")


@pytest.mark.anyio
async def test_non_matching_requests_pass_through_untouched():
    mw, down, conn = make()
    await call(mw, "/v1/models", {"x": 1}, method="GET")
    await call(mw, "/v1/responses", {"model": "gpt-5", "input": "q"})
    assert down.seen == [{"x": 1}, {"model": "gpt-5", "input": "q"}] and down.decision_seen is None
    assert conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


@pytest.mark.anyio
async def test_p0_relays_bytes_and_records_json_and_sse_for_both_providers():
    mw, down, conn = make()
    sent, _ = await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert relayed(sent) == json.dumps(ANTH_RESP).encode() and sent[0]["status"] == 200
    assert down.seen[-1] == ANTH and down.decision_seen["profile"] == "P0" and down.decision_seen["applied"] is True
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["route_key"], row["profile"], row["input_tokens"], row["output_tokens"], row["stop_reason"]) == ("r", "P0", 9, 4, "end_turn")
    assert row["latency_ms"] >= 0 and json.loads(conn.execute("SELECT request_json FROM bodies").fetchone()[0]) == ANTH
    assert db.get_route(conn, "r")["injection_form"] == "user_text" and db.get_route(conn, "r")["provider"] == "anthropic"
    mw, down, conn = make(CHAT_SSE, sse=True)
    sent, _ = await call(mw, "/v1/chat/completions", dict(CHAT, stream=True))
    assert relayed(sent) == CHAT_SSE
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["input_tokens"], row["output_tokens"], row["stop_reason"], row["estimated"]) == (9, 4, "stop", 0)
    assert db.get_route(conn, row["route_key"])["provider"] == "openai"
    assert conn.execute("SELECT response_json FROM bodies").fetchone()[0] == CHAT_SSE.decode()


@pytest.mark.anyio
async def test_bypass_off_and_effort_profiles():
    mw, down, conn = make()
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer": "bypass", "X-Optimizer-Route": "r"})
    assert conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0 and down.decision_seen is None
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    db.set_pin(conn, "r", "P1")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.seen[-1]["output_config"] == {"effort": "low"} and down.seen[-1]["messages"] == ANTH["messages"]
    assert down.decision_seen["profile"] == "P1" and down.decision_seen["applied"] is True
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P1"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == ANTH
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r", "X-Optimizer": "off"})
    assert down.seen[-1] == ANTH
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P0"
    mw, down, conn = make()
    await call(mw, "/v1/chat/completions", CHAT, {"X-Optimizer-Route": "c"})
    db.set_pin(conn, "c", "P1b")
    await call(mw, "/v1/chat/completions", CHAT, {"X-Optimizer-Route": "c"})
    assert down.seen[-1]["verbosity"] == "low" and down.decision_seen["applied"] is True


@pytest.mark.anyio
async def test_shape_pin_is_recorded_as_p0_until_the_pipeline_applies_it():
    mw, down, conn = make()
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    db.set_pin(conn, "r", "P2")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.seen[-1] == ANTH  # the middleware never appends the shape itself
    assert down.decision_seen["profile"] == "P2" and down.decision_seen["applied"] is False
    assert down.decision_seen["target_words"] == 20 and down.decision_seen["exemplar"] is None
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P0"


@pytest.mark.anyio
async def test_rejections_after_a_rewrite_revert_at_three_and_4xx_at_p0_records_nothing():
    mw, down, conn = make(b'{"error":"bad"}', status=400)
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    db.set_pin(conn, "r", "P1")
    for _ in range(3):
        await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    r = db.get_route(conn, "r")
    assert r["pinned_profile"] == "P0" and r["status"] == "reverted" and r["rejections"] == 0
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


@pytest.mark.anyio
async def test_fails_open_logs_class_name_only_and_downstream_errors_propagate(caplog):
    down = Downstream(json.dumps(ANTH_RESP).encode())
    mw = PithMiddleware(down, Runtime(Config(), Boom()))
    with caplog.at_level(logging.WARNING, logger="pith.headroom"):
        sent, _ = await call(mw, "/v1/messages", ANTH)
    assert down.seen[-1] == ANTH and sent[0]["status"] == 200 and down.decision_seen is None
    assert "sk-secret" not in caplog.text and "RuntimeError" in caplog.text
    mw, down, conn = make(raise_exc=ValueError("headroom's own problem"))
    with pytest.raises(ValueError):
        await call(mw, "/v1/messages", ANTH)
    mw, down, conn = make()
    sent, _ = await call(mw, "/v1/messages", {"not": "json"} and "not json")  # body is a JSON string, not an object
    assert sent[0]["status"] == 200 and conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


@pytest.mark.anyio
async def test_receive_delegates_to_the_original_after_the_buffered_body():
    class Waits(Downstream):
        async def __call__(self, scope, receive, send):
            await super().__call__(scope, receive, send)
            self.extra = await receive()
    conn = db.connect(":memory:")
    down = Waits(json.dumps(ANTH_RESP).encode())
    mw = PithMiddleware(down, Runtime(Config(sample_rate=0), conn))
    _, delegated = await call(mw, "/v1/messages", ANTH)
    assert down.extra == {"type": "http.disconnect"} and delegated == [1]


def test_runtime_is_lazy_and_reads_env(tmp_path):
    rt = Runtime(env={"OPTIMIZER_DB_PATH": str(tmp_path / "h.db"), "OPTIMIZER_SAMPLE_RATE": "1"})
    cfg, conn = rt.ready()
    assert cfg.db_path == str(tmp_path / "h.db") and cfg.sample_rate == 1.0 and (tmp_path / "h.db").exists()
    assert rt.ready() == (cfg, conn)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_headroom.py -q`
Expected: `ModuleNotFoundError: No module named 'pith.headroom'`.

- [ ] **Step 3: Implement**

Create `pith/headroom.py`:

```python
"""Headroom adapter: pith's data plane inside a Headroom proxy. Spec: docs/design/specs/2026-10-09-headroom-adapter-design.md.

`install` (entry point headroom.proxy_extension) adds `PithMiddleware`, which decides on the client's original body and
records from the teed response; `PithPipeline` (entry point headroom.pipeline_extension) appends the pinned shape text at
PRE_SEND, after Headroom's compression. Headroom is never imported. Every pith step fails open.
"""
import contextvars
import json
import logging
import os
import time
from typing import Mapping

from pith import db
from pith.config import Config, load_config
from pith.proxy import PURGE_EVERY, choose, record
from pith.rewrite import RouteState, append_shape, apply_effort
from pith.usage import StreamUsage, usage_from_body

log = logging.getLogger("pith.headroom")
PATHS = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai"}
PARAM_PROFILES = ("P1", "P1b", "P4")
SHAPE_PROFILES = ("P2", "P3", "P4")
DECISION: contextvars.ContextVar = contextvars.ContextVar("pith_decision", default=None)


class Runtime:
    """Lazy config + SQLite connection, shared by the middleware and the pipeline extension of one process."""

    def __init__(self, config: Config | None = None, conn=None, env: Mapping[str, str] | None = None):
        self.config, self.conn = config, conn
        self.env = os.environ if env is None else env
        self.n = 0

    def ready(self):
        if self.config is None:
            self.config = load_config(self.env.get("OPTIMIZER_CONFIG"), self.env)
        if self.conn is None:
            self.conn = db.connect(self.config.db_path)  # created on the event-loop thread, on first use
        return self.config, self.conn


RUNTIME = Runtime()


def _decide(rt: Runtime, provider: str, raw: bytes, headers: dict) -> tuple[dict | None, bytes]:
    """(decision, bytes to forward). A None decision means pith stays out of this request entirely."""
    mode = (headers.get("x-optimizer") or "").lower()
    if mode == "bypass":
        return None, raw
    cfg, conn = rt.ready()
    body = json.loads(raw)
    if not isinstance(body, dict):
        return None, raw
    fp, route, profile, _ = choose(cfg, conn, provider, body, mode, headers.get("x-optimizer-route"))
    if route["injection_form"] != "user_text":
        db.set_injection_form(conn, fp.key, "user_text")  # Headroom gets the shape as user text; sweeps must replay it so
    if profile in PARAM_PROFILES:
        raw = json.dumps(apply_effort(provider, body, profile)).encode()
    decision = {"route": fp.key, "profile": profile, "applied": profile not in SHAPE_PROFILES, "body": body,
                "t0": time.monotonic(), "target_words": route["target_words"], "exemplar": route["exemplar"]}
    return decision, raw


class PithMiddleware:
    """Pure ASGI. Buffers the request body, decides, forwards (possibly with effort params rewritten), tees the
    response and records. Only POST /v1/messages and /v1/chat/completions; everything else is untouched."""

    def __init__(self, app, runtime: Runtime | None = None):
        self.app, self.rt = app, runtime or RUNTIME

    async def __call__(self, scope, receive, send):
        provider = PATHS.get(scope.get("path", "").rstrip("/")) if scope.get("type") == "http" else None
        if provider is None or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        chunks, more = [], True
        while more:
            m = await receive()
            chunks.append(m)
            more = m.get("type") == "http.request" and m.get("more_body", False)
        raw = b"".join(c.get("body", b"") for c in chunks if c.get("type") == "http.request")
        decision = None
        try:
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
            decision, raw = _decide(self.rt, provider, raw, headers)
        except Exception as exc:  # fail open: forward the client's bytes untouched; never log exc text
            log.warning("decide failed (%s); forwarding original request", type(exc).__name__)
        DECISION.set(decision)
        queue = [{"type": "http.request", "body": raw, "more_body": False}]
        queue += [c for c in chunks if c.get("type") != "http.request"]

        async def replay():
            return queue.pop(0) if queue else await receive()  # after the buffered body, Headroom waits on the real receive

        su, buf, status = StreamUsage(provider), [], [None]

        async def tee(message):
            if message["type"] == "http.response.start":
                status[0] = message.get("status")
            elif message["type"] == "http.response.body" and decision is not None:
                chunk = message.get("body", b"")
                buf.append(chunk)
                su.feed(chunk)  # never raises
            await send(message)

        await self.app(scope, replay, tee)
        if decision is not None:
            self._finish(provider, decision, status[0], su, b"".join(buf))

    def _finish(self, provider: str, decision: dict, status, su: StreamUsage, raw: bytes) -> None:
        try:
            cfg, conn = self.rt.ready()
            used = decision["profile"] if decision["applied"] else "P0"
            if status is not None and status < 400:
                try:
                    usage = usage_from_body(provider, json.loads(raw))
                except Exception:  # SSE or non-JSON body: the tee already parsed it
                    usage = su.result()
                self.rt.n += 1
                if self.rt.n % PURGE_EVERY == 0:
                    db.purge_expired(conn, time.time())
                record(cfg, conn, decision["route"], used, usage, int((time.monotonic() - decision["t0"]) * 1000),
                       json.dumps(decision["body"]), raw.decode("utf-8", "replace"))
            elif status in (400, 422) and used != "P0":
                if db.bump_rejection(conn, decision["route"]) >= 3:
                    db.set_pin(conn, decision["route"], "P0", status="reverted")
                    log.warning("route %s reverted to P0 after 3 provider rejections", decision["route"])
        except Exception as exc:
            log.warning("record failed (%s)", type(exc).__name__)
```

Note on `test_fails_open...`'s third case: `{"not": "json"} and "not json"` evaluates to the string `"not json"`, so the body is the JSON string `"not json"`; `_decide` returns `(None, raw)` for a non-object body and the request passes through.

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest tests/test_headroom.py -q -W error` then `.venv/bin/pytest -q -W error`
Expected: 8 passed in the new file; whole suite green.

- [ ] **Step 5: Commit**

```bash
git add pith/headroom.py tests/test_headroom.py
git commit -m "feat: Headroom middleware decides on the raw request and records from the teed response"
```

---

### Task 3: `PithPipeline`, `install`, entry points, live mock test

**Files:**
- Modify: `pith/headroom.py` (append), `pyproject.toml`
- Create: `tests/live/test_headroom_mock.py`
- Test: `tests/test_headroom.py`

**Interfaces:**
- Consumes: `DECISION`, `RUNTIME`, `PithMiddleware` (Task 2); `rewrite.append_shape`, `RouteState` (Task 1).
- Produces: `PithPipeline()` with `on_pipeline_event(event) -> None` (reads `event.stage.name`, `event.messages`); `install(app, config) -> None` calling `app.add_middleware(PithMiddleware)`; entry points `headroom.proxy_extension: pith = pith.headroom:install` and `headroom.pipeline_extension: pith = pith.headroom:PithPipeline`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_headroom.py` (add `import sys` and `from types import SimpleNamespace` to the imports, and extend the `pith.headroom` import to `from pith.headroom import DECISION, PithMiddleware, PithPipeline, Runtime, install`):

```python
class Event:
    def __init__(self, stage, messages, provider="anthropic"):
        self.stage, self.messages, self.provider = SimpleNamespace(name=stage), messages, provider


@pytest.mark.anyio
async def test_pipeline_applies_shape_at_pre_send_and_the_request_records_as_pinned():
    mw, down, conn = make()
    pipe = PithPipeline()
    down.hook = lambda body: pipe.on_pipeline_event(Event("PRE_SEND", body["messages"]))
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.seen[-1] == ANTH  # P0: PRE_SEND leaves the messages alone
    db.set_pin(conn, "r", "P2")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    last = down.seen[-1]["messages"][-1]
    assert last["content"][0] == {"type": "text", "text": "q"} and last["content"][1]["text"].startswith("Answer directly.")
    assert down.seen[-1]["system"] == "S" and "output_config" not in down.seen[-1]
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == ANTH
    db.set_pin(conn, "r", "P4")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.seen[-1]["output_config"] == {"effort": "low"}
    assert down.seen[-1]["messages"][-1]["content"][1]["text"].startswith("Answer directly.")
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P4"


def test_pipeline_ignores_other_stages_missing_decisions_and_fails_open(caplog):
    pipe = PithPipeline()
    msgs = [{"role": "user", "content": "q"}]
    DECISION.set({"route": "r", "profile": "P2", "applied": False, "body": {}, "t0": 0.0, "target_words": 20, "exemplar": None})
    assert pipe.on_pipeline_event(Event("POST_SEND", msgs)) is None and msgs == [{"role": "user", "content": "q"}]
    pipe.on_pipeline_event(Event("PRE_SEND", []))
    assert DECISION.get()["applied"] is False
    with caplog.at_level(logging.WARNING, logger="pith.headroom"):
        pipe.on_pipeline_event(Event("PRE_SEND", ["not a message dict"]))
    assert DECISION.get()["applied"] is False and "AttributeError" in caplog.text
    DECISION.set(None)
    pipe.on_pipeline_event(Event("PRE_SEND", msgs))
    assert msgs == [{"role": "user", "content": "q"}]
    pipe.on_pipeline_event(object())  # an event without the expected attributes is ignored


def test_install_adds_the_middleware_and_module_needs_no_headroom():
    added = []
    install(SimpleNamespace(add_middleware=lambda cls, **kw: added.append((cls, kw))), config=None)
    assert added == [(PithMiddleware, {})]
    assert "headroom" not in sys.modules
```

Create `tests/live/test_headroom_mock.py`:

```python
"""Boots Headroom's real proxy app with the pith extensions against a mock upstream: no API key, no network.

Run: .venv/bin/pip install headroom-ai && .venv/bin/pip install -e . && OPTIMIZER_HEADROOM_LIVE=1 .venv/bin/pytest tests/live/test_headroom_mock.py -v
Headroom is not a pith dependency and is not installed in CI, so this file is skipped unless the flag is set.
The second pip command re-registers pith's entry points so Headroom can discover `pith` by name.
"""
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from pith import db

pytestmark = pytest.mark.skipif(os.environ.get("OPTIMIZER_HEADROOM_LIVE") != "1", reason="set OPTIMIZER_HEADROOM_LIVE=1")

SYSTEM = "You classify support tickets. " * 40
SEEN = []


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append((self.path, body))
        anthropic, stream = self.path.startswith("/v1/messages"), body.get("stream")
        if stream:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            if anthropic:
                events = [("message_start", {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": body["model"], "content": [], "usage": {"input_tokens": 11, "output_tokens": 1}}}),
                          ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                          ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
                          ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                          ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}}),
                          ("message_stop", {"type": "message_stop"})]
                for name, ev in events:
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(ev)}\n\n".encode())
            else:
                for c in [{"id": "c", "object": "chat.completion.chunk", "model": body["model"], "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
                          {"id": "c", "object": "chat.completion.chunk", "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                          {"id": "c", "object": "chat.completion.chunk", "model": body["model"], "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}]:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            return
        if anthropic:
            out = {"id": "m", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "end_turn",
                   "content": [{"type": "text", "text": "Hello"}], "usage": {"input_tokens": 11, "output_tokens": 5}}
        else:
            out = {"id": "c", "object": "chat.completion", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
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


@pytest.fixture
def headroom(tmp_path, monkeypatch):
    create_app = pytest.importorskip("headroom.proxy.server").create_app
    ProxyConfig = pytest.importorskip("headroom.proxy.models").ProxyConfig
    import uvicorn

    from pith.headroom import PithPipeline

    monkeypatch.setenv("HEADROOM_BEACON", "off")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("OPTIMIZER_DB_PATH", str(tmp_path / "pith.db"))
    monkeypatch.setenv("OPTIMIZER_SAMPLE_RATE", "1")
    up = free_port()
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", up), Upstream).serve_forever, daemon=True).start()
    cfg = ProxyConfig(anthropic_api_url=f"http://127.0.0.1:{up}", openai_api_url=f"http://127.0.0.1:{up}",
                      proxy_extensions=["pith"], pipeline_extensions=[PithPipeline()], discover_pipeline_extensions=False,
                      cache_enabled=False, cost_tracking_enabled=False, subscription_tracking_enabled=False,
                      periodic_toin_stats_enabled=False, license_report_interval=10**6)
    app = create_app(cfg)
    assert any(type(getattr(m, "cls", None)).__name__ == "type" and m.cls.__name__ == "PithMiddleware" for m in app.user_middleware), \
        "Headroom did not install the pith proxy extension: run `pip install -e .` so the entry point is registered"
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(base + "/health", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        pytest.fail("headroom did not start")
    yield base, str(tmp_path / "pith.db")
    server.should_exit = True


def test_pith_inside_headroom_records_and_applies_pins(headroom):
    base, db_path = headroom
    H = {"x-optimizer-route": "hr", "content-type": "application/json"}
    chat = {"model": "gpt-4o", "max_tokens": 50, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Where is my order?"}]}
    msg = {"model": "claude-haiku-4-5", "max_tokens": 50, "system": SYSTEM, "messages": [{"role": "user", "content": "Where is my order?"}]}
    oa = {**H, "authorization": "Bearer sk-x"}
    an = {**H, "x-api-key": "sk-ant-x", "anthropic-version": "2023-06-01"}
    for path, body, hdr in [("/v1/chat/completions", chat, oa), ("/v1/chat/completions", dict(chat, stream=True, stream_options={"include_usage": True}), oa),
                            ("/v1/messages", msg, an), ("/v1/messages", dict(msg, stream=True), an)]:
        r = httpx.post(base + path, json=body, headers=hdr, timeout=30)
        assert r.status_code == 200, r.text
    time.sleep(0.5)
    conn = db.connect(db_path)
    rows = conn.execute("SELECT profile, input_tokens, output_tokens, stop_reason FROM requests ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("P0", 12, 5, "stop"), ("P0", 12, 5, "stop"), ("P0", 11, 5, "end_turn"), ("P0", 11, 5, "end_turn")]
    assert db.get_route(conn, "hr")["injection_form"] == "user_text"
    db.set_pin(conn, "hr", "P2")
    conn.close()
    assert httpx.post(base + "/v1/messages", json=msg, headers=an, timeout=30).status_code == 200
    time.sleep(0.5)
    path, sent = SEEN[-1]
    assert sent["system"] == SYSTEM and sent["messages"][-1]["content"][-1]["text"].startswith("Answer directly.")
    conn = db.connect(db_path)
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_headroom.py -q`
Expected: `ImportError: cannot import name 'PithPipeline'`.

- [ ] **Step 3: Implement**

Append to `pith/headroom.py`:

```python
class PithPipeline:
    """Entry point headroom.pipeline_extension. At PRE_SEND (after compression) append the pinned shape as user text."""

    def on_pipeline_event(self, event):
        try:
            d = DECISION.get()
            if d is None or getattr(getattr(event, "stage", None), "name", None) != "PRE_SEND":
                return None
            if d["profile"] in SHAPE_PROFILES and event.messages:
                append_shape(event.messages, RouteState(d["profile"], "user_text", d["target_words"], d["exemplar"]))
                d["applied"] = True
        except Exception as exc:  # fail open: the request goes out unshaped and is recorded as P0
            log.warning("PRE_SEND shape failed (%s); request forwarded unchanged", type(exc).__name__)
        return None


def install(app, config) -> None:
    """Entry point headroom.proxy_extension: Headroom calls this while building its app."""
    app.add_middleware(PithMiddleware)
```

Append to `pyproject.toml` after the `[project.optional-dependencies]` table:

```toml
[project.entry-points."headroom.proxy_extension"]
pith = "pith.headroom:install"

[project.entry-points."headroom.pipeline_extension"]
pith = "pith.headroom:PithPipeline"
```

- [ ] **Step 4: Run the suite, then the live test**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass; `tests/live/test_headroom_mock.py` skipped.

Then:

```bash
.venv/bin/pip install -q headroom-ai uvicorn && .venv/bin/pip install -q -e . && OPTIMIZER_HEADROOM_LIVE=1 .venv/bin/pytest tests/live/test_headroom_mock.py -v
```

Expected: 1 passed (Headroom prints its own startup banner; that is fine). If the middleware assertion in the fixture fails, the entry point is not registered: re-run `.venv/bin/pip install -e .`. Afterwards `.venv/bin/pytest -q -W error` must still be green with Headroom installed (the unit suite never imports it).

- [ ] **Step 5: Commit**

```bash
git add pith/headroom.py pyproject.toml tests/test_headroom.py tests/live/test_headroom_mock.py
git commit -m "feat: Headroom pipeline extension applies pins at PRE_SEND; entry points and live mock test"
```

---

### Task 4: README section

**Files:**
- Modify: `README.md` (new section after "## LiteLLM plugin", before "## Sweeps: turning observation into pins"; one line in "## Tests")

- [ ] **Step 1: Insert the section**

Immediately before the line `## Sweeps: turning observation into pins` insert:

````markdown
## Headroom plugin

Running [Headroom](https://github.com/chopratejas/headroom)'s proxy? Install pith into the same environment and enable
its two extensions; no second hop and no Headroom code changes:

    pip install git+https://github.com/dex0shubham/pith
    HEADROOM_PROXY_EXTENSIONS=pith HEADROOM_PIPELINE_EXTENSIONS=pith OPTIMIZER_CONFIG=/path/to/pith.toml headroom proxy

(`headroom proxy --proxy-extension pith` is the flag form of the first variable; the pipeline extension has no flag.)
The proxy extension installs a middleware that fingerprints each `/v1/messages` and `/v1/chat/completions` request on
the client's original body, records usage from the response, and rewrites effort parameters for P1/P4 pins. The pipeline
extension appends the shape text for P2/P3/P4 pins at Headroom's `PRE_SEND` stage, after compression, as user text.
Routes recorded this way are ordinary Anthropic/OpenAI routes: the same keys as traffic through `pith serve`, swept with
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` and the full profile set. `X-Optimizer: off|bypass` and `X-Optimizer-Route` work
as usual. Keep Headroom's own output shaper off (`HEADROOM_OUTPUT_SHAPER` unset) while pith is enabled: two steering
instructions would fight. `/v1/responses` passes through unrecorded. Any pith error forwards the request unchanged.
Headroom's beacon and telemetry switches are Headroom's own (`HEADROOM_BEACON=off`).

````

In "## Tests", after the LiteLLM live-test paragraph, append:

```markdown
`OPTIMIZER_HEADROOM_LIVE=1 .venv/bin/pytest tests/live/test_headroom_mock.py` boots Headroom's real proxy app with the
pith extensions against a mock upstream (needs `pip install headroom-ai` and a re-run of `pip install -e .`, no API key).
```

- [ ] **Step 2: Check and commit**

Run: `.venv/bin/pytest -q -W error` (unchanged, all pass) and confirm the README gained no attribution or trailer lines.

```bash
git add README.md
git commit -m "docs: Headroom plugin section"
```

---

## Self-review against the spec

- §1 decisions: 1 (two entry points, one module, opt-in env) → Tasks 3–4; 2 (decide/record in middleware, shape at PRE_SEND, own parsers) → Tasks 2–3; 3 (full profile set via `apply_effort` + `append_shape`, plain provider routes, `injection_form = user_text`) → Tasks 1–2; 4 (fail open, class names only) → Tasks 2–3 (tested with `Boom` and a bad event); 5 (scope: two paths) → Task 2.
- §2 facts: receive delegation after the buffered body → Task 2 (`replay` + test); tee parsing → Task 2; `PRE_SEND` by `stage.name` → Task 3; shaper note → Task 4.
- §3 module contract: `DECISION` keys, `_decide` steps 1–5, `tee`/`_finish` steps 6–8, `PithPipeline`, `Runtime` → Tasks 2–3. The spec's step 4 lists P1/P4 for the raw-body rewrite; the plan also applies P1b (the OpenAI verbosity flag is a parameter too and `apply_effort` owns it). Treat that as a one-line spec amendment, recorded here.
- §4 rewrite split → Task 1. §5 packaging and docs → Tasks 3–4. §6 tests → every task; live test → Task 3.
- Names consistent across tasks: `Runtime`, `RUNTIME`, `DECISION`, `PATHS`, `PARAM_PROFILES`, `SHAPE_PROFILES`, `PithMiddleware`, `PithPipeline`, `install`, `apply_effort`, `append_shape`, `OPTIMIZER_HEADROOM_LIVE`.
