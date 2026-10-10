# Plan 4: Headroom Adapter Design

Date: 2026-10-09. Status: approved in brainstorming; awaiting user review of this document.
Parent specs: `2026-10-07-output-token-optimizer-design.md` (§5 profiles, §7 rewriting, §8 recording),
`2026-10-07-control-plane-design.md` (sweeps), `2026-10-09-litellm-plugin-design.md` (the first host adapter; this one
follows its conventions). Headroom (`headroom-ai` on PyPI, 0.40 at the time of writing) is a Python FastAPI proxy that
compresses LLM context; this plan runs pith's data plane inside it so a Headroom user gets output-profile pins without a
second hop.

## 1. Decisions

1. **Two Headroom entry points, one pith module.** `pith/headroom.py` provides `install(app, config)` registered under
   the `headroom.proxy_extension` entry-point group and `PithPipeline` under `headroom.pipeline_extension`. Both are
   declared in pith's `pyproject.toml`; installing pith into Headroom's environment makes them discoverable. Enabling is
   Headroom's opt-in: `HEADROOM_PROXY_EXTENSIONS=pith` (or `headroom proxy --proxy-extension pith`) and
   `HEADROOM_PIPELINE_EXTENSIONS=pith`. Headroom is never a dependency and is never imported at module level.
2. **Decide and record in middleware; shape at `PRE_SEND`.** A pure-ASGI middleware (installed by the proxy extension)
   sees the client's original body before compression and the response bytes after it, which is where the full system
   prompt, the request params and the provider's usage live. Headroom's pipeline events carry neither the Anthropic
   `system` nor params, and its outcome event has no finish reason for OpenAI (and none for Anthropic streams), so the
   adapter uses pith's own parsers on the teed response. The shape text is appended at `PRE_SEND`, the last event
   before Headroom forwards upstream and after its compression, so compression never reshapes pith's text and
   pith never overwrites a message another extension changed.
3. **Full profile set; routes are ordinary provider routes.** Effort profiles (P1, P4) rewrite the effort field in the
   raw body, which Headroom passes through untouched; shape profiles (P2, P3, P4) append user text at `PRE_SEND`.
   Stored bodies are the clients' originals, so routes carry provider `anthropic` or `openai`, share keys with the same
   app's traffic through pith's own proxy, and are swept directly against the providers with the existing profile lists.
   The adapter sets `injection_form = "user_text"` on routes it registers so sweep replays match what Headroom sends:
   `_openai_shape` honours `user_text` for chat bodies (the shape goes to the last user message, not a developer
   message), so sweeps of Headroom-recorded OpenAI routes replay the served form. Responses bodies keep the developer
   item; the adapter never handles them.
4. **Fail open at every step.** Any pith error forwards the request unchanged and logs the exception class name only.
5. **Scope:** `POST /v1/messages` and `POST /v1/chat/completions`. `/v1/responses` passes through unrecorded.

## 2. Verified Headroom facts (spike, 2026-10-09, headroom-ai 0.40.0, built with `create_app(ProxyConfig(...))`)

- A `contextvars.ContextVar` set in a pure-ASGI middleware is visible in the `PRE_SEND`, `POST_SEND`,
  `RESPONSE_RECEIVED` and `OUTCOME_OBSERVED` events of the same request, for both providers, streamed or not.
- `PipelineEvent` fields: `stage` (`PipelineStage` enum; compare `stage.name == "PRE_SEND"`), `provider` (`"anthropic"`
  or `"openai"`), `model`, `messages` (the post-compression list, Anthropic- or OpenAI-shaped; mutate in place),
  `tools`, `headers` (outbound, includes client headers such as `x-optimizer-route` and credentials), `metadata`
  (`{"path", "stream"}` at `PRE_SEND`). An in-place append to the last user message at `PRE_SEND` reached the upstream in
  all four cases. The Anthropic `system` is not on the event.
- `OUTCOME_OBSERVED` gives `(input_tokens, output_tokens, stop_reason)` of `(12, 5, None)` for OpenAI non-stream and
  stream, `(11, 5, "end_turn")` for Anthropic non-stream and `(16, 1, None)` for Anthropic stream: not usable for pith.
- Teeing `http.response.body` ASGI messages in the middleware and parsing with `usage_from_body` (JSON) or
  `StreamUsage` (SSE) gave `(12, 5, "stop")` and `(11, 5, "end_turn")` for all four cases.
- A middleware that buffers the request body must, after replaying the buffered messages, delegate further `receive`
  calls to the original callable: Headroom's streaming path waits on `receive` for the client disconnect, and a fabricated
  `http.request` there raises `RuntimeError: Unexpected message received`.
- Headroom runs proxy extensions' `install` before its token gate and body-limit middleware; `install` may call
  `app.add_middleware(cls)` with a pure-ASGI class whose constructor takes `app`. Pipeline extension exceptions are caught
  by Headroom and logged (fail-open).
- Headroom has an opt-in output shaper (`HEADROOM_OUTPUT_SHAPER`, system-tail instruction with the sentinel
  `<headroom_output_shaping>`) and no effort or `max_tokens` steering. Two steering instructions would fight: the README
  tells operators to keep the shaper off when pith is enabled. The same holds for Headroom's learned verbosity steering
  (`HEADROOM_VERBOSITY_LEVEL`, `headroom learn --verbosity`).
- `ProxyConfig.cache_enabled` defaults to True (TTL 3600 s, `HEADROOM_CACHE_ENABLED`). On a non-stream cache hit Headroom
  answers before `PRE_SEND` and emits `INPUT_CACHED` instead; its cache key is computed before `PRE_SEND`, so cached
  entries are shared between shaped and unshaped requests. The README tells operators to turn the cache off while pins
  are in use; the adapter never records a cache hit.
- Proxy extensions are enabled by name (`proxy_extensions` / `HEADROOM_PROXY_EXTENSIONS`); pipeline extensions are
  discovered from entry points only when named in `HEADROOM_PIPELINE_EXTENSIONS` (with `discover_pipeline_extensions`
  True, the default). An entry point that is a class is instantiated with no arguments.

## 3. Module: `pith/headroom.py`

```python
DECISION: ContextVar[dict | None]   # {"route", "profile", "applied": bool, "request_json": str, "t0": float,
                                    #  "target_words": int, "exemplar": str | None, "cached": bool (set at INPUT_CACHED)}
PATHS = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai"}

def install(app, config) -> None            # proxy extension: app.add_middleware(PithMiddleware)
class PithMiddleware:                       # pure ASGI; __init__(self, app, *, config=None, conn=None, env=None)
class PithPipeline:                         # pipeline extension; on_pipeline_event(event) -> None
```

`PithMiddleware.__call__(scope, receive, send)`:
1. Pass through unless `scope["type"] == "http"`, method `POST`, and `scope["path"].rstrip("/")` in `PATHS`.
2. Buffer the request body (all `http.request` messages); afterwards `receive` replays them then delegates to the
   original. Decode headers lower-cased; `mode = headers.get("x-optimizer", "")`; `bypass` passes through untouched
   (pith records nothing). On every matching request the scope forwarded to Headroom drops `x-optimizer` and
   `x-optimizer-route`, so Headroom never sends pith's headers upstream.
3. `fp, route, profile, body = choose(cfg, conn, provider, json.loads(raw), mode, headers.get("x-optimizer-route"))`
   (the shared core from `pith/proxy.py`). If the route's `injection_form` is not `"user_text"`,
   `db.set_injection_form(conn, fp.key, "user_text")`.
4. If `profile in ("P1", "P4")`: `raw = json.dumps(apply_effort(provider, original_body, profile)).encode()` where
   `apply_effort` is the effort half of `apply_profile` (§5). The shape half is not applied here.
5. `DECISION.set({"route": fp.key, "profile": profile, "applied": profile not in ("P2", "P3", "P4"),
   "request_json": original_raw.decode("utf-8", "replace"),
   "t0": time.monotonic(), "target_words": route["target_words"], "exemplar": route["exemplar"]})`. A pure-effort profile counts as applied at this point; shape profiles become applied only
   when `PithPipeline` runs.
6. Call the downstream app with a `send` wrapper that captures the status and tees every `http.response.body` chunk into
   a `StreamUsage(provider)` and a byte buffer.
7. After the downstream call returns, nothing is recorded or counted if `decision["cached"]` is set (Headroom answered
   from its response cache, no upstream call) or if the response never sent its final `http.response.body` message
   (`more_body` False or absent). Otherwise `profile_used = profile if decision["applied"] else ("P1" if profile ==
   "P4" else "P0")`: an unapplied P4 still had its effort rewritten. If `status < 400`:
   usage is `usage_from_body(provider, json.loads(bytes))` when the body parses as JSON, else `StreamUsage.result()`;
   `record(cfg, conn, route, profile_used, usage, latency_ms, decision["request_json"], bytes.decode("utf-8", "replace"))`.
   If `status in (400, 422)`, `profile in ("P1", "P1b", "P4")` and the response body (decoded, lower-cased) contains
   neither `context_length_exceeded` nor `prompt is too long`: `db.bump_rejection`, reverting to P0 at 3. The
   middleware cannot retry the original request to learn whether pith's change caused the rejection, so it attributes
   conservatively: a user-text append cannot plausibly be rejected, and a prompt-size error is the client's.
   Steps 2–5 and 7 are wrapped so any exception passes the original request through (or skips recording) with a
   warning carrying the exception class name only. The `send` wrapper never raises into the stream.
8. `db.purge_expired` every `PURGE_EVERY` records, as in the guardrail.

`PithPipeline.on_pipeline_event(event)`: if `DECISION.get()` is `None`, return. At `INPUT_CACHED`, set
`decision["cached"] = True`. At `PRE_SEND`, if `decision["applied"]` is not already True, `decision["profile"] in ("P2",
"P3", "P4")` and `event.messages`: `decision["applied"] = append_shape(event.messages, state)` where `state =
RouteState(profile, "user_text", decision["target_words"], decision["exemplar"])` (False when there is no user message;
the already-applied check keeps a doubly registered extension from appending twice). Returns `None` (in-place
mutation). Wrapped; never raises.

Configuration and connection: constructed lazily on first use from `load_config(env.get("OPTIMIZER_CONFIG"), env)` and
`db.connect(cfg.db_path)` on the event-loop thread, identical to `PithHooks._ready`; the first `ready()` also starts a
daemon thread that warms tiktoken, as `pith/proxy.py:create_app` does. `PithMiddleware` and `PithPipeline`
share one module-level `_Runtime` holder so both use one connection.

## 4. `pith/rewrite.py` changes

`apply_profile` is split into two public halves it then composes:
- `apply_effort(provider, body, profile, responses_api=False) -> dict`: the P1/P4 effort step-down (Anthropic
  `output_config.effort`, OpenAI `reasoning_effort` / `reasoning.effort`) and the P1b verbosity flag; returns a deep copy.
- `append_shape(messages, state) -> bool`: in-place user-text append of `_shape(state)` to the last user message,
  returning False when there is no user message
  (today's `_anthropic_shape` user-text path), used by the LiteLLM branch, the Headroom pipeline extension and the
  Portkey webhook.
`apply_profile` keeps its signature and behaviour; `tests/test_rewrite.py` passes unchanged plus tests for the halves.

## 5. Packaging and docs

`pyproject.toml`:

    [project.entry-points."headroom.proxy_extension"]
    pith = "pith.headroom:install"
    [project.entry-points."headroom.pipeline_extension"]
    pith = "pith.headroom:PithPipeline"

No dependency added. README "Headroom plugin" section: install pith into Headroom's environment, the two env variables
(or the `--proxy-extension` flag), `OPTIMIZER_CONFIG`, that routes are plain Anthropic/OpenAI routes swept with the
provider keys, that `X-Optimizer`/`X-Optimizer-Route` work, that `HEADROOM_OUTPUT_SHAPER` must stay off, that
`/v1/responses` is passed through unrecorded, and that Headroom's beacon/telemetry settings are Headroom's own.

## 6. Testing

- `tests/test_headroom.py` (anyio): drive `PithMiddleware` with a fake downstream ASGI app that records the body it
  receives and emits a JSON or SSE response; drive `PithPipeline` with a plain object carrying `stage` (an object with
  `.name`), `provider`, `messages`. Cases: non-matching path/method passes through untouched and records nothing; bypass;
  `off` forces P0; route header; P0 records usage for JSON and SSE responses, Anthropic and OpenAI; P1 rewrites the
  effort field in the body the downstream receives and records P1; P2 pin: `PRE_SEND` appends to the last user message
  and the request records as P2, while a P2 pin with no `PRE_SEND` records as P0; the stored request body is the
  original; the route's injection form becomes `user_text`; 400 after a rewrite counts and reverts at three; a raising
  connection forwards the request unchanged and logs only a class name; the replayed `receive` delegates to the
  original after the buffered messages; the downstream app raising propagates (fail-open does not swallow Headroom's
  own errors); `pith.headroom` imports without Headroom installed.
- `tests/live/test_headroom_mock.py`, skipped unless `OPTIMIZER_HEADROOM_LIVE=1`: builds Headroom's app with
  `create_app(ProxyConfig(anthropic_api_url=mock, openai_api_url=mock, proxy_extensions=["pith"],
  discover_pipeline_extensions=True, ...))` and `HEADROOM_PIPELINE_EXTENSIONS=pith` (entry-point discovery for both
  extensions) against a mock upstream (as the spike did), sends the four request shapes on one route per provider, pins
  P2 then P4, and asserts the upstream saw the shape text and the DB rows. A second test enables Headroom's response
  cache, sends one non-stream Anthropic request three times and asserts one upstream call and one row. Requires `pip install headroom-ai` and a re-install of pith so the entry points register.

## 7. Out of scope

Responses API through Headroom; Headroom's `/v1/compress` gateway contract and LiteLLM-fronted Headroom (use the pith
LiteLLM guardrail there); the Rust `headroom-proxy`; yielding automatically to Headroom's output shaper.
