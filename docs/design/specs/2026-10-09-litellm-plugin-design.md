# Plan 3: LiteLLM Guardrail Plugin Design

Date: 2026-10-09. Status: approved in brainstorming; awaiting user review of this document.
Parent specs: `2026-10-07-output-token-optimizer-design.md` (§5 profiles, §7 rewriting, §8 recording) and
`2026-10-07-control-plane-design.md` (sweeps, judge, pin rule). Plans 1 and 2 are merged. This plan packages the data
plane as a LiteLLM proxy guardrail so a team already running LiteLLM gets pith without a second proxy hop, and lets the
existing sweep CLI tune the routes that guardrail records.

## 1. Decisions

1. **Guardrail, not a custom provider and not a documented chain.** A `CustomGuardrail` subclass registered in LiteLLM's
   `config.yaml` sees every chat-completion request before the provider call and every response after it. It cannot retry
   a rejected request, so it applies only the shape profiles (decision 3) and relies on the failure hook for bookkeeping.
2. **Sweeps replay through the LiteLLM proxy.** A third provider kind, `litellm`, sends OpenAI-shaped bodies to
   `/v1/chat/completions` on `litellm_upstream` with `LITELLM_API_KEY`. Replays and judge calls carry
   `X-Optimizer: bypass`, so the guardrail neither rewrites nor records them.
3. **Shape-only profiles, user-text form.** `PROFILES_BY_PROVIDER["litellm"] = ("P0", "P2", "P3")`. LiteLLM folds
   `system`/`developer` messages into the provider's top-level system prompt, which would break prompt caches, so the
   shape text is appended to the last user message, the form the Anthropic path already falls back to. P1/P1b/P4 are
   no-ops for this provider: LiteLLM maps effort and verbosity differently per backend and a guardrail cannot recover
   from a rejection with the customer's original request.
4. **Fail open at every hook.** A guardrail exception rejects the customer's request in LiteLLM, so every hook catches
   everything, logs the exception class name only (never its text: it can embed headers), and returns its input
   unchanged.
5. **No new dependency.** pith never imports `litellm` at module import time. The operator installs pith into the
   LiteLLM proxy's environment, which already has LiteLLM.

## 2. Verified LiteLLM facts (spike, 2026-10-09, LiteLLM proxy with a `mock_response` model)

- At `async_pre_call_hook(user_api_key_dict, cache, data, call_type)` on `/v1/chat/completions`, `call_type` is
  `"acompletion"` and `data` holds the client fields plus `litellm_call_id`, `litellm_logging_obj`, `metadata`,
  `proxy_server_request`, `secret_fields`. `data["proxy_server_request"]["body"]` is the client's original body with no
  LiteLLM keys. `data["metadata"]` carries any request-body `metadata` keys at top level, `headers` (every request
  header, lower-cased), `requester_metadata`, and `user_api_key_*`. Request headers are not forwarded to providers by
  default; body `metadata` is a LiteLLM logging param, not a provider param.
- A dict stored under `data["metadata"]["pith"]` at pre-call is present, by identity, in
  `async_post_call_success_hook(data, user_api_key_dict, response)`,
  `async_post_call_streaming_iterator_hook(user_api_key_dict, response, request_data)` and
  `async_post_call_failure_hook(request_data, original_exception, user_api_key_dict, traceback_str=None)`.
- Post-call `response` is a `litellm.ModelResponse`; `response.model_dump()` is an OpenAI chat-completion dict with
  `usage` and `choices[].finish_reason`. `response.model` is the backend model, `data["model"]` the client-facing alias.
- Streaming chunks are `litellm.ModelResponseStream`; `finish_reason` arrives on a chunk before the last, and `usage`
  only on the last chunk, only when the client sent `stream_options.include_usage`.
- The failure hook fires twice for one failed request. `original_exception` carries `status_code` for provider errors.
- `mode: [pre_call, post_call]` on the guardrail entry enables all four hooks; `default_on: true` applies it to every key.

## 3. Module: `pith/guardrail.py`

Two classes:

- `PithHooks` — framework-free. Constructed with `(config, conn)`; the guardrail subclass constructs it lazily on the
  first hook from `load_config(env.get("OPTIMIZER_CONFIG"), env)` and `db.connect(cfg.db_path)` (so the connection is
  created on the event-loop thread). Methods below.
- `PithGuardrail(litellm.integrations.custom_guardrail.CustomGuardrail, PithHooks)` — defined inside a function or
  guarded import so `pith.guardrail` imports without LiteLLM. Its `__init__` forwards `**kwargs` to `CustomGuardrail`.

Hooks (all wrapped per decision 4):

- `async_pre_call_hook(user_api_key_dict, cache, data, call_type)`:
  1. Return `data` untouched unless `call_type in ("completion", "acompletion")` and `data` has `messages`.
  2. `body = data["proxy_server_request"]["body"]` if present, else `{k: v for k, v in data.items() if k not in LITELLM_KEYS}`
     where `LITELLM_KEYS = {"litellm_call_id", "litellm_logging_obj", "metadata", "litellm_metadata", "proxy_server_request", "secret_fields"}`.
     Drop `metadata` from the copy either way (a replayed body must not carry LiteLLM's merged metadata).
  3. `headers = data["metadata"]["headers"]`; `mode = headers.get("x-optimizer", "").lower()`; `bypass` returns `data`
     untouched and records nothing.
  4. `fp, route, profile, new_body = choose(cfg, conn, "litellm", body, mode, headers.get("x-optimizer-route"), responses_api=False)`
     (§5). `choose` applies `config.enabled`, `mode == "off"`, and per-route `enabled`, exactly as the proxy.
  5. If `profile != "P0"`: `data["messages"] = new_body["messages"]` (the only key the `litellm` branch of
     `apply_profile` changes).
  6. `data["metadata"]["pith"] = {"route": fp.key, "profile": profile, "body": body, "t0": time.monotonic()}`; return `data`.
- `async_post_call_success_hook(data, user_api_key_dict, response)`: if `pith` stash present and `response` has
  `model_dump`, `resp = response.model_dump()`, `usage = usage_from_body("openai", resp)`, then
  `record(cfg, conn, stash["route"], stash["profile"], usage, latency_ms, json.dumps(stash["body"]), json.dumps(resp))`
  (§5). Return `response`.
- `async_post_call_streaming_iterator_hook(user_api_key_dict, response, request_data)`: async generator. Yield every
  chunk unchanged; on the side, collect `choices[0].delta.content` text, the last non-null `finish_reason`, and
  `usage` when a chunk carries one. After the iterator is exhausted (not on cancellation), build
  `resp = {"object": "chat.completion", "model": ..., "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": fr}], "usage": usage or {}}`;
  if usage is absent, `output_tokens = estimate_tokens(text)` with `estimated=True` (same rule as `StreamUsage.result`
  for OpenAI); then `record(...)` as above. `stored_response_text("openai", ...)` reads this dict unchanged.
- `async_post_call_failure_hook(request_data, original_exception, user_api_key_dict, traceback_str=None)`: if the stash
  exists, `stash["profile"] != "P0"`, `getattr(original_exception, "status_code", None) in (400, 422)` and
  `stash.get("failed")` is not set: set `stash["failed"] = True` (idempotent across the double fire), then
  `db.bump_rejection(conn, route) >= 3` → `db.set_pin(conn, route, "P0", status="reverted")` and a warning. A P0
  request's failure is the customer's problem and is ignored. Nothing is recorded for failures (the proxy records only
  `< 400`).

Sampling, retention, purge: `record` samples bodies at `cfg.sample_rate` and stores them with `retention_days` expiry
exactly as the proxy; the guardrail calls `db.purge_expired` every `PURGE_EVERY` records like the proxy does.

## 4. Provider kind `litellm`

- `Config.litellm_upstream: str = "http://localhost:4000"`; `OPTIMIZER_LITELLM_UPSTREAM` override follows from `load_config`.
- `providers.upstream("litellm", cfg)` → `cfg.litellm_upstream`. `detect_provider` is unchanged (the proxy never sees
  this provider; routes with it are created only by the guardrail).
- `__main__.ENV_KEYS["litellm"] = "LITELLM_API_KEY"`; `judge_provider = "litellm"` is therefore accepted by the
  existing key check and `build_judge_request` already produces the OpenAI shape for any non-Anthropic provider.
- `replay.call`: for `provider == "litellm"` add header `X-Optimizer: bypass`. `endpoint_for` (chat completions when
  `messages` present), `auth_headers` (bearer), `response_text`, `usage_from_body`, `cost_of` already take the OpenAI path.
- `rewrite.apply_profile("litellm", body, state)`: P2/P3 → append `_shape(state)` to the last user message via the
  existing user-text helper (`state.injection_form` is ignored; the form is always user text). Everything else returns
  the body unchanged, which the sweep's "no-op for every item → skipped" rule turns into a skipped row. The route's
  `injection_form` column stays at its default and is never consulted for this provider.
- `sweep.PROFILES_BY_PROVIDER["litellm"] = ("P0", "P2", "P3")`.
- Prices: `price_for` is keyed by `route.model`, which for this provider is the client-facing LiteLLM alias. A
  `[prices."<alias>"]` entry is required before a sweep; without one the sweep refuses with the existing `NoPrice`,
  and the report shows no dollar columns for the route. The `x-litellm-response-cost` header is not used.
- Everything in `sweep.run_sweep`, `recheck`, `judge`, `fingerprint` (messages path: leading `system`/`developer`
  messages hash as the system prompt), `db` and `report` is unchanged.

## 5. Changes to `pith/proxy.py`

Two extractions so the guardrail and the proxy share one decision and one recording path:

- `choose(config, conn, provider, body, mode, route_name, responses_api) -> tuple[Fingerprint, dict, str, dict]`:
  fingerprint, `upsert_route`, `get_route`, enabled check, `apply_profile` (returns the original `body` for P0).
  `_decide` becomes: detect provider, handle `bypass`, `json.loads`, call `choose`, build `Decision`.
- `record(config, conn, route_key, profile, usage, latency_ms, request_json, response_json)`: the body of today's
  `_record` with the `Decision` parameter replaced by `route_key`. The proxy's two call sites pass `d.fp.key`.

Behaviour of the proxy is unchanged; `tests/test_proxy.py` must pass untouched.

## 6. Configuration and documentation

LiteLLM `config.yaml`:

    guardrails:
      - guardrail_name: pith
        litellm_params:
          guardrail: pith.guardrail.PithGuardrail
          mode: [pre_call, post_call]
          default_on: true

Environment of the LiteLLM proxy process: `OPTIMIZER_CONFIG=/path/pith.toml` (optional; every pith setting is also
reachable as `OPTIMIZER_<FIELD>`, e.g. `OPTIMIZER_DB_PATH`). Install: `pip install git+https://github.com/dex0shubham/pith`
into the LiteLLM proxy's environment. Sweep from the same host:

    export LITELLM_API_KEY=sk-...          # a LiteLLM virtual key or the master key
    .venv/bin/python -m pith sweep --config pith.toml   # litellm_upstream points at the LiteLLM proxy

`pith.example.toml` gains `litellm_upstream`. README gains a "LiteLLM plugin" section with the above, the kill switches
(`X-Optimizer: off|bypass`, `X-Optimizer-Route` still work because the guardrail reads the recorded headers), and the
note that only P2/P3 apply through LiteLLM and that `[prices]` entries are needed for aliases.

## 7. Testing

- `tests/test_guardrail.py` (anyio, already a dev dependency): drive `PithHooks` with request dicts shaped like §2 and
  fake responses exposing `model_dump()`; fake async iterators of chunk objects for streaming; an exception object with
  `status_code` for failures. Cases:
  1. P0 route and non-chat `call_type` leave `data` identical and record a P0 request (chat) or nothing (non-chat).
  2. `x-optimizer: bypass` records nothing and leaves `data` identical; `off` records at P0 on a pinned route.
  3. `x-optimizer-route` names the route.
  4. A P2 pin appends the shape text to the last user message only, leaves other keys untouched, and stashes
     `metadata.pith` with the original body (no `metadata` key inside it).
  5. Post-call records usage, latency and profile, and samples a body at `sample_rate = 1`.
  6. Streaming: chunks pass through unchanged; usage recorded when present; estimated when absent; nothing recorded
     when the consumer stops early.
  7. Failure hook: 400 after a rewrite bumps once even when called twice for the same request; the third distinct
     rejection reverts the pin; failures at P0 and non-4xx failures change nothing.
  8. Any exception inside a hook (e.g. a `conn` that raises) returns `data`/`response` unchanged and logs only a class name.
  9. `pith.guardrail` imports without `litellm` installed; `PithGuardrail` is created only when LiteLLM is importable.
- Existing suites gain: `apply_profile("litellm", …)` for every profile; `upstream("litellm")`; `replay.call` sends
  the bypass header for `litellm` only; `PROFILES_BY_PROVIDER["litellm"]`; `ENV_KEYS["litellm"]`; `choose`/`record`
  through the proxy tests unchanged.
- `tests/live/test_litellm_mock.py`: `pytest.importorskip("litellm")`; starts `litellm --config` with a
  `mock_response` model on a free port, no API key; asserts a route and requests appear in a temp DB, bypass is not
  recorded, and streaming records usage. CI does not install LiteLLM, so this is skipped there.

## 8. Out of scope

Responses API and `/v1/messages` through LiteLLM (chat completions only); effort/verbosity profiles via LiteLLM;
running sweeps from inside the LiteLLM process; LiteLLM's own spend logs or `x-litellm-response-cost`; Portkey or
Headroom adapters.
