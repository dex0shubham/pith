# Plan 5: Portkey Webhook Design

Date: 2026-10-09. Status: approved in brainstorming; awaiting user review of this document.
Parent specs: `2026-10-07-output-token-optimizer-design.md`, `2026-10-07-control-plane-design.md`,
`2026-10-09-litellm-plugin-design.md` (conventions for host adapters). Portkey's AI gateway runs hooks on every request;
its built-in `default.webhook` check POSTs the hook context to a URL and applies a returned transformed request. pith
therefore needs no gateway code: one endpoint in its own app.

## 1. Decisions

1. **An endpoint, not a plugin.** pith's FastAPI app gains `POST /optimizer/portkey`. Portkey is configured with a
   `before_request_hooks` mutator and an `after_request_hooks` guardrail that both use `default.webhook` pointing at
   that URL. No TypeScript, no gateway build.
2. **OpenAI-format requests only.** The handler acts when `requestType == "chatComplete"`; any other payload gets
   `{"verdict": true}` untouched. Anthropic-format (`messages`) traffic through Portkey is out of scope.
3. **Shape-only profiles, user text.** Provider kind `portkey` with `PROFILES_BY_PROVIDER["portkey"] = ("P0", "P2", "P3")`.
   A before-hook cannot retry a rejected request and after hooks run only on HTTP 200, so provider rejections are
   invisible: there is no revert-on-rejection through Portkey, and `recheck` remains the drift guard.
4. **Streams are not recorded.** Portkey delivers `response.json = null` for streaming responses; the after hook then
   records nothing. Route statistics through Portkey cover non-streaming traffic only (documented).
5. **Never raise.** Any error returns `{"verdict": true}` with no transform and logs the exception class name only, so a
   pith failure never blocks or alters a customer's request.

## 2. Verified Portkey facts (spike, 2026-10-09, open-source gateway via `npx @portkey-ai/gateway`, default port 8787)

- Request config (`x-portkey-config` header or a saved config) accepts `before_request_hooks` and `after_request_hooks`;
  each hook has `type` (`mutator` or `guardrail`), `id`, optional `deny`, and `checks: [{"id": "default.webhook",
  "parameters": {"webhookURL": ..., "headers": {...}, "timeout": ms}}]`.
- The webhook receives a JSON POST with keys `eventType` (`beforeRequestHook` | `afterRequestHook`), `provider`
  (e.g. `openai`), `requestType` (`chatComplete`), `metadata` (the parsed `x-portkey-metadata` header, `{}` when
  absent), `request: {json, text, isStreamingRequest, isTransformed}` (headers removed by Portkey) and
  `response: {json, text, statusCode, isTransformed}`. `parameters.headers` are sent as HTTP headers to the webhook.
- A before-hook reply `{"verdict": true, "transformedData": {"request": {"json": <body>}}}` replaces the body sent
  upstream (verified: the upstream received the appended text). A reply without `transformedData` leaves it unchanged.
- The after hook runs only for status 200; `request.json` there is the transformed body and `response.json` is the
  provider's JSON for non-streaming calls and `null` for streams.
- Portkey appends `hook_results` to the client-facing JSON response whenever hooks ran. That is Portkey's behaviour.
- The default webhook timeout is 3000 ms; a timeout or non-2xx from the webhook fails the check (verdict false), which
  with `deny: false` still forwards the request.

## 3. Module: `pith/portkey.py`

```python
def handle(cfg: Config, conn, payload: dict) -> dict
```

- Returns `{"verdict": True}` unless `payload.get("requestType") == "chatComplete"` and `payload["request"]["json"]` is a
  dict with `messages`.
- `meta = payload.get("metadata") or {}`; `meta.get("pith_bypass")` truthy → `{"verdict": True}`;
  `mode = "off" if str(meta.get("pith", "")).lower() == "off" else ""`; `route_name = meta.get("pith_route")`.
- `beforeRequestHook`: `fp, route, profile, body = choose(cfg, conn, "portkey", req, mode, route_name)`; if
  `profile == "P0"` → `{"verdict": True}`; else `{"verdict": True, "transformedData": {"request": {"json": body}}}` where
  `body` is `apply_profile("portkey", ...)`'s result (shape appended to the last user message as user text, §4).
- `afterRequestHook`: `resp = payload["response"]["json"]`; `None`/non-dict → `{"verdict": True}`. Otherwise
  `original, stripped = strip_shape(req)` (§4); `fp = fingerprint("portkey", original, route_name)`; `route =
  db.get_route(conn, fp.key)` (register with `upsert_route` if missing); `profile = route["pinned_profile"] if stripped
  else "P0"`; `record(cfg, conn, fp.key, profile, usage_from_body("openai", resp), 0, json.dumps(original),
  json.dumps(resp))`; purge every `PURGE_EVERY` records; return `{"verdict": True}`. If `stripped` is true but
  `route["pinned_profile"]` is not P2/P3 (the pin changed between the two hooks: a revert or a model-change unpin),
  record nothing: a shaped response must never enter the P0 baseline.
- Everything is inside `try/except Exception` returning `{"verdict": True}` and logging the class name.

Route in `pith/proxy.py` (`create_app`):

```python
@app.post("/optimizer/portkey")
async def portkey_hook(request: Request):
    if config.webhook_token and request.headers.get("authorization") != f"Bearer {config.webhook_token}":
        return Response(status_code=401)
    return handle(config, conn, await request.json())
```

A malformed JSON body returns `{"verdict": True}` (handled inside `handle` via a guarded parse in the route).

## 4. `pith/rewrite.py` additions

- `apply_profile("portkey", body, state)` behaves as the `litellm` branch: P2/P3 → `append_shape` on the last user
  message, every other profile unchanged. The branch condition becomes `provider in ("litellm", "portkey")`.
- `strip_shape(body) -> tuple[dict, bool]`: deep-copies `body`; if the last user message's content is a list whose last
  part is `{"type": "text", "text": t}` with `t.startswith(SHAPE_TEXT.split("{n}")[0])`, removes that part, and when a
  single text part remains, restores it to string content; returns `(body, True)`. Otherwise `(copy, False)`.
  `append_shape` followed by `strip_shape` on a string-content message is the identity (tested).

## 5. Provider kind `portkey`

- `Config.portkey_upstream: str = "http://localhost:8787"` (Portkey's default port; pith's own default `listen` is also
  8787, so one of the two must move; the README says so) and `Config.webhook_token: str | None = None`
  (`OPTIMIZER_WEBHOOK_TOKEN`). `Config.portkey_headers: dict[str, str]` from a `[portkey_headers]` table in `pith.toml`
  (for `x-portkey-config`, `x-portkey-provider`, `x-portkey-virtual-key` or custom-host headers the replays need).
- `__main__.ENV_KEYS["portkey"] = "PORTKEY_API_KEY"`; `providers.upstream("portkey", cfg)` → `cfg.portkey_upstream`.
- `replay.auth_headers("portkey", key)` → `{"x-portkey-api-key": key}`; `replay.call` for `portkey` merges
  `cfg.portkey_headers` and sets `x-portkey-metadata: {"pith_bypass": true}`; `endpoint_for` already yields
  `/v1/chat/completions`; usage, text and cost take the OpenAI path. `judge_provider = "portkey"` therefore works.
- `sweep.PROFILES_BY_PROVIDER["portkey"] = ("P0", "P2", "P3")`. Prices: `[prices."<model>"]` keyed by the model name
  the client sends.

## 6. Docs

README "Portkey plugin" section: the config JSON (mutator before hook, guardrail after hook with `deny: false`, both
`default.webhook` with the pith URL and an optional `Authorization` header), `x-portkey-metadata` keys `pith_route`,
`pith_bypass`, `pith: "off"`, the two limits (OpenAI-format only; streams not recorded), the `hook_results` note, the
port clash, `portkey_upstream`/`[portkey_headers]`/`PORTKEY_API_KEY` for sweeps, and that the endpoint should be reachable
from the gateway but not from the public internet (`webhook_token` when it must be).

## 7. Testing

- `tests/test_portkey.py`: `handle` with payloads shaped like §2 (both event types, streaming after-hook with `null`
  json, non-chatComplete, bypass, route name, `off`, P2 pin transforms and the after hook strips and records at P2,
  P0 records at P0, raising connection returns verdict true and logs a class name); the FastAPI route with and
  without `webhook_token` (401 on a wrong token) through `httpx.ASGITransport`; `strip_shape` round trip.
- Existing suites gain: `apply_profile("portkey")`, `upstream("portkey")`, `auth_headers`/`call` headers for
  `portkey`, `PROFILES_BY_PROVIDER`, `ENV_KEYS`, config fields and the `[portkey_headers]` table.
- `tests/live/test_portkey_mock.py`, skipped unless `OPTIMIZER_PORTKEY_LIVE=1` and `npx` is available: starts the
  open-source gateway, a mock OpenAI upstream and pith's app, sends a non-streaming request through the gateway with
  the hook config, and asserts the route and request rows and, after a P2 pin, the shape text at the upstream.

## 8. Out of scope

Anthropic-format requests through Portkey; recording streams; revert-on-rejection; a TypeScript plugin in the gateway
repo (can be contributed later, calling the same endpoint); Portkey's hosted-product UI configuration beyond the JSON.
