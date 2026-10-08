# Output-Token Optimizer — Design Spec

Date: 2026-10-07. Status: approved in brainstorming; awaiting user review of this document.
Research backing this spec: `docs/research/2026-10-07-shorthand-tokens-research.md`.

## 1. What and why

A self-hosted, drop-in proxy for the Claude and OpenAI APIs that learns, per route, the shortest model output that still passes a quality bar, enforces it through the providers' own knobs, and proves it with continuous judge-scored shadow evaluation.

Positioning: **the output-token control plane.** Output tokens cost 5× input on every current Claude model (4× industry median, up to 8×). Input compression is a crowded market (Headroom 74.5k★, RTK, Compresr, The Token Company, LLMLingua) and research shows it raises realized cost on API models; output compression cuts realized cost 1.4–2.4× on short-answer workloads and nobody sells the policy-and-measurement loop for it. Providers ship the knobs (`effort`, `verbosity`, `max_tokens`, format, cache-safe mid-conversation instructions) but not the decision of which to use per route or the proof that quality held.

ICP: API product teams on Claude/OpenAI with high-volume short-answer routes (classification, extraction, support chat, summarization). Not coding agents (output is ~2% of their bill).

## 2. Scope

### In v1
- Proxy for `POST /v1/messages` (Anthropic), `POST /v1/chat/completions` and `POST /v1/responses` (OpenAI). Streaming and non-streaming.
- Route fingerprinting, traffic sampling, per-route sweeps, judge, pin rule, drift detection, per-route report.
- Profiles P0–P4 (and P1b on OpenAI) as defined in §5.
- Self-hosted single container; all data on the customer's disk; customer's own API keys and judge account.

### Out of v1 (roadmap, §12)
- Per-request adaptive budgets; online bandits.
- Structured-format conversion and stop-sequence sentinels as profiles.
- Hosted/multi-tenant control plane, dashboard product, at-rest encryption.
- Any input-side compression.

## 3. Architecture

One Python process (FastAPI + httpx streaming), SQLite for state, no other services.

**Data plane (every request):** receive → detect provider by path → fingerprint route → load pinned profile → rewrite request cache-safely (§7) if a profile is pinned → forward with the customer's auth headers untouched → stream/return the provider response byte-for-byte → record `{route, profile, usage, latency, stop_reason, estimated_usage flag}` and, for a sampled fraction, the full request and response body. Fail-open: any proxy-side failure forwards the original bytes.

**Control plane (background task in the same process):** per route, once ≥50 sampled requests exist, freeze the sample, run the sweep (§6), pin the winning profile, and continue shadow-sampling for drift. Re-sweep on route hash change, drift, or schedule.

**Customer-facing output:** `GET /optimizer/report` (JSON) and `GET /optimizer/report.html` (static render): per route — baseline vs pinned profile, output tokens, $/1k requests, equivalence %, noise floor, sample size, last sweep, status (`observing | pinned | no-savings | not-applicable | sweeping | reverted`; every route starts as `observing`).

**Trust boundary:** prompts, responses, sweep artifacts, and judge outputs never leave the customer's network. No telemetry.

## 4. Route fingerprinting and applicability

**Route key** = `provider + model + sha256(normalized system prompt ‖ sorted tool names and schemas)`. For OpenAI, "system prompt" means the leading `system`/`developer` messages (Chat Completions) or `instructions` plus leading developer items (Responses). The customer may override with an `X-Optimizer-Route: <name>` header.

Normalization: whitespace-collapsed, no other transformation. Two routes that differ only by a timestamp in the system prompt are two routes; the report flags routes with low repeat counts so the customer can fix their caching.

**Applicability gate:** a route is eligible when ≥80% of its sampled baseline responses end with `end_turn`/`stop` and contain text. Routes dominated by tool calls are `not-applicable`. Routes already using structured outputs (`output_config.format` / `response_format` / `text.format`) get only the effort profiles.

## 5. Profile library

Ordered least → most invasive. Settings are pinned per route, never varied per request.

| Profile | Claude (`/v1/messages`) | OpenAI (Chat Completions / Responses) |
|---|---|---|
| P0 baseline | passthrough | passthrough |
| P1 effort-down | `output_config.effort` one notch below the route's current value (default per model if unset); skipped on models without `effort` | `reasoning_effort` / `reasoning.effort` one notch down; skipped if the model rejects it |
| P1b verbosity-low | — | `verbosity: "low"` / `text.verbosity: "low"` (GPT-5+ only) |
| P2 shape instruction | mid-conversation `{"role":"system","content":"<shape text>"}` appended after the last user message | `{"role":"developer","content":"<shape text>"}` appended at the end of `messages` / a developer item at the end of `input` |
| P3 shape + exemplar | P2 text plus one judge-approved short answer from the route's own sample as a one-shot example | same |
| P4 effort-down + shape | P1 ∘ P2 | P1 ∘ P2 |

**Shape text:** "Answer directly. No preamble, restatement, or closing summary. Target at most N words unless the task genuinely needs more." N = round(0.5 × baseline p50 word count of the route), minimum 20.

**Never a lever:** `max_tokens` / `max_completion_tokens` / `max_output_tokens`, `thinking`, `model`, top-level `system`, `tools`, `temperature`. Any rise in `stop_reason: max_tokens` (or `finish_reason: length`) disqualifies a profile.

## 6. Sweep, judge, and pin rule

> **Amended 2026-10-07 by the Plan 2 spec** (`2026-10-07-control-plane-design.md`), which is authoritative where they differ: sweeps and rechecks run from an operator-invoked CLI with provider keys in its environment — the proxy stays keyless and runs no background work; replays go straight to the provider upstream (no `X-Optimizer: bypass` hop); drift is detected by an offline `recheck` rather than a live shadow twin; `shadow_rate` is retired.

**Sample:** 50 frozen requests per route, stratified across baseline output-length quintiles so the long tail is represented. Frozen samples are immutable and versioned per sweep.

**Replay:** for each profile, replay the full sample sequentially (keeps the customer's prompt cache warm and comparable across profiles), 3 trials per item. Replay requests carry `X-Optimizer: bypass` so they are neither rewritten again nor counted as traffic. Side-effect guard: the frozen sample excludes any item whose baseline response contains a tool call (a route may still be eligible under §4 with up to 20% such responses), so replays can never trigger customer tools.

**Cost gate:** before any sweep the runner prices it from the route's own usage (items × profiles × trials × mean cost, plus judge calls) and refuses to exceed the configured per-route or monthly ceiling. Default ceiling: $0 until the customer sets one — the proxy ships observing only.

**Judge:** one prompt, versioned and frozen per sweep, on a configurable model (default `claude-sonnet-5-5` via the customer's own account). Input: the request's last user turn, the baseline response, the candidate response. Output: exactly one label — `equivalent | missing-info | contradiction | format-broken`. The judge sees responses in randomized A/B order per trial to limit position bias.

**Noise floor:** the judge also scores baseline-vs-baseline across trials. A route's self-consistency rate is its ceiling; profiles are compared against it, not against 100%.

**Pin rule:** choose the cheapest profile (cost per request including amortized judge/sweep spend) satisfying all of:
1. equivalence rate ≥ bar (default 0.95, per-route override);
2. equivalence rate ≥ noise floor − 0.03;
3. no increase in `max_tokens`/`length` stops versus P0;
4. mean `cache_read_input_tokens` (Claude) not lower than P0's.
If none qualifies: pin P0, status `no-savings`.

**Drift:** `recheck` (CLI) replays up to 20 recent sampled live responses per pinned route at P0 with the operator's key and judges live-vs-P0, one `shadow` row per pair. If rolling equivalence over the route's last 100 shadow rows drops below the bar, revert to P0 and mark `reverted` (eligible for the next sweep). Route hash change → the proxy unpins keylessly inside `upsert_route`.

## 7. Cache-safe request rewriting

The proxy never edits top-level `system`, `tools`, `model`, `thinking`, or any token cap. Those sit at the front of the cache prefix or are hard caps; editing them re-bills the whole conversation.

**Claude:**
- P1: set `output_config.effort`. Constant per route → cache-neutral after the first request.
- P2/P3: append `{"role":"system","content":"..."}` after the last `user` message (supported on Opus 5 / 5.5 / 4.8, Fable 5 / 5.1, Sonnet 5.5; no beta header). On a 400 containing `role 'system' is not supported`, retry once with the customer's **original** request and, if that succeeds, switch the route to the fallback form for subsequent requests: a `{"type":"text"}` block appended at the end of the last user message's content, after any customer `cache_control` block. Record which form the route uses. Rejection bookkeeping of any kind runs only when the retried original succeeds (status < 400); a 4xx or 5xx on the original is the customer's or provider's problem, not the rewrite's.
- Forward `anthropic-version`, `anthropic-beta`, and every unknown field verbatim.

**OpenAI:**
- P1: `reasoning_effort` (Chat) / `reasoning.effort` (Responses). On gpt-6-astra the proxy may instead insert a `configuration_update` item before the last user item; v1 uses the request-level field because it is constant per route.
- P1b: `verbosity` / `text.verbosity`.
- P2/P3: append a `developer` message/item at the end. OpenAI caching is exact-prefix; end-append is safe.

**Provider detection:** by path. Unknown paths and methods are forwarded without inspection.

## 8. Error handling and safety

- **Fail-open:** exception anywhere before forwarding → forward the original bytes. Provider **400 or 422** on a rewritten request (the statuses a rewrite can plausibly cause) → retry once with the original request, log `profile_rejected`, auto-unpin after 3 rejections on a route. Other 4xx (401/403/404/413/429…) are not rewrite-caused: pass through unchanged, no retry, no count. Upstream transport failures become a synthetic 502 (unreachable) or 504 (timeout).
- **Streaming:** chunks pass through untouched. Claude usage is read from `message_start`/`message_delta`. OpenAI streams without `stream_options.include_usage` have no usage; estimate with tiktoken and flag `estimated_usage=true` — the proxy does not alter the customer's stream shape.
- **Kill switches:** `X-Optimizer: off` per request; `routes.<key>.enabled=false` in config; `OPTIMIZER_ENABLED=0` globally. All three force P0 passthrough.
- **Secrets:** `Authorization` / `x-api-key` forwarded, never logged or stored.
- **Data retention:** sampled bodies stored plain in SQLite on the customer's disk; `retention_days` default 14. `note:` no at-rest encryption in v1 — inside the customer's VPC; add when a customer requires it.
- **Budget:** §6 cost gate.

## 9. Configuration

Single `pith.toml` (env-var overrides):

```toml
listen = "0.0.0.0:8787"
anthropic_upstream = "https://api.anthropic.com"
openai_upstream = "https://api.openai.com"
db_path = "./pith.db"
sample_rate = 0.05          # fraction of requests whose bodies are stored
retention_days = 14
sweep_budget_usd_month = 0  # 0 = observe only, never sweep
equivalence_bar = 0.95
judge_model = "claude-sonnet-5-5"
judge_provider = "anthropic"

[prices."gpt-5"]            # $/M tokens; extends/overrides the built-in Claude table (illustrative values — check current pricing)
input = 1.25
output = 10.0

[routes."<route-key-or-name>"]
enabled = true
equivalence_bar = 0.97
```

## 10. Data model (SQLite)

- `routes(key, provider, model, system_hash, name, status, pinned_profile, injection_form, eligible, first_seen, last_seen, last_sweep_id)`
- `requests(id, ts, route_key, profile, input_tokens, output_tokens, cache_read, cache_create, estimated, stop_reason, latency_ms, body_ref)`
- `bodies(id, request_json, response_json, expires_at)` — sampled only
- `samples(id, route_key, created_at, item_ids_json)` — frozen per sweep
- `sweeps(id, route_key, sample_id, judge_model, judge_prompt_version, started_at, finished_at, cost_usd, result_json, winner)`
- `judgments(id, sweep_id, item_id, profile, trial, label, order_ab)`
- `shadow(id, ts, route_key, request_id, label)`

## 11. Testing

- **Unit:** fingerprint determinism and normalization; per-provider rewrite functions against golden JSON for every profile and both injection forms; fail-open on malformed bodies; pin rule against synthetic sweep tables (pin / no-savings / noise-floor-limited / max_tokens-disqualified / cache-regression-disqualified).
- **Contract:** record-replay fixtures of real streaming and non-streaming responses from both providers; a P0 passthrough test asserting the response is byte-identical to a direct call; streaming usage extraction for both providers.
- **Cache safety (live, standing):** two identical requests through a route pinned to P2 → `cache_read_input_tokens > 0` on the second. Run against the real Claude API in CI with a scoped key; this is the test that proves the product doesn't cost more than it saves.
- **E2E (live):** one synthetic route — 50 support-ticket classifications — through observe → sweep → pin against both providers; assert a profile pins at or above the bar with fewer output tokens than P0. Doubles as the demo.
- pytest only. One `test_*.py` per module; no fixture frameworks.

## 12. Roadmap (not v1)

1. Per-request adaptive budgets on routes whose sweep shows a wide complexity spread (use Claude's per-message effort system message and OpenAI's `configuration_update` to stay cache-safe).
2. Structured-format and stop-sequence-sentinel profiles.
3. Plugin packaging for LiteLLM / Portkey / Headroom for distribution.
4. Hosted control plane with aggregate, non-prompt telemetry.

## 13. Success criteria for v1

- Passthrough overhead ≤ 5 ms p50 added latency, zero behavioral difference at P0.
- On the E2E demo route: ≥25% fewer output tokens with equivalence ≥ 0.95 and no cache-read regression.
- A route with no available savings is reported as such, not forced.
- The customer can read, for every pinned route, the sweep that justified it.
