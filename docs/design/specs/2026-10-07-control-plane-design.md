# Output-Token Optimizer — Plan 2: Control Plane Design

Date: 2026-10-07. Status: approved in brainstorming; awaiting user review of this document.
Parent spec: `2026-10-07-output-token-optimizer-design.md` (§6 is amended by this document where noted). Plan 1 (data plane) is merged; this plan adds the part that decides pins.

## 1. Decisions that reshape §6

1. **The proxy stays keyless.** Sweeps and judge calls run from an operator-invoked CLI with provider keys in its environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). The proxy never holds credentials and runs no background sweeps.
2. **Drift is detected offline.** The in-proxy 2% live shadow twin is dropped. `recheck` replays recent sampled live responses at P0 and judges them. `shadow_rate` is retired from config.
3. **Hash-change unpin happens in the proxy, keylessly**, inside `db.upsert_route`.
4. **Pins are applied by the sweep automatically** (the operator chose to run it); `--dry-run` prints the decision without writing.

## 2. CLI

`python -m optimizer <subcommand>`:

| Subcommand | Behaviour |
|---|---|
| `serve` (default) | Existing proxy. |
| `sweep [--route KEY] [--dry-run] [--budget-usd N] [--trials 3] [--sample 50]` | Sweep every eligible route (or one). Refuses a provider whose key is absent from the environment. |
| `recheck [--route KEY] [--n 20]` | Drift check on pinned routes. Never pins; may revert. |

Both read `optimizer.toml` via `load_config` (and `OPTIMIZER_*` overrides), open `db.connect(cfg.db_path)`, and print one table per route to stdout. Exit code 0 on completion, 2 on budget refusal, 1 on abort.

## 3. Modules

- `optimizer/sweep.py` — `eligible_routes(conn, cfg)`, `freeze_sample(conn, route_key, n)`, `estimate_cost(route, sample, profiles, trials, cfg)`, `run_sweep(conn, cfg, route, client, keys, *, trials, dry_run, budget_usd)`, `pin_rule(table, bar, floor)`, `recheck(conn, cfg, route, client, keys, n)`.
- `optimizer/judge.py` — `JUDGE_PROMPT_VERSION = "v1"`, `build_judge_request(provider, model, question, answer_a, answer_b)`, `parse_label(text) -> str | None`, `judge(client, cfg, keys, question, baseline, candidate, rng) -> (label, usage)`.
- `optimizer/__main__.py` — argparse subcommands.
- Reused unchanged: `rewrite.apply_profile`, `usage.usage_from_body`, `report.PRICES`, `db`, `config.load_config`.

Config additions (§9 amended): `[prices."<model>"] input = <$/M> output = <$/M>` overrides/extends `PRICES`; `shadow_rate` removed.

## 4. Eligibility and sampling

A route is swept when:
- `status in ("observing", "reverted")`, and
- it has ≥ `--sample` (default 50) P0 requests with `body_ref` not null, and
- ≥ 80% of its P0 requests have a text-ending `stop_reason` (`end_turn`, `stop`, `completed`).

If the request count is ≥ 50 but the stop-reason gate fails: `status = not-applicable`, `eligible = 0`. Otherwise `eligible = 1`.

Sample: P0 requests with `body_ref` and a text-ending stop, ordered by `ts` desc within the retention window, bucketed into five `output_tokens` quintiles, up to 10 from each (fill from neighbouring quintiles if one is short), total ≤ `--sample`. Item ids (request ids) are frozen in `samples.item_ids_json`. Replay bodies are the stored `request_json` with `stream` removed; nothing else is changed, including `cache_control` blocks. OpenAI shape is detected from the body: `messages` → `/v1/chat/completions`, else `/v1/responses`.

## 5. Replay

Profiles: Anthropic `P0 P1 P2 P3 P4`; OpenAI `P0 P1 P1b P2 P3 P4`. Order: P0 for all trials first; then from P0 trial-1 responses derive `target_words = max(20, round(0.5 × median word count))` and `exemplar` = the shortest trial-1 answer text; then the remaining profiles. A profile whose rewritten body equals the original (e.g. P1 on a model without `effort`) is skipped and marked `skipped` in the result table.

Each call: non-streaming POST to the provider upstream from config; Anthropic headers `x-api-key`, `anthropic-version: 2023-06-01`; OpenAI `Authorization: Bearer`. Sequential within a profile. A 429 is retried once after `retry-after` (max 60 s). Per item and profile the sweep records: text, `output_tokens`, `input_tokens`, `cache_read`, `stop_reason`, status, usage-derived cost. An item fails a profile (`format-broken`, no judge call) when status ≥ 400, `stop_reason` is `max_tokens`/`length`/`max_output_tokens`, or the text is empty.

Trials: `--trials` (default 3) for every profile including P0.

## 6. Judge

System prompt (frozen under `JUDGE_PROMPT_VERSION`): "You compare two answers to the same request. Decide whether they convey the same facts, decisions and required output. Reply with exactly one label."

User message: the request's last user-turn text (truncated to 4,000 characters), `ANSWER A:` …, `ANSWER B:` …, then the slot-qualified labels with definitions (symmetric, so the judge cannot be blind to an omission in either slot):
- `equivalent` — both answers convey the same facts, decisions and required output, nothing contradictory.
- `A-omits` — Answer A omits a fact, decision or required output that Answer B states.
- `B-omits` — Answer B omits a fact, decision or required output that Answer A states.
- `contradiction` — the answers assert incompatible things.
- `A-broken` / `B-broken` — that answer is empty, truncated, or not a usable answer.

Which of baseline/candidate is A is randomized per trial with the sweep's seeded RNG and stored in `judgments.order_ab` (`"baseline-first"` / `"candidate-first"`). Labels are normalized to the candidate's perspective and stored as one of `equivalent | missing-info | extra-info | contradiction | format-broken | judge-error`: the slot holding the candidate maps `*-omits → missing-info` and `*-broken → format-broken`; the slot holding the baseline maps `*-omits → extra-info` (the candidate added claims — a failure) and `*-broken → judge-error` (the baseline itself was unusable). Only `equivalent` counts as a pass; `judge-error` is excluded from rates. `parse_label` takes the first label token in the reply (case-insensitive); an unparseable reply is retried once, then recorded as `judge-error`. The judge model/provider come from config; the key from the environment; judge usage is added to sweep cost. Items whose candidate already failed mechanically (status ≥400, `max_tokens` stop, empty text) get `format-broken` without a call. The Anthropic judge request uses `max_tokens: 1024` with `output_config.effort: low`; the OpenAI request `max_completion_tokens: 1024` (hidden reasoning counts against both).

Baseline for every candidate: P0 trial 1. Noise floor: P0 trial 1 judged against P0 trials 2…`--trials`.

## 7. Pin rule

Per profile: `rate = equivalent / (judged − judge-error)`. `floor` = noise-floor rate. `bar` = route override else `equivalence_bar`.

A profile qualifies when all hold:
1. `rate ≥ bar`
2. `rate ≥ floor − 0.03`
3. `max_tokens` stops ≤ P0's count
4. mean `cache_read` ≥ P0's mean `cache_read` (checked only when P0's mean > 0)
5. `$/request < P0's $/request`

`$/request` = mean input × input price + mean output × output price (+ amortized sweep cost for candidates) where the sweep's total actual cost is amortized over projected monthly volume = `requests in last 7 days × 30/7`, floored at 1,000. Prices from `PRICES` merged with `[prices]` config; a model with no price makes `estimate_cost` refuse with exit 2.

Cheapest qualifying profile is pinned: `target_words`, `exemplar`, `last_sweep_id` are written first, then `set_pin(key, profile)` (so the data plane never sees a pin with stale targets). None qualifies: `set_pin(key, "P0", status="no-savings")`, `winner = "P0"` — a re-sweep that finds nothing must stop a previously pinned profile from serving. A route that fails the eligibility gate is likewise reset to P0 with `status = not-applicable`. In both cases `sweeps.result_json` holds the per-profile table (rate, floor, stops, cache_read, $/request, qualifies, reason). `--dry-run` runs the replays and judge calls (it spends the same money), prints that table, and writes nothing — no `sweeps`, `judgments`, or pin changes. To preview only the cost, use `--budget-usd 0`, which prints the estimate and refuses.

## 8. Cost gate and budget

`estimate_cost` prices `items × trials × profiles` replays from the route's mean P0 input/output tokens and the price table, plus judge calls (`items × trials × (candidates + noise pairs)` at ~ (4,000 + 2 × mean output) input and 8 output tokens of the judge model). The sweep refuses (exit 2, nothing written) when the estimate exceeds the ceiling: `--budget-usd` when given (an explicit per-run ceiling that overrides the monthly one), otherwise `sweep_budget_usd_month − sum(sweeps.cost_usd this month)`. Default `sweep_budget_usd_month = 0` means refuse until the operator sets a ceiling. Actual spend is re-checked against the ceiling every 50 calls; exceeding it aborts (exit 1, `finished_at NULL`, no pin).

## 9. Recheck and drift

`recheck`: for each pinned route, take up to `--n` (default 20) most recent requests with `profile = pinned_profile` and `body_ref`, replay each once at P0 with the CLI key, judge live-vs-P0 (live is the candidate), write one `shadow` row per pair (`label`, `request_id`). If the route has at least 20 judged (non-`judge-error`) `shadow` rows and the rolling equivalence rate over its last 100 is below the bar: `set_pin(key, "P0", status="reverted")`. Every `set_pin` deletes the route's `shadow` rows, so the window only ever holds rows written under the current pin. A P0 replay that fails at the transport level writes no row for that pair. Recheck never pins. Recheck spend (≤ n replays + n judge calls) is operator-invoked and bounded by `--n`; like `--dry-run` spend it is not counted against the monthly sweep ceiling.

Hash-change unpin (proxy, keyless): `db.upsert_route`'s conflict clause becomes
`DO UPDATE SET last_seen=excluded.last_seen, model=excluded.model, system_hash=excluded.system_hash, pinned_profile=CASE WHEN routes.system_hash != excluded.system_hash OR routes.model != excluded.model THEN 'P0' ELSE routes.pinned_profile END, status=CASE WHEN routes.system_hash != excluded.system_hash OR routes.model != excluded.model THEN 'observing' ELSE routes.status END` — a model switch under an `X-Optimizer-Route` name unpins too, since profiles are tuned per model.

## 10. Report and audit

`route_rows` gains `equivalence_pct`, `noise_floor_pct`, `sample_n`, `last_sweep_at`, `sweep_cost_usd`, `recheck_pct` (rolling 100). New endpoint `GET /optimizer/sweeps/{route_key}` returns that route's `sweeps` rows including `result_json`. HTML report shows the new columns.

## 11. Errors

- One failed replay fails that item for that profile only.
- Transport failures on > 20% of a profile's calls, or any judge outage, abort the sweep: `sweeps` row left with `finished_at NULL`, `winner NULL`; nothing pinned; exit 1.
- Budget exceeded mid-sweep aborts the same way.
- Rechecks that cannot complete write no `shadow` rows and never revert.

## 12. Testing

- Unit (pure): eligibility gate; stratified sampling incl. short quintiles; `estimate_cost`; `pin_rule` for each disqualifier, noise-floor-limited, no-savings, cheapest-wins; judge request building, label parsing, order normalization; `[prices]` merge.
- `run_sweep` end to end against an `httpx.MockTransport` scripting P0/candidate replies and judge labels: asserts the pin, `target_words`/`exemplar`, and the `sweeps`/`judgments` rows; dry-run writes no pin; budget refusal before any call.
- `recheck` revert against a mock; `upsert_route` hash-change unpin (db test).
- CLI: subcommand parsing, missing-key refusal, exit codes.
- Live E2E (gated on `OPTIMIZER_LIVE=1` + keys): push a 50-item synthetic support-ticket route through the proxy, then `sweep --route`; assert a profile pins with ≥ 25% fewer output tokens and equivalence ≥ 0.95. This is spec §11's demo.

## 13. Success criteria

- Live demo route pins at ≥ 25% fewer output tokens with equivalence ≥ 0.95.
- A route with no qualifying profile reports `no-savings` with the full table.
- A sweep under a zero budget refuses before the first call.
- Every pinned route has a readable sweep behind it at `/optimizer/sweeps/{key}`.
- The proxy binary is unchanged in behaviour except hash-change unpin and the new report fields.
