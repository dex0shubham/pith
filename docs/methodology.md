# How pith decides a pin

## The problem

Output tokens cost 5x input tokens on current Claude models.
Providers ship knobs that affect length: `effort` on Claude, `reasoning_effort` and `verbosity` on OpenAI.
Nothing decides which knob, if any, a given route can take without losing quality.
Input compression leaves the output untouched.
pith measures one route at a time and changes it only when the measurement supports it.

## Routes

Its key is provider, model, and a hash of the whitespace-normalised system prompt plus the sorted tool definitions.
A changed prompt or model therefore produces a new route, which starts unpinned.
A client can name a route with the `X-Optimizer-Route` header instead, and then the key stays fixed.
If a named route's prompt hash or model changes, the proxy drops its pin and returns it to observing.

Routes that are `observing` or `reverted` are swept automatically; `pinned`, `no-savings` and `not-applicable` ones only with `--route`.

## Profiles

A profile is a fixed rewrite applied to every request on a route (`pith/rewrite.py`).

- **P0** passes the request through. It is the baseline.
- **P1** steps effort down one notch. On a model without an effort parameter it changes nothing.
- **P1b** (OpenAI only) sets `verbosity` to `low`.
- **P2** appends the shape text: "Be concise: no preamble, restatement, or closing summary. Use at most N words unless the task genuinely needs more." N is half the median word count of the baseline answers, minimum 20.
- **P3** is P2 plus one example: the shortest baseline answer.
- **P4** is P1 and P2 together.

pith never touches `max_tokens` or its OpenAI equivalents.
A lower ceiling does not shorten the answer the model plans; it cuts it off.
The rewrite also leaves `system`, `tools`, `model`, `thinking` and `temperature` alone; by the design's reasoning they sit at the front of the prompt-cache prefix or define the task.

The standalone proxy appends the shape text after the last user message, as a `system` message on Claude or a `developer` message on OpenAI.
Appending at the end leaves the cached prefix intact.
When a provider rejects a rewritten request with 400 or 422, the proxy retries the original.
If the original succeeds, a rejected mid-conversation `system` message switches the route to the user-text form: a text part on the last user message.
Any other such rejection is counted, and the third reverts the route to P0.
The LiteLLM guardrail counts and reverts the same way; Headroom counts only effort rewrites; Portkey hooks never see rejections.

The LiteLLM and Portkey integrations use only the user-text form, and so apply only P2 and P3.
LiteLLM folds system messages into the provider's top-level system prompt, which would break the cache.
A gateway hook also cannot retry a rejected request.

## The sweep

`run_sweep` in `pith/sweep.py` makes the decision.

### Sample

A route is eligible once it has 50 sampled baseline requests and at least 80% of its baseline responses end in text.
A route with at least 50 baseline requests that fails the 80% gate is marked `not-applicable`; below 50 it is simply not yet eligible.
Each sweep picks its sample afresh from the 500 most recent text-ending baseline requests that still have stored bodies.
The pick is stratified across five output-length quintiles, newest first within each, and the chosen ids are stored with the sweep.
Requests carrying server-side tools are excluded, because a replay would execute and bill them.
With fewer than half the requested items available, the sweep refuses.

### Replays

Each item is replayed under P0 `--trials` times (default 3), then as often under each other profile.
P0's first trial is the baseline answer for every comparison.
An error status, a token-limit stop or an empty reply fails mechanically, with no judge call.
A failed candidate trial is labelled `format-broken`.
A failed baseline trial, or a failure in either trial of a P0 pair, is labelled `judge-error`.

### The judge

The judge (`pith/judge.py`) sees the last user message and two answers in slots A and B.
It replies with one of six slot-qualified labels: `equivalent`, `A-omits`, `B-omits`, `contradiction`, `A-broken`, `B-broken`.
The labels are symmetric, so the judge has to check both answers for omissions.
Slot order is randomised per judgment, against position bias.
The label is then mapped to the candidate's side: an omission by the candidate is `missing-info`, one by the baseline is `extra-info`.
Only `equivalent` counts as a pass.
A label parses only when it is the whole of the first or last non-empty line, apart from markdown characters and a trailing period.
A label inside prose, such as "not equivalent" or "Label: equivalent", never parses.
An unparseable reply is retried once, then recorded as `judge-error`.
`judge-error` is left out of rates but counts toward the 20% limit below.

### Per-item majority and the noise floor

`item_scores` scores an item 1 when more than half its non-error judgments are `equivalent`, 0.5 on a tie, 0 otherwise.
A profile's rate is the mean item score, with its standard error.

Every pair of P0 trials is also judged: three pairs per item at three trials.
The same majority over those pairs gives the noise floor, the rate at which the unconstrained model agrees with itself.

### The pin rule

`pin_rule` qualifies a profile when all of these hold:

1. Its rate is at least `min(equivalence_bar, floor)`. The bar defaults to 0.95, can be overridden per route, and is capped by the floor.
2. Its rate is at least `floor − tol`, with `tol = max(0.03, √(se_profile² + se_floor²))`.
3. It has no more token-limit stops than P0.
4. Its mean cache reads are not below P0's. This applies only when P0's mean is above zero.
5. Its cost per request, with the sweep amortised in, is below P0's.
6. At most 20% of its judgments are `judge-error`.

A route whose floor is below 0.5 pins nothing: the judge signal is too weak.
The cheapest qualifying profile wins.
If none qualifies, the route is set to P0 with status `no-savings`.

### Refusal reasons

`rate below bar` and `rate below noise floor tolerance` fail rules 1 and 2.
`more max_tokens stops than P0`, `cache reads below P0` and `judge mostly errored` fail rules 3, 4 and 6.
`not cheaper than P0` means a higher raw cost per request.
`sweep cost not recovered at projected volume` means cheaper, but not enough to repay the sweep within a month.
`route too noisy to judge` means a floor below 0.5; `noise floor undefined` means every P0 pair errored.
`skipped` means the rewrite changes no request; `same request as Pn on this model` marks an alias.
`failed holdout: …` means the winner did not hold on the held-out items.

### Aliases

On `claude-haiku-4-5`, which has no effort parameter, P4 is byte-identical to P2.
The sweep judges an identical request once and reports the duplicate as an alias.

### Cost and budget

The sweep refuses to start if its estimated cost exceeds the ceiling.
The ceiling is the monthly budget minus this month's sweep spend; `--budget-usd` replaces it with a per-run ceiling.
The default monthly budget is 0, so nothing runs until an operator sets one.
The gate applies to `--dry-run` as well.
A sweep aborts if spend passes the ceiling, checked every 50 calls.

### Amortisation

At decision time, the sweep's spend so far plus the estimated holdout cost is divided by the route's projected monthly volume and added to every candidate's cost per request.
Projected volume is the last seven days of requests times 30/7, floored at 1,000.
After the holdout, the stored table is re-amortised with the actual spend.

### The holdout

`--holdout F`, with F at least 0 and below 0.5, keeps a fraction of the sample out of the decision.
The held items are the tail of the stratified list: the oldest picked items of each quintile, not a random draw.
Held items whose request also appears in the fit set are dropped.
If the fit set yields a winner, P0 and the winner are replayed on the held items, with their own noise floor.
The winner pins only if it passes the pin rule there, with a tolerant bar (`rate ≥ min(bar, holdout floor) − tol`) and raw, unamortised cost.
An undefined holdout floor, or a winner whose rewrite changes no held item, fails the holdout.
If every held item duplicates a fit-set prompt, the holdout is reported as not run and the fit-set winner pins unconfirmed.

## Why it refuses

The runs below are from the README's live results, on `claude-haiku-4-5` with three trials and judge `claude-sonnet-5-5`.
The bar was 0.9, set in the live demo's configuration (`tests/live/test_sweep_demo.py`).
Every run except the first used `--holdout 0.3`, which is why the fit sets hold 70 and 139–140 items.

### 30 repeated tickets: P2 pinned

On thirty requests from ten repeated support tickets, P2 matched the floor of 0.933 ± 0.046 exactly, at 78 output tokens per request against 160 (−51%) and $0.000579 against $0.000860 (−33%).
P3 qualified at 0.900 ± 0.056 and $0.000804, cheaper than P0 but dearer than P2.

### 100 and 200 distinct tickets: refused

The floor fell as the prompt set widened: 0.871 ± 0.040 on 70 fit items, 0.829 ± 0.032 on 140.
P2 cut output tokens by 53% both times, but scored 0.829 ± 0.045 and 0.779 ± 0.035.
Both floors were under 0.9, so the bar equalled the floor, and P2 failed with `rate below bar`.
P3 did worse.
Nothing won on the fit set, so the holdout never ran.
The gap was inside the sampling error at 70 items and at its edge at 140, in the same direction both times.
The refusal means the shape text halves output here but is probably slightly less faithful than the baseline is to itself.

### Classification, first wording: refused

The classification route asks for one of five labels, with no length hint.
Unprompted, the model gives the label and a sentence of justification.
On 100 tickets, P2 scored 0.929 ± 0.031 against a floor of 0.971 ± 0.020 and missed the floor's tolerance.
On 200 tickets, P2 fell to 0.885 ± 0.027, below the bar.
Of 417 P2 trials in that run, 24 were `extra-info`, 19 `contradiction` and 10 `missing-info`.
The shape text then began "Answer directly", which nudged the model to answer the ticket itself, sometimes with invented steps, or to change the label.

### Classification, shape text v2: held by economics

The shape text was rewritten to ask only for concision, and the 200-ticket run was repeated.
`extra-info` fell from 24 to 4 and `missing-info` from 10 to 3.
P2 still disagreed on the label in 28 `contradiction` trials, against 10 in the baseline's own noise pairs.
At 0.906 ± 0.025 against a floor of 0.942 ± 0.020, it missed the floor's tolerance by 0.004.
P3, the label plus a one-line justification, scored 0.942 ± 0.020 at 24 output tokens against 42 (−44%).
It passed every quality condition and failed on `sweep cost not recovered at projected volume`.
A $2.49 sweep cannot be repaid by a $0.00003-per-request saving at the demo's simulated 13k requests a month.
At roughly 80k requests a month the same numbers pin P3.

## What a pin promises

A pin says that, on a sample of the route's own traffic, the profile agreed with the baseline about as often as the baseline agreed with itself.
It is a relative claim made by one judge model, not correctness against ground truth: if the baseline is wrong, an equivalent profile is wrong the same way.

### Drift

`recheck` replays recent live requests of a pinned route at P0 and judges the live answer against the fresh baseline.
With at least 20 judged rows, a rolling rate over the last 100 below the bar reverts the route to P0.
Recheck currently uses the configured bar, not the floor-capped one, so a route pinned with a floor below the bar can revert at the quality it was pinned at.

### Pin version

Every pin records the shape-text version it was validated under, currently `v2` (`SHAPE_VERSION`).
If an upgrade changes the text, the proxy drops the pin on the route's next request.

## Known limits

- **Judge context.** The judge sees only the last user message, truncated to 4,000 characters, not the system prompt or earlier turns.
- **Judge cost and noise.** Every candidate trial and every P0 pair is one judge call. Across 30-item runs the floor swung between 0.80 and 0.97.
- **Recheck is thin.** One comparison per request, no majority vote.
- **Aborts.** A sweep aborts, pinning nothing, when more than 20% of a profile's calls fail at the transport level or the judge is unavailable.
- **Tool use.** Tool-heavy routes rarely pass the 80% text gate.
- **Streaming through Portkey.** Portkey delivers no body for streamed responses, so pith does not record them. Provider rejections are invisible to its hooks, which leaves `recheck` as the only drift guard there.
- **The fixed cost of a sweep.** The 200-ticket runs cost $2.49 to $5.99 and took 56 minutes to 1 h 50 min. A low-volume or cheap route does not repay that.
