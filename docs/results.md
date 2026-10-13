# Live results

Every run below was made against the real Anthropic API with `tests/live/test_sweep_demo.py` (knobs in
[development.md](development.md)); the decision rules are in [methodology.md](methodology.md). The runs are in the order
they were made, and each one changed something: a wording, a rule, or what the README claims.

Measured against the real Anthropic API on 2026-10-10 with `tests/live/test_sweep_demo.py`: a support-ticket
classification route on `claude-haiku-4-5` (30 real requests through the proxy, then a sweep at `--trials 3`,
judge `claude-sonnet-5-5`). The sweep cost about $1.3 and took 24 minutes.

| Profile | What it does | Equivalence (per-item majority) | Output tokens / request | $/request* | Verdict |
|---|---|---|---|---|---|
| P0 | unconstrained baseline | 0.933 ± 0.046 (self-consistency = noise floor) | 160 | $0.000860 | baseline |
| P1 | effort one notch down | — | — | — | skipped (no `effort` on this model) |
| **P2** | **terse-shape instruction** | **0.933 ± 0.046** | **78 (−51%)** | **$0.000579 (−33%)** | **pinned** |
| P3 | shape + one-shot exemplar | 0.900 ± 0.056 | 99 (−38%) | $0.000804 | qualifies, costlier than P2 |
| P4 | effort down + shape | — | — | — | same request as P2 on this model |

\* Haiku 4.5 at $1/$5 per million tokens, including the sweep's own cost amortized over projected monthly volume.

The pinned instruction matches the unconstrained model's agreement with itself exactly, at 51% fewer output tokens
and 33% lower cost per request, on one 30-item run; a larger run with a holdout is the next step, and `recheck`
re-judges live traffic after a pin and reverts on drift. An earlier run (2026-10-08) of the same demo reported "P4
pinned at −56%": Haiku has no `effort` parameter, so P4's request was byte-identical to P2's and the sweep judged the
same request twice, once at 0.833 and once at 0.933, and pinned on the second. The sweep now judges identical
effective requests once and reports the duplicate as an alias, which is the P4 row above.

Two larger runs on distinct tickets with a 30% holdout requested (`OPTIMIZER_DEMO_HOLDOUT=0.3`) did not pin anything,
and those are the more informative results:

| Run | Profile | Equivalence on the fit set | Output tokens / request | Verdict |
|---|---|---|---|---|
| 100 tickets, fit 70 ($2.99, 55 min) | P0 | 0.871 ± 0.040 (noise floor) | 171 | baseline |
| | P2 | 0.829 ± 0.045 | 80 (−53%) | rate below bar |
| | P3 | 0.786 ± 0.049 | 85 (−50%) | rate below bar |
| 200 tickets, fit 140 ($5.99, 1 h 50 min) | P0 | 0.829 ± 0.032 (noise floor) | 175 | baseline |
| | P2 | 0.779 ± 0.035 | 82 (−53%) | rate below bar |
| | P3 | 0.671 ± 0.040 | 89 (−49%) | rate below bar |

P4 was the same request as P2 in both runs and was not judged again. With diverse prompts the unconstrained model agrees
with itself less often than on the ten repeated tickets (0.93 → 0.87 → 0.83 as the prompt set widens), and the terse
instruction lands 0.04–0.05 below that floor each time: inside the sampling error at n=70, at the edge of it at n=140,
and in the same direction both times. So on this route the shape instruction halves output tokens but is probably
slightly less faithful than the baseline is to itself, and the one-shot exemplar is clearly worse. The sweep refused to
pin on that evidence, which is the behaviour it should have, and the holdout never ran because nothing reached it.
Read the three runs together: the 30-item pin above is what ten repeated tickets look like, and the larger runs are
what a real route looks like.

A short-answer route tells the other half of the story, and the stored judgments say exactly where the loss is. Same
tickets, system prompt "Classify each ticket as one of: shipping, refund, account, product, other" with no length hint
(`OPTIMIZER_DEMO_ROUTE=classify`). Unprompted, Haiku answers with the label and a sentence of justification. Two runs:

| Run | Profile | Equivalence on the fit set | Output tokens / request | $/request (raw) | Verdict |
|---|---|---|---|---|---|
| 100 tickets, fit 70 ($1.30, 28 min) | P0 | 0.971 ± 0.020 (noise floor) | 38 | $0.000237 | baseline |
| | P2 | 0.929 ± 0.031 | 9 (−77%) | $0.000122 (−48%) | rate below noise floor tolerance |
| | P3 | 0.957 ± 0.024 | 27 (−29%) | $0.000244 | not cheaper than P0 |
| 200 tickets, fit 139 ($2.56, 56 min) | P0 | 0.971 ± 0.014 (noise floor) | 39 | $0.000244 | baseline |
| | P2 | 0.885 ± 0.027 | 13 (−67%) | $0.000143 (−41%) | rate below bar |
| | P3 | 0.914 ± 0.024 | 23 (−43%) | $0.000215 | rate below noise floor tolerance |

Reading the first 200-item run's judgments (`OPTIMIZER_DEMO_DB` keeps them): of 417 P2 trials, 24 were `extra-info`,
19 `contradiction` and 10 `missing-info`, concentrated on 16 of 139 tickets. Replaying three of them showed the
mechanism: the shape text then began "Answer directly", and on a classification route that nudged the model to
*answer the ticket* (for "How do I delete my account and all my data?" the terse reply was the label followed by
invented steps) or to change the label. The shape text was changed to ask only for concision (shape text v2; pins are
tied to the text version they were validated under), and the same 200-item run was repeated ($2.49, 58 minutes):

| Profile | Equivalence on the 140-item fit set | Output tokens / request | $/request (raw) | Verdict |
|---|---|---|---|---|
| P0 | 0.942 ± 0.020 (noise floor) | 42 | $0.000256 | baseline |
| P2 | 0.906 ± 0.025 | 10 (−77%) | $0.000126 (−51%) | rate below noise floor tolerance |
| P3 | 0.942 ± 0.020 | 24 (−44%) | $0.000223 (−13%) | sweep cost not recovered at projected volume |

The wording fix removed the failure it targeted: `extra-info` fell from 24 to 4 and `missing-info` from 10 to 3. What
remains for the label-only P2 is label disagreement (28 `contradiction` trials, against 10 inside the baseline's own
noise pairs), and it misses the floor's tolerance by 0.004. P3, label plus a one-line justification, now matches the
baseline's self-consistency exactly at 44% fewer output tokens and qualifies on every quality condition; it is held
back only by economics: a $2.49 sweep cannot repay a $0.00003-per-request saving at the demo's simulated 13k
requests a month, and the table says so. At roughly 80k requests a month the same numbers pin P3, which is the
honest shape of the product on a cheap route: the sweep is a fixed cost, so it pays on volume.

A JSON-extraction route was tried as a candidate for an easy win (same tickets, "extract a JSON object with intent,
product, order_number and urgency", no format hint; `OPTIMIZER_DEMO_ROUTE=extract`, 200 tickets, 30% holdout
requested; $2.44, 56 minutes) and gave none, for a reason worth recording: unprompted, Haiku answers every one of the
200 tickets with a fenced JSON object and nothing else (about 47 output tokens), so there is no prose for a shape
instruction to remove. P2 saved 5% of output tokens and scored 0.700 ± 0.039 against a baseline self-consistency of
0.793 ± 0.034, because the instruction perturbs the model's field choices (`urgency`, `product` null or named) on
tickets where the baseline is itself unstable; P3 did the same at 0.679 ± 0.040. The sweep refused both. The lesson is
the mirror image of the classification route: a shape profile only pays where the baseline spends tokens on
something the task does not need, and the sweep is what tells you whether that is the case.

Three public datasets were then run the same way (200 prompts each, 30% holdout requested, `--trials 3`,
`OPTIMIZER_DEMO_DATASET=samsum|dolly|bitext`, 2026-10-12): SAMSum dialogue summarisation ("Summarize the
conversation."), Dolly 15k summarisation ("Answer the request using the provided text.") and Bitext customer-support
queries ("You are a customer support assistant for an online store. Reply to the customer."). On all three, P2 cut
output tokens by about half and the sweep refused it, because the judge found the shorter answers missing information
far more often than the baseline misses against itself:

| Route | P0 tokens / floor | P2 tokens / rate | Judged missing-info, P0 vs P2 | Cost, time |
|---|---|---|---|---|
| SAMSum | 102 / 0.841 ± 0.032 | 48 (−53%) / 0.641 ± 0.042 | 7% vs 30% | $5.61, 1 h 29 min |
| Dolly summarisation | 184 / 0.768 ± 0.038 | 83 (−55%) / 0.448 ± 0.045 | 6% vs 45% | $6.20, 1 h 31 min |
| Bitext support | 189 / 0.721 ± 0.038 | 88 (−54%) / 0.507 ± 0.042 | 9% vs 30% | $6.31, 1 h 54 min |

Contradictions stayed at the baseline's own level (4 to 6%), so this is omission, not re-steering. For a summary that
is what a word cap does by construction: the content of a summary is the detail it keeps, so an answer half the
length is an answer with half the details, and a judge that asks "does this say everything the other one says" is
right to fail it. The same holds for the support replies, which drop steps rather than pleasantries. P3 did slightly
worse than P2 on all three. None of the holdouts ran, as there was no winner on the fit set to confirm. Together with
the routes above this makes seven routes on `claude-haiku-4-5` with one pin, on the repeated-ticket route where the
baseline restated the ticket; the shape profiles save tokens by dropping content, and on this model there is rarely
content worth dropping.

The effort profiles (P1, P4: one step down on `output_config.effort`) need a model that has the parameter, so GSM8K
(`OPTIMIZER_DEMO_DATASET=gsm8k`, "Solve the problem. Reply with only the final number.", 200 problems, 30% holdout
requested, 2026-10-12) was run on `claude-haiku-5-5` and `claude-sonnet-5-5`, both of which think adaptively and bill
the thinking as output tokens. The judge compares the visible answer, a number, so the floor is close to 1 and every
profile passed on quality on both models; the saving is thinking tokens, not dropped content:

| Model, profile | Rate (floor) | Output tokens | Cost per 1,000 requests | Sweep |
|---|---|---|---|---|
| Haiku 5.5, P0 | 0.971 ± 0.014 | 138 | $0.080 | $4.47, 2 h 01 min |
| Haiku 5.5, P1 effort low | 0.993 ± 0.007 | 106 (−23%) | $0.064 | |
| Haiku 5.5, P4 effort low + concise | 0.957 ± 0.017 | 90 (−35%) | $0.060 | |
| Sonnet 5.5, P0 | 0.993 ± 0.007 | 123 | $1.44 | $7.13, 2 h 12 min |
| Sonnet 5.5, P2 concise | 0.986 ± 0.010 | 81 (−34%) | $1.11 | |
| Sonnet 5.5, P4 effort low + concise | 0.979 ± 0.012 | 74 (−40%) | $1.04 | |

Neither pinned, and both refusals read "sweep cost not recovered at projected volume". The demo projects about
10,500 requests a month. On Haiku 5.5 the whole route costs $0.84 a month at that volume and the best profile saves
21 cents of it, against a $4.47 sweep that is almost entirely Sonnet 5.5 judge calls; break-even is around 220,000
requests a month. On Sonnet 5.5 the best profile saves 28% of the route's cost and the $7.13 sweep repays itself from
about 18,000 requests a month, so the same route at twice the demo's volume pins P4. These are the first runs where
the candidates were as good as the baseline, and the lesson is the complement of the one above: the effort lever
saves tokens the task really does not need, and whether a sweep is worth running is then a question of volume and of
how much the judge costs relative to the model being judged.

The same Sonnet 5.5 route was then rerun with the demo's volume set to 25,000 requests a month
(`OPTIMIZER_DEMO_VOLUME=25000`, an ordinary volume for a production route and above the 18,000 break-even; everything
else unchanged, 2026-10-13). It pinned P4, the first pin on a public dataset, and the 30% holdout confirmed it on 60
problems the fit set had not seen. A second rerun swapped the judge for `claude-haiku-5-5`
(`OPTIMIZER_DEMO_JUDGE`) to see what a judge twenty times cheaper does to the decision:

| Judge | Fit: floor, P4 rate, P4 tokens | Holdout (60): floor, P4 rate, P4 tokens | Sweep cost, time | Outcome |
|---|---|---|---|---|
| Sonnet 5.5 | 0.979 ± 0.012, 0.986 ± 0.010, 75 (−37%) | 0.983, 1.000, 72 (−37%) | $8.38, 2 h 22 min | P4 pinned, $1.43 → $1.06 per 1,000 requests |
| Haiku 5.5 | 0.979 ± 0.012, 0.986 ± 0.010, 75 (−40%) | 0.933, 0.933, 72 (−38%) | $3.31, 2 h 20 min | P4 pinned, same saving |

The cheaper judge reached the same pin for 60% less, but it is a noisier and stricter judge: its holdout floor was
0.933 against Sonnet's 0.983, and it marked 28 of the 420 P2 answers as missing information where Sonnet marked none,
which pushed P2 and P3 below its noise-floor tolerance. On a numeric task the difference did not change the outcome;
on the summarisation routes above, where the floor is already 0.72 to 0.84, a judge that noisy would leave no room
to decide anything. The default stays Sonnet 5.5, and `judge_model` is the knob when a route's volume justifies a
sweep but not a Sonnet-judged one.

The cache-safety check (`tests/live/test_cache_safety.py`) passed in the same session: a P2-pinned route on
`claude-opus-5-5` still reported `cache_read_input_tokens > 0` on the second request, so the rewrite does not
re-bill the customer's prompt cache.

What the earlier failing runs showed, and what changed because of them: haiku rejects the mid-conversation `system`
message the shape profiles inject (the sweep now falls back to the user-text form like the proxy does); the judge
reasons in prose and ends with the bare label (the parser now reads the first or last standalone line); a $1 sweep
cannot repay itself on a 30-request route (the table says so instead of "not cheaper"); and across runs the noise
floor swung 0.80–0.97 on 30 items, so the absolute `equivalence_bar` is capped by the floor and the comparison
tolerance follows the sampling error.
