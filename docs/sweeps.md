# Sweeps, drift and audit

The full reference for `pith sweep` and `pith recheck`. The short version is in the README; the reasoning behind the
rules is in [methodology.md](methodology.md).

The proxy never holds an API key, so sweeps run from the CLI with keys in its environment:

    export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY
    .venv/bin/python -m pith sweep --config pith.toml --dry-run     # spends, prints, writes nothing
    .venv/bin/python -m pith sweep --config pith.toml               # pins the cheapest profile that clears the bar

A route is swept once it has 50 sampled baseline requests (`sample_rate` controls sampling) and ≥80% text-ending
responses. The sweep replays the frozen sample under each profile, judges equivalence against the unconstrained
baseline (`judge_model`), and pins only a profile that is at least as consistent as the baseline is with itself. `equivalence_bar` is an absolute floor on quality, but it is capped by the route's own self-consistency: if the unconstrained model agrees with itself only 85% of the time, a profile that also reaches 85% qualifies. Routes whose self-consistency is under 50% are never pinned. Judgments are taken per item by majority across trials (use an odd `--trials`, default 3), every pair of baseline trials is judged for the noise floor, and the comparison tolerance widens with the sweep's sampling error.
A pin is tied to the shape-text version it was validated under; after a pith upgrade that changes the text, the route drops its pin on the next request and goes back to observing until it is swept again.
`sweep_budget_usd_month = 0` (the default) refuses every sweep; set a ceiling, or pass `--budget-usd` per run.
Sweep flags: `--route <key>` (one route; also sweeps a pinned one), `--trials N` (replays per item, default 3),
`--sample N` (items per sweep, default 50), `--dry-run`, and `--budget-usd X`, a ceiling for the whole run: each
swept route draws it down and a route whose estimate no longer fits is refused. `--holdout F` (0 ≤ F < 0.5, default 0)
holds back a fraction of the sample; the winner is replayed and judged on it alone and pinned only if it clears the same
rule there (reason `failed holdout: …` otherwise). Held items whose request also appears in the fit set are dropped, so
the holdout is out-of-sample; its bar allows the same sampling-error tolerance as the noise-floor check, and a holdout
that cannot be judged (winner a no-op there, or no baseline consistency) fails rather than pins. The printed table ends
with a holdout line when a holdout was requested (`not run (…)` if there was no winner or no out-of-sample item), and
the stored result carries the holdout's own P0/winner table; its spend is amortized like the rest of the sweep. `--dry-run` and `recheck` spend is not
counted against `sweep_budget_usd_month`; only live sweeps are. `recheck --route <key> --n N` re-judges the last N live responses.
Exit codes: 0 done, 2 refused (budget, price, or missing key), 1 aborted.
A sweep's own cost is amortized over the route's projected monthly volume (last 7 days × 30/7, floor 1,000 requests) and
added to every candidate's $/request, so a profile is pinned only if it repays the sweep within a month. On a tiny route
the table will say `sweep cost not recovered at projected volume` even for a profile that is cheaper per request —
that is the honest answer, not a failure.

Drift: `python -m pith recheck` re-judges recent live responses on pinned routes and reverts a route to P0 when
its rolling equivalence falls below the same floor-capped bar the pin used (`min(bar, noise floor)`, with the floor read from the route's last sweep and the current configured bar used;
the configured bar alone when no sweep with a numeric floor is on record), less the same sampling tolerance the pin allows. Run both from cron, e.g. a nightly `recheck` and a weekly `sweep`.

Audit: `GET /optimizer/sweeps/<route key>` returns every sweep for a route with its per-profile table.
Name a route explicitly with the request header `X-Optimizer-Route: <name>`.

The proxy has no authentication of its own — bind `listen` to a private interface. On first use of an OpenAI stream
without usage, token estimation downloads the `o200k_base` vocabulary once (set `TIKTOKEN_CACHE_DIR` to pre-seed it on
egress-filtered hosts; if the download fails the proxy falls back to a length estimate and flags the row as estimated).
