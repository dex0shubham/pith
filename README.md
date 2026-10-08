# Output-Token Optimizer

Self-hosted drop-in proxy for the Claude and OpenAI APIs. Passes traffic through byte-for-byte, fingerprints routes,
records usage, and — once a route has a pinned output profile — rewrites requests cache-safely to shorten outputs.
Design: `docs/design/specs/2026-10-07-output-token-optimizer-design.md`.

## Run

    python3 -m venv .venv && .venv/bin/pip install -e .
    cp pith.example.toml pith.toml
    .venv/bin/python -m pith serve --config pith.toml

Point your client at it and keep your own API key:

    ANTHROPIC_BASE_URL=http://localhost:8787   # Anthropic SDKs
    OPENAI_BASE_URL=http://localhost:8787/v1   # OpenAI SDKs

## Kill switches

- Per request: header `X-Optimizer: off` (forces P0). `X-Optimizer: bypass` also skips recording.
- Per route: `[routes."<key>"] enabled = false` in `pith.toml`.
- Global: `OPTIMIZER_ENABLED=0`.

Any proxy-side failure forwards your original request unchanged.

## Sweeps: turning observation into pins

The proxy never holds an API key, so sweeps run from the CLI with keys in its environment:

    export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY
    .venv/bin/python -m pith sweep --config pith.toml --dry-run     # spends, prints, writes nothing
    .venv/bin/python -m pith sweep --config pith.toml               # pins the cheapest profile that clears the bar

A route is swept once it has 50 sampled baseline requests (`sample_rate` controls sampling) and ≥80% text-ending
responses. The sweep replays the frozen sample under each profile, judges equivalence against the unconstrained
baseline (`judge_model`), and pins only a profile that is at least as consistent as the baseline is with itself.
`sweep_budget_usd_month = 0` (the default) refuses every sweep; set a ceiling, or pass `--budget-usd` per run.
Sweep flags: `--route <key>` (one route; also sweeps a pinned one), `--trials N` (replays per item, default 3),
`--sample N` (items per sweep, default 50), `--dry-run`, and `--budget-usd X`, a ceiling for the whole run: each
swept route draws it down and a route whose estimate no longer fits is refused. `--dry-run` and `recheck` spend is not
counted against `sweep_budget_usd_month`; only live sweeps are. `recheck --route <key> --n N` re-judges the last N live responses.
Exit codes: 0 done, 2 refused (budget, price, or missing key), 1 aborted.

Drift: `python -m pith recheck` re-judges recent live responses on pinned routes and reverts a route to P0 when
its rolling equivalence falls below the bar. Run both from cron, e.g. a nightly `recheck` and a weekly `sweep`.

Audit: `GET /optimizer/sweeps/<route key>` returns every sweep for a route with its per-profile table.
Name a route explicitly with the request header `X-Optimizer-Route: <name>`.

The proxy has no authentication of its own — bind `listen` to a private interface. On first use of an OpenAI stream
without usage, token estimation downloads the `o200k_base` vocabulary once (set `TIKTOKEN_CACHE_DIR` to pre-seed it on
egress-filtered hosts; if the download fails the proxy falls back to a length estimate and flags the row as estimated).

## Report

`GET /optimizer/report` (JSON) · `GET /optimizer/report.html`

## Tests

    .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest
    OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live   # cache-safety check (~$0.02) and sweep demo (~$1-3)
