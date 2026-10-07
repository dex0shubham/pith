# Output-Token Optimizer (data plane)

Self-hosted drop-in proxy for the Claude and OpenAI APIs. Passes traffic through byte-for-byte, fingerprints routes,
records usage, and — once a route has a pinned output profile — rewrites requests cache-safely to shorten outputs.
Design: `docs/design/specs/2026-10-07-output-token-optimizer-design.md`.

## Run

    python3 -m venv .venv && .venv/bin/pip install -e .
    cp optimizer.example.toml optimizer.toml
    .venv/bin/python -m optimizer --config optimizer.toml

Point your client at it and keep your own API key:

    ANTHROPIC_BASE_URL=http://localhost:8787   # Anthropic SDKs
    OPENAI_BASE_URL=http://localhost:8787/v1   # OpenAI SDKs

## Kill switches

- Per request: header `X-Optimizer: off` (forces P0). `X-Optimizer: bypass` also skips recording.
- Per route: `[routes."<key>"] enabled = false` in `optimizer.toml`.
- Global: `OPTIMIZER_ENABLED=0`.

Any proxy-side failure forwards your original request unchanged.

## What you will see at first

Plan 1 is observe-only: the proxy fingerprints routes and records usage but never rewrites a request until a route has a pinned profile. Pins come from the control plane (Plan 2). To try a profile by hand on one route:

    sqlite3 optimizer.db "UPDATE routes SET pinned_profile='P2', status='pinned' WHERE key='<route key from /optimizer/report>'"

Name a route explicitly with the request header `X-Optimizer-Route: <name>`.

The proxy has no authentication of its own — bind `listen` to a private interface. On first use of an OpenAI stream without usage, token estimation downloads the `o200k_base` vocabulary once (set `TIKTOKEN_CACHE_DIR` to pre-seed it on egress-filtered hosts; if the download fails the proxy falls back to a length estimate and flags the row as estimated).

## Report

`GET /optimizer/report` (JSON) · `GET /optimizer/report.html`

## Tests

    .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest
    OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live   # live cache-safety check
