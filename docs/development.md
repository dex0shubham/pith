# Development and tests

Unit tests run offline in a few seconds. The live tests spend real money and are opt-in.

    .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest
    OPTIMIZER_LIVE=1 ANTHROPIC_API_KEY=... .venv/bin/pytest tests/live   # cache-safety check (~$0.02) and sweep demo (~$1.3, ~30 min)

For repeated live runs keep the key in a git-ignored `.env` (`ANTHROPIC_API_KEY=...`, `chmod 600`) and run
`set -a; . ./.env; set +a; OPTIMIZER_LIVE=1 .venv/bin/pytest tests/live`.
The sweep demo takes `OPTIMIZER_DEMO_SAMPLE` (items, default 30), `OPTIMIZER_DEMO_HOLDOUT` (fraction, default 0) and
`OPTIMIZER_DEMO_BUDGET` (USD ceiling, default 10), plus `OPTIMIZER_DEMO_ROUTE` (`support`, the explanatory
default, `classify` or `extract`), `OPTIMIZER_DEMO_DATASET` (`dolly`, `dolly:<category>`, `samsum` or `bitext`: public prompts
in place of the built-in tickets, downloaded once from Hugging Face), `OPTIMIZER_DEMO_MODEL` (default `claude-haiku-4-5`), `OPTIMIZER_DEMO_VOLUME` (simulated requests per month,
over which the sweep's cost is amortised, default 12857), `OPTIMIZER_DEMO_JUDGE` (judge model) and `OPTIMIZER_DEMO_DB` (a SQLite path that keeps the sweep's judgments for inspection): 30 items cost about $1.3 and take about 30 minutes; 100 items with a
30% holdout are estimated at about $4, have cost about $4, take about 75 minutes, and need the budget of 10; 200 items with a 30% holdout are estimated at about $7, have cost
about $6, and need the budget of 15.

`OPTIMIZER_LITELLM_LIVE=1 .venv/bin/pytest tests/live/test_litellm_mock.py` boots a real LiteLLM proxy with a mock
model and the guardrail (needs `pip install 'litellm[proxy]'`, no API key).

`OPTIMIZER_HEADROOM_LIVE=1 .venv/bin/pytest tests/live/test_headroom_mock.py` boots Headroom's real proxy app with the
pith extensions against a mock upstream (needs `pip install headroom-ai` and a re-run of `pip install -e .`, no API key).

`OPTIMIZER_PORTKEY_LIVE=1 .venv/bin/pytest tests/live/test_portkey_mock.py` runs the open-source Portkey gateway via
`npx` (needs Node and a free port 8787) with the webhook hooks against a mock upstream; no API key.

The three gateway live tests also run weekly in CI (the `integrations` workflow, also runnable by hand) against pinned
LiteLLM 1.104.2, Headroom 0.40.0 and Portkey gateway 1.15.2. The job guards pith's own changes against those pinned
versions (and unpinned transitive dependencies); a new gateway release is tested by bumping the pin in the workflow.
