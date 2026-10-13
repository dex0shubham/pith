# LiteLLM plugin

pith runs inside a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) as a guardrail. Nothing in the
proxy changes; the sweep CLI reads the same SQLite.

Already running a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)? Register pith as a guardrail instead of
adding a second hop. Install pith into the LiteLLM proxy's environment and add to its `config.yaml`:

    guardrails:
      - guardrail_name: pith
        litellm_params:
          guardrail: pith.guardrail.PithGuardrail
          mode: [pre_call, post_call]
          default_on: true

    # in the proxy's environment
    OPTIMIZER_CONFIG=/path/to/pith.toml      # optional; every scalar setting is also an OPTIMIZER_<FIELD> variable; [routes] and [prices] need the toml
    pip install pith-optimizer

The guardrail fingerprints every `/v1/chat/completions` request, records usage into the same SQLite the CLI reads, and
applies a pinned profile by appending the shape text to the last user message. Only P2 and P3 apply through LiteLLM:
LiteLLM folds system messages into the provider's system prompt (cache-breaking), and a guardrail cannot retry a
rejected request, so effort profiles are left to the standalone proxy. The README's kill switches still work: the
guardrail reads `X-Optimizer` and `X-Optimizer-Route` from the request headers LiteLLM records. Any pith error inside a
hook is logged and the request proceeds unchanged. Three provider-attributed 400/422 responses on a pinned route revert
it to P0; the count resets when the route is re-pinned. The guardrail writes to SQLite from each LiteLLM worker process;
keep the database on local disk (WAL handles concurrent workers). To see the report for guardrail-recorded routes, run
`python -m pith serve --config pith.toml` against the same `db_path` and open `/optimizer/report.html`.

Sweep those routes from the same host, through the LiteLLM proxy (replays carry `X-Optimizer: bypass`, so the guardrail
ignores them):

    litellm_upstream = "http://localhost:4000"   # pith.toml
    [prices."<model_name as clients send it>"]   # required: LiteLLM aliases are not in the built-in price table
    input = 1.25
    output = 10.0

    export LITELLM_API_KEY=sk-...                 # a LiteLLM virtual key or the master key
    .venv/bin/python -m pith sweep --config pith.toml

Set `judge_provider = "litellm"` and a `judge_model` LiteLLM serves to run the judge through it as well.
