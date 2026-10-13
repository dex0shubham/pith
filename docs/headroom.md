# Headroom plugin

pith runs inside [Headroom](https://github.com/chopratejas/headroom)'s proxy as two extensions. No second hop
and no Headroom code changes.

Running [Headroom](https://github.com/chopratejas/headroom)'s proxy? Install pith into the same environment and enable
its two extensions; no second hop and no Headroom code changes:

    pip install pith-optimizer
    HEADROOM_PROXY_EXTENSIONS=pith HEADROOM_PIPELINE_EXTENSIONS=pith OPTIMIZER_CONFIG=/path/to/pith.toml headroom proxy

(`headroom proxy --proxy-extension pith` is the flag form of the first variable; the pipeline extension has no flag.)
The proxy extension installs a middleware that fingerprints each `/v1/messages` and `/v1/chat/completions` request on
the client's original body, records usage from the response, and rewrites effort parameters for P1/P4 pins. The pipeline
extension appends the shape text for P2/P3/P4 pins at Headroom's `PRE_SEND` stage, after compression, as user text.
Routes recorded this way are ordinary Anthropic/OpenAI routes: the same keys as traffic through `pith serve`, swept with
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` and the full profile set. `X-Optimizer: off|bypass` and `X-Optimizer-Route` work
as usual. Keep Headroom's own output shaper off (`HEADROOM_OUTPUT_SHAPER` unset) and its learned verbosity steering off
(`HEADROOM_VERBOSITY_LEVEL` unset, no `headroom learn --verbosity`) while pith is enabled: two steering instructions
would fight. Turn Headroom's response cache off while pins are in use (`HEADROOM_CACHE_ENABLED=false`): its entries are
keyed before pith shapes a request, so a shaped response can be served to an unshaped one and vice versa. Cache hits
never reach the provider and are never recorded. `/v1/responses` passes through unrecorded. Any pith error forwards the
request unchanged. Headroom's beacon and telemetry switches are Headroom's own (`HEADROOM_BEACON=off`).
