# Portkey plugin

Behind a [Portkey](https://github.com/Portkey-AI/gateway) gateway, pith needs no gateway code: Portkey's built-in
`default.webhook` check calls `pith serve` before and after each request.

Behind a [Portkey](https://github.com/Portkey-AI/gateway) gateway, pith needs no gateway code: Portkey's built-in
`default.webhook` check calls pith before and after each request. Run `pith serve` where the gateway can reach it and
add two hooks to the Portkey config (`x-portkey-config` header or a saved config):

    {
      "before_request_hooks": [{"type": "mutator", "id": "pith-before",
        "checks": [{"id": "default.webhook", "parameters": {"webhookURL": "http://pith:8787/optimizer/portkey",
                                                            "headers": {"authorization": "Bearer <webhook_token>"}}}]}],
      "after_request_hooks":  [{"type": "guardrail", "id": "pith-after", "deny": false,
        "checks": [{"id": "default.webhook", "parameters": {"webhookURL": "http://pith:8787/optimizer/portkey",
                                                            "headers": {"authorization": "Bearer <webhook_token>"}}}]}]
    }

The before hook fingerprints the request and, for a pinned route, returns it with the shape text appended to the last
user message; the after hook records usage into the same SQLite the CLI reads. `x-portkey-metadata` keys: `pith_route`
(name the route), `pith_bypass` (skip pith entirely), `pith: "off"` (force P0). Set `webhook_token` in `pith.toml`
when the endpoint is reachable beyond the gateway. Limits: only OpenAI-format (`chatComplete`) requests are handled;
streaming responses are seen but not recorded (Portkey delivers no body for them); provider rejections are invisible
to after hooks, so `recheck` is the drift guard; Portkey appends `hook_results` to responses whenever hooks run. Only
P2/P3 apply through Portkey. Note the port clash: Portkey's gateway and pith both default to 8787, so move one.

Sweep those routes through the gateway: set `portkey_upstream`, put a `[prices."<model>"]` entry for each model name
the clients send, add any routing headers under `[portkey_headers]` (a saved config id, provider, virtual key), export
`PORTKEY_API_KEY`, and run `python -m pith sweep`. Replays and the judge carry `x-portkey-metadata: {"pith_bypass": true}`
so the webhook ignores them; `judge_provider = "portkey"` runs the judge through the gateway too.

Without `webhook_token`, anyone who can reach pith can post hook payloads that write usage rows and sweep samples, so set
the token or keep pith on a private network (pith logs a warning at startup when the endpoint is open and not bound to
loopback). Open-source gateway users: sweeps still require `PORTKEY_API_KEY` in the environment, but the OSS gateway
ignores it, so any dummy value works. Provider credentials go in `[portkey_headers]` (`authorization`, or an
`x-portkey-config` JSON with `api_key`), which puts a secret in `pith.toml`, so protect that file; saved config ids and
virtual keys exist only on the hosted product. `judge_provider = "portkey"` requires the gateway config to route
`judge_model`: with `x-portkey-provider = "openai"` the judge model must be an OpenAI model. Do not chain pith adapters
(for example `pith serve` with Portkey as its upstream while the webhook is also active): the inner adapter would strip
the outer one's shape text and misattribute it. Metadata flags accept booleans or the strings `true/false`, `1/0`,
`yes/no`, `on/off`.
