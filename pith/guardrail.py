"""LiteLLM guardrail: the data plane inside a LiteLLM proxy. Spec: docs/design/specs/2026-10-09-litellm-plugin-design.md.

`PithHooks` is framework-free and fully testable; `PithGuardrail` (defined only where litellm is importable) inherits
it next to litellm's CustomGuardrail. Every hook fails open: a guardrail exception would reject the customer's request.
"""
import json
import logging
import os
import time
from typing import Mapping

from pith import db
from pith.config import Config, load_config
from pith.proxy import PURGE_EVERY, choose, record
from pith.usage import estimate_tokens, usage_from_body

log = logging.getLogger("pith.guardrail")
PROVIDER = "litellm"
CHAT_CALLS = ("completion", "acompletion")
LITELLM_KEYS = {"litellm_call_id", "litellm_logging_obj", "metadata", "litellm_metadata", "proxy_server_request",
                "secret_fields"}


def _stash(data) -> dict | None:
    return ((data or {}).get("metadata") or {}).get("pith")


class PithHooks:
    def __init__(self, config: Config | None = None, conn=None, env: Mapping[str, str] | None = None):
        self._config, self._conn = config, conn
        self._env = os.environ if env is None else env
        self._n = 0

    def _ready(self):
        if self._config is None:
            self._config = load_config(self._env.get("OPTIMIZER_CONFIG"), self._env)
        if self._conn is None:
            self._conn = db.connect(self._config.db_path)  # created on the event-loop thread, on first use
        return self._config, self._conn

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            if call_type not in CHAT_CALLS or not isinstance(data, dict) or "messages" not in data:
                return data
            cfg, conn = self._ready()
            snapshot = (data.get("proxy_server_request") or {}).get("body")
            body = dict(snapshot) if snapshot is not None else {k: v for k, v in data.items() if k not in LITELLM_KEYS}
            body.pop("metadata", None)  # LiteLLM's merged metadata must never be replayed
            headers = {k.lower(): v for k, v in (((data.get("metadata") or {}).get("headers")) or {}).items()}
            mode = (headers.get("x-optimizer") or "").lower()
            if mode == "bypass":
                return data
            fp, route, profile, new_body = choose(cfg, conn, PROVIDER, body, mode, headers.get("x-optimizer-route"))
            if profile != "P0":
                data["messages"] = new_body["messages"]
            data.setdefault("metadata", {})["pith"] = {"route": fp.key, "profile": profile, "body": body,
                                                       "t0": time.monotonic()}
        except Exception as exc:  # fail open; never log exc text (it can embed header values)
            log.warning("pre_call failed (%s); request forwarded unchanged", type(exc).__name__)
        return data

    def _record(self, stash: dict, resp: dict, usage) -> None:
        cfg, conn = self._ready()
        self._n += 1
        if self._n % PURGE_EVERY == 0:
            db.purge_expired(conn, time.time())
        record(cfg, conn, stash["route"], stash["profile"], usage, int((time.monotonic() - stash["t0"]) * 1000),
               json.dumps(stash["body"]), json.dumps(resp, default=str))

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            stash = _stash(data)
            if stash and hasattr(response, "model_dump"):
                resp = response.model_dump()
                self._record(stash, resp, usage_from_body("openai", resp))
        except Exception as exc:
            log.warning("post_call record failed (%s)", type(exc).__name__)
        return response

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        text, finish, usage, model = [], None, None, None
        async for chunk in response:
            try:  # read-only tee on the customer's stream: a bad chunk is passed on, not parsed
                d = chunk.model_dump()
                model = d.get("model") or model
                for c in d.get("choices") or []:
                    t = (c.get("delta") or {}).get("content")
                    if t:
                        text.append(t)
                    finish = c.get("finish_reason") or finish
                usage = d.get("usage") or usage
            except Exception:
                pass
            yield chunk
        try:  # reached only when the stream ended; a consumer that stops early closes the generator at `yield`
            stash = _stash(request_data)
            if stash:
                resp = {"object": "chat.completion", "model": model, "usage": usage or {},
                        "choices": [{"index": 0, "finish_reason": finish,
                                     "message": {"role": "assistant", "content": "".join(text)}}]}
                u = usage_from_body("openai", resp)
                if u.output_tokens is None and text:
                    u.output_tokens, u.estimated = estimate_tokens("".join(text)), True
                self._record(stash, resp, u)
        except Exception as exc:
            log.warning("stream record failed (%s)", type(exc).__name__)

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict, traceback_str=None):
        try:
            stash = _stash(request_data)
            if not stash or stash["profile"] == "P0" or stash.get("failed") or \
                    getattr(original_exception, "status_code", None) not in (400, 422):
                return
            stash["failed"] = True  # LiteLLM fires this hook twice per failure
            cfg, conn = self._ready()
            if db.bump_rejection(conn, stash["route"]) >= 3:
                db.set_pin(conn, stash["route"], "P0", status="reverted")
                log.warning("route %s reverted to P0 after 3 provider rejections", stash["route"])
        except Exception as exc:
            log.warning("failure bookkeeping failed (%s)", type(exc).__name__)


try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:  # pith stays importable without LiteLLM; the guardrail class exists only where it can be registered
    CustomGuardrail = None

if CustomGuardrail is not None:
    class PithGuardrail(PithHooks, CustomGuardrail):  # PithHooks first: CustomGuardrail defines no-op defaults of every hook
        """config.yaml: guardrail: pith.guardrail.PithGuardrail, mode: [pre_call, post_call], default_on: true."""

        def __init__(self, **kwargs):
            CustomGuardrail.__init__(self, **kwargs)
            PithHooks.__init__(self)
