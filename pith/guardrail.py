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
