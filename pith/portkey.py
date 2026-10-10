"""Portkey webhook: pith's data plane behind Portkey's built-in `default.webhook` check.
Spec: docs/design/specs/2026-10-09-portkey-webhook-design.md.

Portkey POSTs the hook context here before and after each request. The handler never raises: a pith failure answers
{"verdict": true} with no transform, so Portkey forwards the customer's request unchanged.
"""
import json
import logging
import time

from pith import db
from pith.proxy import PURGE_EVERY, choose, record
from pith.rewrite import strip_shape
from pith.usage import usage_from_body

log = logging.getLogger("pith.portkey")
PROVIDER = "portkey"
_counter = {"n": 0}


def _flag(v) -> bool:
    """Metadata flags may arrive as strings from Portkey's hosted product."""
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


def handle(cfg, conn, payload) -> dict:
    try:
        if not isinstance(payload, dict) or payload.get("requestType") != "chatComplete":
            return {"verdict": True}
        req = (payload.get("request") or {}).get("json")
        if not isinstance(req, dict) or "messages" not in req:
            return {"verdict": True}
        meta = payload.get("metadata") or {}
        if _flag(meta.get("pith_bypass")):
            return {"verdict": True}
        mode = "off" if str(meta.get("pith", "")).lower() == "off" else ""
        route_name = meta.get("pith_route") or None
        event = payload.get("eventType")
        if event == "beforeRequestHook":
            fp, route, profile, body = choose(cfg, conn, PROVIDER, req, mode, route_name)
            if profile == "P0":
                return {"verdict": True}
            return {"verdict": True, "transformedData": {"request": {"json": body}}}
        if event == "afterRequestHook":
            resp = (payload.get("response") or {}).get("json")
            if not isinstance(resp, dict) or not resp:
                return {"verdict": True}  # streams deliver null here; nothing to record
            original, stripped = strip_shape(req)
            fp, route, _, _ = choose(cfg, conn, PROVIDER, original, "off", route_name)  # register/refresh, never rewrite
            profile = route["pinned_profile"] if stripped else "P0"
            if stripped and profile not in ("P2", "P3"):
                return {"verdict": True}  # pin changed between the hooks; a shaped response must not enter the P0 baseline
            _counter["n"] += 1
            if _counter["n"] % PURGE_EVERY == 0:
                db.purge_expired(conn, time.time())
            record(cfg, conn, fp.key, profile, usage_from_body("openai", resp), 0, json.dumps(original), json.dumps(resp))
        return {"verdict": True}
    except Exception as exc:  # never raise into Portkey; never log exc text
        log.warning("portkey hook failed (%s); request left unchanged", type(exc).__name__)
        return {"verdict": True}
