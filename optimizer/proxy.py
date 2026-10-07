"""The data plane: forward, record, apply pinned profile, fail open. Spec §3, §7, §8."""
import json
import logging
import random
import time
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from optimizer import db
from optimizer.config import Config
from optimizer.fingerprint import Fingerprint, fingerprint
from optimizer.providers import detect_provider, is_responses_api, upstream
from optimizer.rewrite import RouteState, apply_profile
from optimizer.usage import StreamUsage, usage_from_body

log = logging.getLogger("optimizer")
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}
RESP_DROP = {"content-length", "content-encoding", "transfer-encoding", "connection"}


@dataclass
class Decision:
    provider: str | None
    fp: Fingerprint | None
    route: dict | None
    profile: str
    body_bytes: bytes
    record: bool


def _decide(config: Config, conn, path: str, headers, raw: bytes) -> Decision:
    """Everything before forwarding. Any exception here is caught by the caller -> fail open."""
    provider = detect_provider(path)
    if provider is None:
        return Decision(None, None, None, "P0", raw, False)
    mode = (headers.get("x-optimizer") or "").lower()
    if mode == "bypass":
        return Decision(provider, None, None, "P0", raw, False)
    body = json.loads(raw)
    fp = fingerprint(provider, body, headers.get("x-optimizer-route"))
    db.upsert_route(conn, fp.key, provider, fp.model, fp.system_hash, headers.get("x-optimizer-route"))
    route = db.get_route(conn, fp.key)
    route_cfg = config.routes.get(fp.key)
    enabled = config.enabled and mode != "off" and (route_cfg is None or route_cfg.enabled)
    profile = route["pinned_profile"] if enabled else "P0"
    if profile == "P0":
        return Decision(provider, fp, route, "P0", raw, True)
    state = RouteState(profile, route["injection_form"], route["target_words"], route["exemplar"])
    new_body = apply_profile(provider, body, state, responses_api=is_responses_api(path))
    return Decision(provider, fp, route, profile, json.dumps(new_body).encode(), True)


def _upstream_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS}


async def _forward(client: httpx.AsyncClient, method: str, url: str, headers: dict, content: bytes, stream: bool):
    req = client.build_request(method, url, headers=headers, content=content)
    try:
        return await client.send(req, stream=stream)
    except httpx.HTTPError as exc:  # transport failure: surface as 502/504 rather than an unhandled 500
        log.warning("upstream request to %s failed: %r", url, exc)
        if isinstance(exc, httpx.TimeoutException):
            return httpx.Response(504, json={"error": {"type": "upstream_timeout", "message": str(exc)}}, request=req)
        return httpx.Response(502, json={"error": {"type": "upstream_unreachable", "message": str(exc)}}, request=req)


def _record(config: Config, conn, d: Decision, usage, latency_ms: int, request_raw: bytes, response_text: str,
            profile: str):
    body_ref = None
    if random.random() < config.sample_rate:
        body_ref = db.store_body(conn, request_raw.decode("utf-8", "replace"), response_text,
                                 time.time() + config.retention_days * 86400)
    db.record_request(conn, ts=time.time(), route_key=d.fp.key, profile=profile, input_tokens=usage.input_tokens,
                      output_tokens=usage.output_tokens, cache_read=usage.cache_read, cache_create=usage.cache_create,
                      estimated=usage.estimated, stop_reason=usage.stop_reason, latency_ms=latency_ms, body_ref=body_ref)


def create_app(config: Config, conn, client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI()
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    @app.get("/optimizer/health")
    async def health():
        return {"ok": True}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request):
        raw = await request.body()
        path = "/" + path
        try:
            d = _decide(config, conn, path, request.headers, raw) if request.method == "POST" else \
                Decision(detect_provider(path), None, None, "P0", raw, False)
        except Exception:  # fail open: forward the customer's original bytes
            log.exception("decide failed; forwarding original request")
            d = Decision(detect_provider(path), None, None, "P0", raw, False)
        # Unknown paths still need an upstream: Anthropic for /v1/messages/*, OpenAI otherwise.
        base = upstream(d.provider, config) if d.provider else (
            config.anthropic_upstream if path.startswith("/v1/messages") else config.openai_upstream)
        url = base + path + (f"?{request.url.query}" if request.url.query else "")
        headers = _upstream_headers(request.headers)
        t0 = time.monotonic()
        resp = await _forward(client, request.method, url, headers, d.body_bytes, stream=True)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESP_DROP}
        if "text/event-stream" in resp.headers.get("content-type", ""):
            su = StreamUsage(d.provider or "openai")
            collected = []

            async def relay():
                try:
                    async for chunk in resp.aiter_raw():
                        su.feed(chunk)
                        if d.record:
                            collected.append(chunk)
                        yield chunk
                finally:
                    await resp.aclose()
                    if d.record and resp.status_code < 400:
                        try:
                            _record(config, conn, d, su.result(), int((time.monotonic() - t0) * 1000), raw,
                                    b"".join(collected).decode("utf-8", "replace"), d.profile)
                        except Exception:
                            log.exception("record failed")

            return StreamingResponse(relay(), status_code=resp.status_code, headers=out_headers)
        content = await resp.aread()
        await resp.aclose()
        latency = int((time.monotonic() - t0) * 1000)
        if d.record and resp.status_code < 400:
            try:
                usage = usage_from_body(d.provider, json.loads(content))
                _record(config, conn, d, usage, latency, raw, content.decode("utf-8", "replace"), d.profile)
            except Exception:
                log.exception("record failed")
        return Response(content=content, status_code=resp.status_code, headers=out_headers)

    return app
