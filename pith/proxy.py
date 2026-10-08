"""The data plane: forward, record, apply pinned profile, fail open. Spec §3, §7, §8."""
import json
import logging
import random
import threading
import time
from dataclasses import dataclass

import anyio
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from pith import db
from pith.config import Config
from pith.fingerprint import Fingerprint, fingerprint
from pith.providers import detect_provider, is_responses_api, upstream
from pith.report import render_html, route_rows, sweep_rows
from pith.rewrite import RouteState, apply_profile, is_system_role_rejection
from pith.usage import StreamUsage, estimate_tokens, usage_from_body

log = logging.getLogger("pith")
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding",
               "x-optimizer", "x-optimizer-route"}
PURGE_EVERY = 1000
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
    out = {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS}
    out["accept-encoding"] = "identity"  # we relay raw bytes and drop content-encoding, so upstream must not compress
    return out


def _transport_error_response(exc: Exception, req) -> httpx.Response:
    if isinstance(exc, httpx.TimeoutException):
        return httpx.Response(504, json={"error": {"type": "upstream_timeout", "message": str(exc)}}, request=req)
    return httpx.Response(502, json={"error": {"type": "upstream_unreachable", "message": str(exc)}}, request=req)


async def _forward(client: httpx.AsyncClient, method: str, url: str, headers: dict, content: bytes, stream: bool):
    req = None
    try:
        req = client.build_request(method, url, headers=headers, content=content)
        return await client.send(req, stream=stream)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:  # transport failure: surface as 502/504 rather than an unhandled 500
        log.warning("upstream request to %s failed: %r", url, exc)
        return _transport_error_response(exc, req)


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
    threading.Thread(target=estimate_tokens, args=("warm",), daemon=True).start()  # load tiktoken off the event loop
    counter = {"n": 0}
    db.purge_expired(conn, time.time())

    @app.get("/optimizer/health")
    async def health():
        return {"ok": True}

    @app.get("/optimizer/report")
    async def report():
        return route_rows(conn, config.prices)

    @app.get("/optimizer/report.html")
    async def report_html():
        return HTMLResponse(render_html(route_rows(conn, config.prices)))

    @app.get("/optimizer/sweeps/{route_key:path}")
    async def sweeps(route_key: str):
        return sweep_rows(conn, route_key)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request):
        counter["n"] += 1
        if counter["n"] % PURGE_EVERY == 0:
            try:
                db.purge_expired(conn, time.time())
            except Exception:  # fail open: retention housekeeping must never fail a request
                log.exception("purge failed")
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
        profile_used = d.profile
        if d.profile != "P0" and resp.status_code in (400, 422):
            rewritten_status = resp.status_code
            try:
                err_text = (await resp.aread()).decode("utf-8", "replace")
            except Exception:  # a body-read failure must not escape as a 500; treat the error body as empty
                log.exception("reading rejection body failed")
                err_text = ""
            finally:
                await resp.aclose()
            log.info("provider rejected profile %s on %s (%s); retrying original", d.profile, d.fp.key, rewritten_status)
            t0 = time.monotonic()
            resp = await _forward(client, request.method, url, headers, raw, stream=True)
            profile_used = "P0"
            if resp.status_code < 400:  # original succeeded, so the rewrite was the problem (4xx/5xx = inconclusive)
                try:
                    if is_system_role_rejection(rewritten_status, err_text):
                        db.set_injection_form(conn, d.fp.key, "user_text")
                    elif db.bump_rejection(conn, d.fp.key) >= 3:
                        db.set_pin(conn, d.fp.key, "P0", status="reverted")
                        log.warning("route %s reverted to P0 after 3 provider rejections", d.fp.key)
                except Exception:
                    log.exception("rejection bookkeeping failed")
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESP_DROP}
        if "text/event-stream" in resp.headers.get("content-type", ""):
            su = StreamUsage(d.provider or "openai")
            collected = []

            async def relay():
                completed = False
                try:
                    async for chunk in resp.aiter_bytes():
                        su.feed(chunk)
                        if d.record:
                            collected.append(chunk)
                        yield chunk
                    completed = True
                finally:
                    with anyio.CancelScope(shield=True):
                        await resp.aclose()
                    if completed and d.record and resp.status_code < 400:
                        try:
                            _record(config, conn, d, su.result(), int((time.monotonic() - t0) * 1000), raw,
                                    b"".join(collected).decode("utf-8", "replace"), profile_used)
                        except Exception:
                            log.exception("record failed")

            return StreamingResponse(relay(), status_code=resp.status_code, headers=out_headers)
        try:
            content = await resp.aread()
            await resp.aclose()
        except httpx.HTTPError as exc:  # upstream died mid-body (aread already closed the response)
            log.warning("upstream body read for %s failed: %r", url, exc)
            err = _transport_error_response(exc, resp.request)
            return Response(content=err.content, status_code=err.status_code,
                            headers={"content-type": "application/json"})
        latency = int((time.monotonic() - t0) * 1000)
        if d.record and resp.status_code < 400:
            try:
                usage = usage_from_body(d.provider, json.loads(content))
                _record(config, conn, d, usage, latency, raw, content.decode("utf-8", "replace"), profile_used)
            except Exception:
                log.exception("record failed")
        return Response(content=content, status_code=resp.status_code, headers=out_headers)

    return app
