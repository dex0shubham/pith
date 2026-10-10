"""The data plane: forward, record, apply pinned profile, fail open. Spec §3, §7, §8."""
import hmac
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


def choose(config: Config, conn, provider: str, body: dict, mode: str, route_name: str | None,
           responses_api: bool = False) -> tuple[Fingerprint, dict, str, dict]:
    """Fingerprint, register and pick the profile for one request: (fp, route, profile, body_to_send).
    body_to_send is `body` itself at P0. Shared by the proxy and the LiteLLM guardrail."""
    fp = fingerprint(provider, body, route_name)
    db.upsert_route(conn, fp.key, provider, fp.model, fp.system_hash, route_name)
    route = db.get_route(conn, fp.key)
    route_cfg = config.routes.get(fp.key)
    enabled = config.enabled and mode != "off" and (route_cfg is None or route_cfg.enabled)
    profile = route["pinned_profile"] if enabled else "P0"
    if profile == "P0":
        return fp, route, "P0", body
    state = RouteState(profile, route["injection_form"], route["target_words"], route["exemplar"])
    return fp, route, profile, apply_profile(provider, body, state, responses_api=responses_api)


def _decide(config: Config, conn, path: str, headers, raw: bytes) -> Decision:
    """Everything before forwarding. Any exception here is caught by the caller -> fail open."""
    provider = detect_provider(path)
    if provider is None:
        return Decision(None, None, None, "P0", raw, False)
    mode = (headers.get("x-optimizer") or "").lower()
    if mode == "bypass":
        return Decision(provider, None, None, "P0", raw, False)
    fp, route, profile, body = choose(config, conn, provider, json.loads(raw), mode, headers.get("x-optimizer-route"),
                                      responses_api=is_responses_api(path))
    return Decision(provider, fp, route, profile, raw if profile == "P0" else json.dumps(body).encode(), True)


def _upstream_headers(headers) -> dict:
    out = {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS}
    out["accept-encoding"] = "identity"  # we relay raw bytes and drop content-encoding, so upstream must not compress
    return out


def _transport_error_response(exc: Exception, req) -> httpx.Response:
    if isinstance(exc, httpx.TimeoutException):
        return httpx.Response(504, json={"error": {"type": "upstream_timeout", "message": type(exc).__name__}}, request=req)
    return httpx.Response(502, json={"error": {"type": "upstream_unreachable", "message": type(exc).__name__}}, request=req)


async def _forward(client: httpx.AsyncClient, method: str, url: str, headers: dict, content: bytes, stream: bool):
    req = None
    try:
        req = client.build_request(method, url, headers=headers, content=content)
        return await client.send(req, stream=stream)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:  # transport failure: surface as 502/504 rather than an unhandled 500
        log.warning("upstream request to %s failed: %s", url, type(exc).__name__)  # never log exc text: httpx embeds header values
        return _transport_error_response(exc, req)


def record(config: Config, conn, route_key: str, profile: str, usage, latency_ms: int, request_json: str,
           response_json: str) -> None:
    body_ref = None
    if random.random() < config.sample_rate:
        body_ref = db.store_body(conn, request_json, response_json, time.time() + config.retention_days * 86400)
    db.record_request(conn, ts=time.time(), route_key=route_key, profile=profile, input_tokens=usage.input_tokens,
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

    from pith.portkey import handle as portkey_handle  # function-level: pith.portkey imports this module

    if not config.webhook_token and not config.listen.startswith(("127.0.0.1:", "localhost:")):
        log.warning("/optimizer/portkey accepts unauthenticated hook posts; set webhook_token or bind listen to loopback")

    @app.post("/optimizer/portkey")
    async def portkey_hook(request: Request):
        supplied = (request.headers.get("authorization") or "").encode()
        if config.webhook_token and not hmac.compare_digest(supplied, f"Bearer {config.webhook_token}".encode()):
            return Response(status_code=401)
        try:
            payload = await request.json()
        except Exception:  # malformed JSON: nothing to decide on
            payload = None
        return portkey_handle(config, conn, payload)

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
                            record(config, conn, d.fp.key, profile_used, su.result(), int((time.monotonic() - t0) * 1000),
                                   raw.decode("utf-8", "replace"), b"".join(collected).decode("utf-8", "replace"))
                        except Exception:
                            log.exception("record failed")

            return StreamingResponse(relay(), status_code=resp.status_code, headers=out_headers)
        try:
            content = await resp.aread()
            await resp.aclose()
        except httpx.HTTPError as exc:  # upstream died mid-body (aread already closed the response)
            log.warning("upstream body read for %s failed: %s", url, type(exc).__name__)
            err = _transport_error_response(exc, resp.request)
            return Response(content=err.content, status_code=err.status_code,
                            headers={"content-type": "application/json"})
        latency = int((time.monotonic() - t0) * 1000)
        if d.record and resp.status_code < 400:
            try:
                usage = usage_from_body(d.provider, json.loads(content))
                record(config, conn, d.fp.key, profile_used, usage, latency, raw.decode("utf-8", "replace"),
                       content.decode("utf-8", "replace"))
            except Exception:
                log.exception("record failed")
        return Response(content=content, status_code=resp.status_code, headers=out_headers)

    return app
