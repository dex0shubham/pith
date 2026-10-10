"""Headroom adapter: pith's data plane inside a Headroom proxy. Spec: docs/design/specs/2026-10-09-headroom-adapter-design.md.

`install` (entry point headroom.proxy_extension) adds `PithMiddleware`, which decides on the client's original body and
records from the teed response; `PithPipeline` (entry point headroom.pipeline_extension) appends the pinned shape text at
PRE_SEND, after Headroom's compression. Headroom is never imported. Every pith step fails open.
"""
import contextvars
import json
import logging
import os
import threading
import time
from typing import Mapping

from pith import db
from pith.config import Config, load_config
from pith.proxy import PURGE_EVERY, choose, record
from pith.rewrite import RouteState, append_shape, apply_effort
from pith.usage import StreamUsage, estimate_tokens, usage_from_body

log = logging.getLogger("pith.headroom")
PATHS = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai"}
PARAM_PROFILES = ("P1", "P1b", "P4")
SHAPE_PROFILES = ("P2", "P3", "P4")
PITH_HEADERS = (b"x-optimizer", b"x-optimizer-route")
PROMPT_SIZE_ERRORS = ("context_length_exceeded", "prompt is too long")
DECISION: contextvars.ContextVar = contextvars.ContextVar("pith_decision", default=None)


class Runtime:
    """Lazy config + SQLite connection, shared by the middleware and the pipeline extension of one process."""

    def __init__(self, config: Config | None = None, conn=None, env: Mapping[str, str] | None = None):
        self.config, self.conn = config, conn
        self.env = os.environ if env is None else env
        self.n, self.warmed = 0, False

    def ready(self):
        if not self.warmed:
            self.warmed = True
            threading.Thread(target=estimate_tokens, args=("warm",), daemon=True).start()  # load tiktoken off the loop
        if self.config is None:
            self.config = load_config(self.env.get("OPTIMIZER_CONFIG"), self.env)
        if self.conn is None:
            self.conn = db.connect(self.config.db_path)  # created on the event-loop thread, on first use
        return self.config, self.conn


RUNTIME = Runtime()


def _decide(rt: Runtime, provider: str, raw: bytes, headers: dict) -> tuple[dict | None, bytes]:
    """(decision, bytes to forward). A None decision means pith stays out of this request entirely."""
    mode = (headers.get("x-optimizer") or "").lower()
    if mode == "bypass":
        return None, raw
    cfg, conn = rt.ready()
    original, body = raw, json.loads(raw)
    if not isinstance(body, dict):
        return None, raw
    fp, route, profile, _ = choose(cfg, conn, provider, body, mode, headers.get("x-optimizer-route"))
    if route["injection_form"] != "user_text":
        db.set_injection_form(conn, fp.key, "user_text")  # Headroom gets the shape as user text; sweeps must replay it so
    if profile in PARAM_PROFILES:
        raw = json.dumps(apply_effort(provider, body, profile)).encode()
    decision = {"route": fp.key, "profile": profile, "applied": profile not in SHAPE_PROFILES,
                "request_json": original.decode("utf-8", "replace"), "t0": time.monotonic(),
                "target_words": route["target_words"], "exemplar": route["exemplar"]}
    return decision, raw


class PithMiddleware:
    """Pure ASGI. Buffers the request body, decides, forwards (possibly with effort params rewritten), tees the
    response and records. Only POST /v1/messages and /v1/chat/completions; everything else is untouched."""

    def __init__(self, app, runtime: Runtime | None = None):
        self.app, self.rt = app, runtime or RUNTIME

    async def __call__(self, scope, receive, send):
        provider = PATHS.get(scope.get("path", "").rstrip("/")) if scope.get("type") == "http" else None
        if provider is None or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        chunks, more = [], True
        while more:
            m = await receive()
            chunks.append(m)
            more = m.get("type") == "http.request" and m.get("more_body", False)
        original_raw = raw = b"".join(c.get("body", b"") for c in chunks if c.get("type") == "http.request")
        decision = None
        try:
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
            decision, raw = _decide(self.rt, provider, raw, headers)
        except Exception as exc:  # fail open: forward the client's bytes untouched; never log exc text
            log.warning("decide failed (%s); forwarding original request", type(exc).__name__)
        DECISION.set(decision)
        drop = PITH_HEADERS + ((b"content-length",) if raw is not original_raw else ())
        headers = [(k, v) for k, v in scope.get("headers") or [] if k.lower() not in drop]
        if raw is not original_raw:
            headers.append((b"content-length", str(len(raw)).encode()))  # the client's length is stale after a rewrite
        scope = dict(scope, headers=headers)
        queue = [{"type": "http.request", "body": raw, "more_body": False}]
        queue += [c for c in chunks if c.get("type") != "http.request"]

        async def replay():
            return queue.pop(0) if queue else await receive()  # after the buffered body, Headroom waits on the real receive

        su, buf, status, done = StreamUsage(provider), [], [None], [False]

        async def tee(message):
            if message["type"] == "http.response.start":
                status[0] = message.get("status")
            elif message["type"] == "http.response.body" and decision is not None:
                chunk = message.get("body", b"")
                buf.append(chunk)
                su.feed(chunk)  # never raises
                done[0] = not message.get("more_body", False)
            await send(message)

        await self.app(scope, replay, tee)
        if decision is not None and done[0]:  # a response cut off before its final body message is not recorded
            self._finish(provider, decision, status[0], su, b"".join(buf))

    def _finish(self, provider: str, decision: dict, status, su: StreamUsage, raw: bytes) -> None:
        if decision.get("cached"):  # served from Headroom's response cache: no upstream call, nothing to record
            return
        try:
            cfg, conn = self.rt.ready()
            profile = decision["profile"]
            # An unapplied P4 still had its effort rewritten by the middleware: it went out as P1.
            used = profile if decision["applied"] else ("P1" if profile == "P4" else "P0")
            if status is not None and status < 400:
                try:
                    usage = usage_from_body(provider, json.loads(raw))
                except Exception:  # SSE or non-JSON body: the tee already parsed it
                    usage = su.result()
                self.rt.n += 1
                if self.rt.n % PURGE_EVERY == 0:
                    db.purge_expired(conn, time.time())
                record(cfg, conn, decision["route"], used, usage, int((time.monotonic() - decision["t0"]) * 1000),
                       decision["request_json"], raw.decode("utf-8", "replace"))
            elif status in (400, 422) and profile in PARAM_PROFILES and not any(
                    e in raw.decode("utf-8", "replace").lower() for e in PROMPT_SIZE_ERRORS):
                # No retry here, so attribute conservatively: only an effort/verbosity param plausibly causes a 4xx,
                # and a prompt-size error is the client's, not the rewrite's.
                if db.bump_rejection(conn, decision["route"]) >= 3:
                    db.set_pin(conn, decision["route"], "P0", status="reverted")
                    log.warning("route %s reverted to P0 after 3 provider rejections", decision["route"])
        except Exception as exc:
            log.warning("record failed (%s)", type(exc).__name__)


class PithPipeline:
    """Entry point headroom.pipeline_extension. At PRE_SEND (after compression) append the pinned shape as user text."""

    def on_pipeline_event(self, event):
        try:
            d = DECISION.get()
            stage = getattr(getattr(event, "stage", None), "name", None)
            if d is None:
                return None
            if stage == "INPUT_CACHED":  # Headroom answers from its response cache; PRE_SEND never runs
                d["cached"] = True
            elif stage == "PRE_SEND" and not d.get("applied") and d["profile"] in SHAPE_PROFILES and event.messages:
                state = RouteState(d["profile"], "user_text", d["target_words"], d["exemplar"])
                d["applied"] = append_shape(event.messages, state)
        except Exception as exc:  # fail open: the request goes out unshaped and is recorded as P0
            log.warning("PRE_SEND shape failed (%s); request forwarded unchanged", type(exc).__name__)
        return None


def install(app, config) -> None:
    """Entry point headroom.proxy_extension: Headroom calls this while building its app."""
    app.add_middleware(PithMiddleware)
