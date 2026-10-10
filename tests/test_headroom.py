import json
import logging
import sys
import threading
from types import SimpleNamespace

import pytest

from pith import db
from pith.config import Config
from pith.headroom import DECISION, PithMiddleware, PithPipeline, Runtime, install

ANTH = {"model": "claude-opus-5-5", "max_tokens": 50, "system": "S", "messages": [{"role": "user", "content": "q"}]}
CHAT = {"model": "gpt-5", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]}
ANTH_RESP = {"id": "m", "type": "message", "stop_reason": "end_turn", "content": [{"type": "text", "text": "A"}],
             "usage": {"input_tokens": 9, "output_tokens": 4}}
CHAT_SSE = (b'data: {"choices":[{"delta":{"content":"A"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\ndata: [DONE]\n\n')


class Downstream:
    """Stands in for Headroom's app: records the body it receives, lets a test hook act on it (Headroom's pipeline
    runs between the two), and answers with a canned response in several chunks."""

    def __init__(self, response=b"", status=200, sse=False, raise_exc=None):
        self.response, self.status, self.sse, self.raise_exc = response, status, sse, raise_exc
        self.seen, self.decision_seen, self.hook, self.scope = [], None, None, None

    async def __call__(self, scope, receive, send):
        body, more = b"", True
        while more:
            m = await receive()
            body += m.get("body", b"")
            more = m.get("more_body", False)
        self.scope = scope
        parsed = json.loads(body) if body else None
        self.seen.append(parsed)
        self.decision_seen = DECISION.get()
        if self.hook and parsed:
            self.hook(parsed)
        if self.raise_exc:
            raise self.raise_exc
        ctype = b"text/event-stream" if self.sse else b"application/json"
        await send({"type": "http.response.start", "status": self.status, "headers": [(b"content-type", ctype)]})
        for i in range(0, len(self.response), 7):
            await send({"type": "http.response.body", "body": self.response[i:i + 7], "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def call(mw, path, body, headers=None, method="POST"):
    scope = {"type": "http", "method": method, "path": path,
             "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]}
    raw = json.dumps(body).encode()
    inbox = [{"type": "http.request", "body": raw[:5], "more_body": True},
             {"type": "http.request", "body": raw[5:], "more_body": False}]
    delegated = []

    async def receive():
        if inbox:
            return inbox.pop(0)
        delegated.append(1)
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent, delegated


def make(response=json.dumps(ANTH_RESP).encode(), **kw):
    conn = db.connect(":memory:")
    down = Downstream(response, **kw)
    return PithMiddleware(down, Runtime(Config(sample_rate=1.0), conn)), down, conn


def relayed(sent):
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


class Boom:
    def execute(self, *a, **k):
        raise RuntimeError("x-api-key: sk-secret")


@pytest.mark.anyio
async def test_non_matching_requests_pass_through_untouched():
    mw, down, conn = make()
    await call(mw, "/v1/models", {"x": 1}, method="GET")
    await call(mw, "/v1/responses", {"model": "gpt-5", "input": "q"})
    assert down.seen == [{"x": 1}, {"model": "gpt-5", "input": "q"}] and down.decision_seen is None
    assert conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


@pytest.mark.anyio
async def test_p0_relays_bytes_and_records_json_and_sse_for_both_providers():
    mw, down, conn = make()
    sent, _ = await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert relayed(sent) == json.dumps(ANTH_RESP).encode() and sent[0]["status"] == 200
    assert down.seen[-1] == ANTH and down.decision_seen["profile"] == "P0" and down.decision_seen["applied"] is True
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["route_key"], row["profile"], row["input_tokens"], row["output_tokens"], row["stop_reason"]) == ("r", "P0", 9, 4, "end_turn")
    assert down.decision_seen["request_json"] == json.dumps(ANTH) and "body" not in down.decision_seen
    assert row["latency_ms"] >= 0 and json.loads(conn.execute("SELECT request_json FROM bodies").fetchone()[0]) == ANTH
    assert db.get_route(conn, "r")["injection_form"] == "user_text" and db.get_route(conn, "r")["provider"] == "anthropic"
    mw, down, conn = make(CHAT_SSE, sse=True)
    sent, _ = await call(mw, "/v1/chat/completions", dict(CHAT, stream=True))
    assert relayed(sent) == CHAT_SSE
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert (row["input_tokens"], row["output_tokens"], row["stop_reason"], row["estimated"]) == (9, 4, "stop", 0)
    assert db.get_route(conn, row["route_key"])["provider"] == "openai"
    assert conn.execute("SELECT response_json FROM bodies").fetchone()[0] == CHAT_SSE.decode()


@pytest.mark.anyio
async def test_bypass_off_and_effort_profiles():
    mw, down, conn = make()
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer": "bypass", "X-Optimizer-Route": "r"})
    assert conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0 and down.decision_seen is None
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r", "Content-Length": "7"})
    assert dict(down.scope["headers"])[b"content-length"] == b"7"  # P0: the client's header is untouched
    db.set_pin(conn, "r", "P1")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r", "Content-Length": "7"})
    assert down.seen[-1]["output_config"] == {"effort": "low"} and down.seen[-1]["messages"] == ANTH["messages"]
    assert [k for k, _ in down.scope["headers"]].count(b"content-length") == 1  # a rewrite replaces the stale length
    assert dict(down.scope["headers"])[b"content-length"] == str(len(json.dumps(down.seen[-1]).encode())).encode()
    assert down.decision_seen["profile"] == "P1" and down.decision_seen["applied"] is True
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P1"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == ANTH
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r", "X-Optimizer": "off"})
    assert down.seen[-1] == ANTH
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P0"
    mw, down, conn = make()
    await call(mw, "/v1/chat/completions", CHAT, {"X-Optimizer-Route": "c"})
    db.set_pin(conn, "c", "P1b")
    await call(mw, "/v1/chat/completions", CHAT, {"X-Optimizer-Route": "c"})
    assert down.seen[-1]["verbosity"] == "low" and down.decision_seen["applied"] is True


@pytest.mark.anyio
async def test_shape_pin_is_recorded_as_p0_until_the_pipeline_applies_it():
    mw, down, conn = make()
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    db.set_pin(conn, "r", "P2")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.seen[-1] == ANTH  # the middleware never appends the shape itself
    assert down.decision_seen["profile"] == "P2" and down.decision_seen["applied"] is False
    assert down.decision_seen["target_words"] == 20 and down.decision_seen["exemplar"] is None
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P0"


@pytest.mark.anyio
async def test_rejections_after_a_rewrite_revert_at_three_and_4xx_at_p0_records_nothing():
    mw, down, conn = make(b'{"error":"bad"}', status=400)
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    db.set_pin(conn, "r", "P1")
    for _ in range(3):
        await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    r = db.get_route(conn, "r")
    assert r["pinned_profile"] == "P0" and r["status"] == "reverted" and r["rejections"] == 0
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


@pytest.mark.anyio
async def test_fails_open_logs_class_name_only_and_downstream_errors_propagate(caplog):
    down = Downstream(json.dumps(ANTH_RESP).encode())
    mw = PithMiddleware(down, Runtime(Config(), Boom()))
    with caplog.at_level(logging.WARNING, logger="pith.headroom"):
        sent, _ = await call(mw, "/v1/messages", ANTH)
    assert down.seen[-1] == ANTH and sent[0]["status"] == 200 and down.decision_seen is None
    assert "sk-secret" not in caplog.text and "RuntimeError" in caplog.text
    mw, down, conn = make(raise_exc=ValueError("headroom's own problem"))
    with pytest.raises(ValueError):
        await call(mw, "/v1/messages", ANTH)
    mw, down, conn = make()
    sent, _ = await call(mw, "/v1/messages", "not json")  # the body is a JSON string, not an object
    assert sent[0]["status"] == 200 and conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 0


@pytest.mark.anyio
async def test_receive_delegates_to_the_original_after_the_buffered_body():
    class Waits(Downstream):
        async def __call__(self, scope, receive, send):
            await super().__call__(scope, receive, send)
            self.extra = await receive()
    conn = db.connect(":memory:")
    down = Waits(json.dumps(ANTH_RESP).encode())
    mw = PithMiddleware(down, Runtime(Config(sample_rate=0), conn))
    _, delegated = await call(mw, "/v1/messages", ANTH)
    assert down.extra == {"type": "http.disconnect"} and delegated == [1]


def test_runtime_is_lazy_reads_env_and_warms_tiktoken_off_thread(tmp_path, monkeypatch):
    warmed = []
    monkeypatch.setattr("pith.headroom.estimate_tokens", lambda text: warmed.append((text, threading.current_thread())))
    rt = Runtime(env={"OPTIMIZER_DB_PATH": str(tmp_path / "h.db"), "OPTIMIZER_SAMPLE_RATE": "1"})
    cfg, conn = rt.ready()
    assert cfg.db_path == str(tmp_path / "h.db") and cfg.sample_rate == 1.0 and (tmp_path / "h.db").exists()
    assert rt.ready() == (cfg, conn)
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(1)
    assert len(warmed) == 1 and warmed[0][0] == "warm" and warmed[0][1] is not threading.current_thread()


class Event:
    def __init__(self, stage, messages, provider="anthropic"):
        self.stage, self.messages, self.provider = SimpleNamespace(name=stage), messages, provider


@pytest.mark.anyio
async def test_pipeline_applies_shape_at_pre_send_and_the_request_records_as_pinned():
    mw, down, conn = make()
    pipe = PithPipeline()
    down.hook = lambda body: pipe.on_pipeline_event(Event("PRE_SEND", body["messages"]))
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.seen[-1] == ANTH  # P0: PRE_SEND leaves the messages alone
    db.set_pin(conn, "r", "P2")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    last = down.seen[-1]["messages"][-1]
    assert last["content"][0] == {"type": "text", "text": "q"} and last["content"][1]["text"].startswith("Answer directly.")
    assert down.seen[-1]["system"] == "S" and "output_config" not in down.seen[-1]
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P2"
    assert json.loads(conn.execute("SELECT request_json FROM bodies ORDER BY id DESC").fetchone()[0]) == ANTH
    db.set_pin(conn, "r", "P4")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.seen[-1]["output_config"] == {"effort": "low"}
    assert down.seen[-1]["messages"][-1]["content"][1]["text"].startswith("Answer directly.")
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P4"


def test_pipeline_ignores_other_stages_missing_decisions_and_fails_open(caplog):
    pipe = PithPipeline()
    msgs = [{"role": "user", "content": "q"}]
    DECISION.set({"route": "r", "profile": "P2", "applied": False, "request_json": "{}", "t0": 0.0, "target_words": 20,
                  "exemplar": None})
    assert pipe.on_pipeline_event(Event("POST_SEND", msgs)) is None and msgs == [{"role": "user", "content": "q"}]
    pipe.on_pipeline_event(Event("PRE_SEND", []))
    assert DECISION.get()["applied"] is False
    with caplog.at_level(logging.WARNING, logger="pith.headroom"):
        pipe.on_pipeline_event(Event("PRE_SEND", ["not a message dict"]))
    assert DECISION.get()["applied"] is False and "AttributeError" in caplog.text
    DECISION.set(None)
    pipe.on_pipeline_event(Event("PRE_SEND", msgs))
    assert msgs == [{"role": "user", "content": "q"}]
    pipe.on_pipeline_event(object())  # an event without the expected attributes is ignored


def test_install_adds_the_middleware_and_module_needs_no_headroom():
    added = []
    install(SimpleNamespace(add_middleware=lambda cls, **kw: added.append((cls, kw))), config=None)
    assert added == [(PithMiddleware, {})]
    assert "headroom" not in sys.modules


@pytest.mark.anyio
async def test_headroom_cache_hit_records_nothing():
    mw, down, conn = make()
    pipe = PithPipeline()
    down.hook = lambda body: pipe.on_pipeline_event(Event("INPUT_CACHED", body["messages"]))
    db.set_pin(conn, "r", "P2")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.decision_seen["cached"] is True and down.seen[-1] == ANTH
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0 and db.get_route(conn, "r") is not None


@pytest.mark.anyio
async def test_rejections_count_only_for_param_profiles_and_not_for_prompt_size_errors():
    mw, down, conn = make(b'{"error":"bad"}', status=400)
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    db.set_pin(conn, "r", "P2")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert db.get_route(conn, "r")["rejections"] == 0  # a user-text append is not plausibly the cause
    mw, down, conn = make(b'{"error":{"message":"Prompt is too long"}}', status=400)
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    db.set_pin(conn, "r", "P1")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert db.get_route(conn, "r")["rejections"] == 0


@pytest.mark.anyio
async def test_unapplied_p4_records_as_p1_because_the_effort_was_rewritten():
    mw, down, conn = make()
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    db.set_pin(conn, "r", "P4")
    await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r"})
    assert down.seen[-1]["output_config"]["effort"] == "low"
    assert conn.execute("SELECT profile FROM requests ORDER BY id DESC").fetchone()["profile"] == "P1"


def test_pipeline_applied_only_when_appended_and_idempotent():
    pipe = PithPipeline()
    DECISION.set({"route": "r", "profile": "P2", "applied": False, "request_json": "{}", "t0": 0.0, "target_words": 20,
                  "exemplar": None})
    pipe.on_pipeline_event(Event("PRE_SEND", [{"role": "assistant", "content": "a"}]))
    assert DECISION.get()["applied"] is False
    msgs = [{"role": "user", "content": "q"}]
    pipe.on_pipeline_event(Event("PRE_SEND", msgs))
    pipe.on_pipeline_event(Event("PRE_SEND", msgs))  # a doubly registered extension
    assert DECISION.get()["applied"] is True and len(msgs[0]["content"]) == 2
    DECISION.set(None)


@pytest.mark.anyio
async def test_a_stream_that_ends_without_a_final_body_message_records_nothing():
    class Partial(Downstream):
        async def __call__(self, scope, receive, send):
            await receive()
            self.decision_seen = DECISION.get()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": CHAT_SSE[:40], "more_body": True})
    conn = db.connect(":memory:")
    down = Partial()
    mw = PithMiddleware(down, Runtime(Config(sample_rate=1.0), conn))
    await call(mw, "/v1/chat/completions", dict(CHAT, stream=True))
    assert down.decision_seen is not None and conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


@pytest.mark.anyio
async def test_pith_headers_are_stripped_before_headroom_forwards():
    mw, down, conn = make()
    for mode in ("off", "bypass"):
        await call(mw, "/v1/messages", ANTH, {"X-Optimizer-Route": "r", "X-Optimizer": mode, "X-Api-Key": "k",
                                              "Anthropic-Version": "2023-06-01"})
        names = [k for k, _ in down.scope["headers"]]
        assert b"x-optimizer" not in names and b"x-optimizer-route" not in names
        assert dict(down.scope["headers"])[b"x-api-key"] == b"k" and b"anthropic-version" in names
