"""One synchronous provider call for the control plane, plus response-text extraction and per-call cost."""
import json
import logging
import time
from dataclasses import dataclass

import httpx

from optimizer.config import Config
from optimizer.providers import upstream
from optimizer.report import price_for
from optimizer.usage import Usage, usage_from_body

log = logging.getLogger("optimizer.replay")


@dataclass
class Reply:
    status: int          # 0 = transport failure
    body: dict | None
    text: str
    usage: Usage
    cost_usd: float


def endpoint_for(provider: str, body: dict) -> str:
    if provider == "anthropic":
        return "/v1/messages"
    return "/v1/chat/completions" if "messages" in body else "/v1/responses"


def auth_headers(provider: str, key: str) -> dict:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"authorization": f"Bearer {key}"}


def response_text(provider: str, body: dict) -> str:
    if provider == "anthropic":
        return "".join(b.get("text", "") for b in body.get("content") or [] if b.get("type") == "text")
    if body.get("object") == "response" or "output" in body or "output_text" in body:
        if body.get("output_text"):
            return body["output_text"]
        parts = []
        for item in body.get("output") or []:
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
        return "".join(parts)
    choices = body.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


def sse_text(provider: str, raw: str) -> str:
    parts = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        if provider == "anthropic":
            d = ev.get("delta") or {}
            if ev.get("type") == "content_block_delta" and d.get("type") == "text_delta":
                parts.append(d.get("text", ""))
        else:
            for c in ev.get("choices") or []:
                parts.append((c.get("delta") or {}).get("content") or "")
            if ev.get("type") == "response.output_text.delta":
                parts.append(ev.get("delta", ""))
    return "".join(parts)


def stored_response_text(provider: str, response_json: str) -> str:
    s = response_json.lstrip()
    if s.startswith("{"):
        try:
            return response_text(provider, json.loads(s))
        except Exception:
            return ""
    if s.startswith(("event:", "data:")):
        return sse_text(provider, s)
    return ""


def cost_of(model: str, usage: Usage, prices) -> float:
    p = price_for(model, prices)
    if not p or usage.input_tokens is None or usage.output_tokens is None:
        return 0.0
    return (usage.input_tokens * p[0] + usage.output_tokens * p[1]) / 1e6


def call(client: httpx.Client, cfg: Config, provider: str, body: dict, key: str, prices=None, sleep=time.sleep) -> Reply:
    body = {k: v for k, v in body.items() if k != "stream"}
    url = upstream(provider, cfg) + endpoint_for(provider, body)
    headers = {**auth_headers(provider, key), "content-type": "application/json"}
    empty = Usage(None, None, None, None, None)
    try:
        resp = client.post(url, headers=headers, content=json.dumps(body).encode())
        if resp.status_code == 429:
            # note: retry-after honoured only in its numeric form; an HTTP-date falls back to 5 s.
            try:
                wait = min(60.0, float(resp.headers.get("retry-after", "5")))
            except ValueError:
                wait = 5.0
            sleep(wait)
            resp = client.post(url, headers=headers, content=json.dumps(body).encode())
    except httpx.HTTPError as exc:
        log.warning("replay to %s failed: %r", url, exc)
        return Reply(0, None, "", empty, 0.0)
    try:
        data = resp.json()
    except Exception:
        data = None
    if resp.status_code >= 400 or not isinstance(data, dict):
        return Reply(resp.status_code, data if isinstance(data, dict) else None, "", empty, 0.0)
    usage = usage_from_body(provider, data)
    return Reply(resp.status_code, data, response_text(provider, data), usage, cost_of(body.get("model", ""), usage, prices))
