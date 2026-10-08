"""Control plane: eligibility, sampling, sweeps, pin rule, recheck. Spec (Plan 2) §4-§9."""
import json
import statistics

from optimizer import db
from optimizer.config import Config

PROFILES_BY_PROVIDER = {"anthropic": ("P0", "P1", "P2", "P3", "P4"), "openai": ("P0", "P1", "P1b", "P2", "P3", "P4")}
TEXT_GATE = 0.8
SWEEPABLE = ("observing", "reverted")


def eligible_routes(conn, cfg: Config, sample_n: int = 50, only: str | None = None) -> list[dict]:
    keys = [only] if only else [r["key"] for r in conn.execute("SELECT key FROM routes ORDER BY last_seen DESC")]
    out = []
    for key in keys:
        route = db.get_route(conn, key)
        if route is None or route["status"] == "sweeping" or (not only and route["status"] not in SWEEPABLE):
            continue
        s = db.route_request_stats(conn, key)
        if s["p0_total"] >= sample_n and (s["text_frac"] or 0) < TEXT_GATE:
            db.set_route_fields(conn, key, status="not-applicable", eligible=0)
            continue
        if s["p0_sampled"] < sample_n or (s["text_frac"] or 0) < TEXT_GATE:
            continue
        db.set_route_fields(conn, key, eligible=1)
        out.append(db.get_route(conn, key))
    return out


def stratify(candidates: list[dict], n: int) -> list[dict]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c["output_tokens"] or 0)
    size = max(1, -(-len(ordered) // 5))
    buckets = [ordered[i:i + size] for i in range(0, len(ordered), size)]
    picked = []
    while len(picked) < n and any(buckets):
        for b in buckets:
            if b and len(picked) < n:
                picked.append(b.pop(0))
    return picked


def pick_sample(conn, key: str, n: int) -> list[dict]:
    items = []
    for c in stratify(db.sample_candidates(conn, key), n):
        body = json.loads(c["request_json"])
        body.pop("stream", None)
        items.append({"id": c["id"], "body": body})
    return items


def derive_targets(p0_texts: list[str]) -> tuple[int, str]:
    texts = [t for t in p0_texts if t.strip()]
    if not texts:
        return 20, ""
    median = statistics.median(len(t.split()) for t in texts)
    return max(20, round(0.5 * median)), min(texts, key=lambda t: len(t.split()))
