"""Control plane: eligibility, sampling, sweeps, pin rule, recheck. Spec (Plan 2) §4-§9."""
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass

from pith import db
from pith.config import Config
from pith.judge import JUDGE_PROMPT_VERSION, JudgeUnavailable, judge, last_user_text
from pith.replay import Reply, call, endpoint_for, stored_response_text
from pith.report import price_for
from pith.rewrite import RouteState, apply_profile, is_system_role_rejection

log = logging.getLogger("pith.sweep")

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
            db.set_pin(conn, key, "P0", status="not-applicable")
            db.set_route_fields(conn, key, eligible=0)
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
    for b in buckets:
        b.sort(key=lambda c: -c["id"])  # newest first within a quintile
    picked = []
    while len(picked) < n and any(buckets):
        for b in buckets:
            if b and len(picked) < n:
                picked.append(b.pop(0))
    return picked


def pick_sample(conn, key: str, n: int) -> list[dict]:
    cands = []
    for c in db.sample_candidates(conn, key):
        body = json.loads(c["request_json"])
        body.pop("stream", None)
        # server-side tools would execute and bill on replay
        if any(isinstance(t, dict) and t.get("type") not in (None, "function", "custom") for t in body.get("tools") or []):
            continue
        cands.append({"id": c["id"], "output_tokens": c["output_tokens"], "body": body})
    return [{"id": c["id"], "body": c["body"]} for c in stratify(cands, n)]


def derive_targets(p0_texts: list[str]) -> tuple[int, str]:
    texts = [t for t in p0_texts if t.strip()]
    if not texts:
        return 20, ""
    median = statistics.median(len(t.split()) for t in texts)
    return max(20, round(0.5 * median)), min(texts, key=lambda t: len(t.split()))


MECHANICAL_FAILS = ("max_tokens", "length", "max_output_tokens")
JUDGE_INPUT_OVERHEAD = 4000  # chars of request text the judge sees, used only to estimate judge cost
TRANSPORT_ABORT_FRAC = 0.2
BUDGET_CHECK_EVERY = 50


class NoPrice(Exception):
    pass


class BudgetRefused(Exception):
    def __init__(self, estimate: float, ceiling: float):
        super().__init__(f"estimated ${estimate:.4f} exceeds ceiling ${ceiling:.4f}")
        self.estimate, self.ceiling = estimate, ceiling


class SweepAborted(Exception):
    pass


class NothingToSample(Exception):
    pass


@dataclass
class SweepOutcome:
    winner: str | None
    table: dict
    floor: float | None
    cost_usd: float
    sweep_id: int | None
    target_words: int
    exemplar: str


def estimate_cost(route: dict, items, profiles, trials: int, cfg: Config, *, mean_input: float, mean_output: float) -> float:
    """Rough upper-ish estimate. Omits: cache tokens, the judge's retry second call, server-side tool billing."""
    p = price_for(route["model"], cfg.prices)
    jp = price_for(cfg.judge_model, cfg.prices)
    if not p:
        raise NoPrice(f"no price for model {route['model']!r}; add [prices.\"{route['model']}\"] to pith.toml")
    if not jp:
        raise NoPrice(f"no price for judge model {cfg.judge_model!r}")
    n = len(items)
    replays = n * trials * len(profiles) * (mean_input * p[0] + mean_output * p[1]) / 1e6
    candidates = len(profiles) - 1
    noise_pairs = 1
    judge_calls = n * trials * (candidates + noise_pairs)
    judges = judge_calls * ((JUDGE_INPUT_OVERHEAD + 2 * mean_output) * jp[0] + 8 * jp[1]) / 1e6
    return replays + judges


def pin_rule(table: dict, bar: float, floor: float | None) -> str | None:
    p0 = table["P0"]
    best = None
    for prof, row in table.items():
        if prof == "P0":
            row["qualifies"], row["reason"] = False, "baseline"
            continue
        reason = None
        if row.get("skipped"):
            reason = "skipped"
        elif row["judged"] and row["judge_error"] > 0.2 * row["judged"]:
            reason = "judge mostly errored"
        elif row["rate"] is None or row["rate"] < bar:
            reason = "rate below bar"
        elif floor is not None and row["rate"] < floor - 0.03:
            reason = "rate below noise floor - 0.03"
        elif row["stops"] > p0["stops"]:
            reason = "more max_tokens stops than P0"
        elif (p0["mean_cache_read"] or 0) > 0 and (row["mean_cache_read"] or 0) < p0["mean_cache_read"]:
            reason = "cache reads below P0"
        elif row["cost_per_request"] >= p0["cost_per_request"]:
            # distinguish "this profile costs more" from "it saves, but not enough to repay the sweep at this volume"
            raw = row.get("raw_cost_per_request", row["cost_per_request"])
            reason = "not cheaper than P0" if raw >= p0["cost_per_request"] else "sweep cost not recovered at projected volume"
        row["qualifies"], row["reason"] = reason is None, reason or "qualifies"
        if reason is None and (best is None or row["cost_per_request"] < table[best]["cost_per_request"]):
            best = prof
    return best


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def _mechanical_fail(r: Reply) -> bool:
    return r.status != 200 or r.usage.stop_reason in MECHANICAL_FAILS or not r.text.strip()


def run_sweep(conn, cfg: Config, route: dict, client, keys: dict, *, trials: int = 3, sample_n: int = 50,
              dry_run: bool = False, budget_usd: float | None = None, rng: random.Random | None = None,
              now: float | None = None) -> SweepOutcome:
    rng = rng or random.Random()
    now = time.time() if now is None else now
    key, provider = route["key"], route["provider"]
    pkey = keys.get(provider)
    if not pkey:
        raise SweepAborted(f"no API key for provider {provider!r}")
    profiles = PROFILES_BY_PROVIDER[provider]
    items = pick_sample(conn, key, sample_n)  # read-only until the budget gate passes
    if len(items) < max(1, sample_n // 2):
        raise NothingToSample(f"{len(items)} sampleable requests, need at least {sample_n // 2}")
    stats = conn.execute("SELECT AVG(input_tokens), AVG(output_tokens) FROM requests WHERE route_key=? AND profile='P0'",
                         (key,)).fetchone()
    estimate = estimate_cost(route, items, profiles, trials, cfg, mean_input=stats[0] or 0, mean_output=stats[1] or 0)
    # --budget-usd is an explicit per-run ceiling and overrides the monthly one; otherwise what is left of the month applies.
    ceiling = budget_usd if budget_usd is not None else cfg.sweep_budget_usd_month - db.month_sweep_cost(conn, now)
    if estimate > ceiling:
        raise BudgetRefused(estimate, ceiling)

    sample_id = None if dry_run else db.create_sample(conn, key, [it["id"] for it in items], now)
    sweep_id = None if dry_run else db.create_sweep(conn, key, sample_id, cfg.judge_model, JUDGE_PROMPT_VERSION, now)
    spent = {"usd": 0.0, "calls": 0}

    def account(cost: float):
        spent["usd"] += cost
        spent["calls"] += 1
        if spent["calls"] % BUDGET_CHECK_EVERY == 0:
            if sweep_id is not None:
                db.update_sweep_cost(conn, sweep_id, spent["usd"])
            if spent["usd"] > ceiling:
                raise SweepAborted(f"spent ${spent['usd']:.4f} over ceiling ${ceiling:.4f}")

    def replay_profile(profile: str, state: RouteState | None) -> tuple[dict[int, list[Reply]], bool]:
        """Returns {item_id: [reply per trial]} and whether the profile was skipped (no-op rewrite)."""
        replies: dict[int, list[Reply]] = {}
        failures = calls = 0
        bodies = {it["id"]: it["body"] for it in items}
        if state is not None:
            bodies = {it["id"]: apply_profile(provider, it["body"], state,
                                              responses_api=endpoint_for(provider, it["body"]) == "/v1/responses")
                      for it in items}
            if all(bodies[it["id"]] == it["body"] for it in items):
                return {}, True  # no-op for every item: skipped before anything is billed
        for t in range(trials):
            for it in items:
                body = bodies[it["id"]]
                r = call(client, cfg, provider, body, pkey, cfg.prices)
                account(r.cost_usd)
                if state is not None and state.injection_form == "system" and \
                        is_system_role_rejection(r.status, json.dumps(r.body or {})):
                    # Same fallback the proxy uses: this model rejects mid-conversation system messages.
                    # Switch the route to the user-text form and restart this profile; only the rejected call was made.
                    state.injection_form = route["injection_form"] = "user_text"
                    db.set_injection_form(conn, key, "user_text")
                    log.info("route %s: provider rejects role 'system'; switching to user_text form", key)
                    return replay_profile(profile, state)
                calls += 1
                failures += r.status == 0 or r.status >= 500 or r.status in (401, 403, 429)  # provider failures are transport-class
                replies.setdefault(it["id"], []).append(r)
        if calls and failures / calls > TRANSPORT_ABORT_FRAC:
            raise SweepAborted(f"{failures}/{calls} transport failures on {profile}")
        return replies, False

    def judged(item_id: int, profile: str, trial: int, label: str, order: str):
        if sweep_id is not None:
            db.add_judgment(conn, sweep_id, item_id, profile, trial, label, order)
        return label

    try:
        p0, _ = replay_profile("P0", None)
        questions = {it["id"]: last_user_text(provider, it["body"]) for it in items}
        baseline = {i: rs[0] for i, rs in p0.items()}
        target_words, exemplar = derive_targets([r.text for r in baseline.values() if not _mechanical_fail(r)])
        table: dict[str, dict] = {}

        def summarize(profile, replies, labels, skipped=False):
            flat = [r for rs in replies.values() for r in rs]
            judged_n = len(labels)
            errors = sum(l == "judge-error" for l in labels)
            eq = sum(l == "equivalent" for l in labels)
            rate = (eq / (judged_n - errors)) if judged_n - errors > 0 else None
            p = price_for(route["model"], cfg.prices)
            mi, mo = _mean([r.usage.input_tokens for r in flat]), _mean([r.usage.output_tokens for r in flat])
            table[profile] = {
                "n": len(replies), "skipped": skipped, "equivalent": eq, "judged": judged_n, "judge_error": errors,
                "stops": sum(r.usage.stop_reason in MECHANICAL_FAILS for r in flat),
                "mean_input": mi, "mean_output": mo, "mean_cache_read": _mean([r.usage.cache_read for r in flat]),
                "rate": rate, "cost_per_request": (mi * p[0] + mo * p[1]) / 1e6,
                "raw_cost_per_request": (mi * p[0] + mo * p[1]) / 1e6}

        noise = []
        for i, rs in p0.items():
            for t, r in enumerate(rs[1:], start=2):
                if _mechanical_fail(baseline[i]) or _mechanical_fail(r):
                    noise.append(judged(i, "P0", t, "judge-error", "n/a"))
                    continue
                label, order, cost = judge(client, cfg, keys, questions[i], baseline[i].text, r.text, rng, cfg.prices)
                account(cost)
                noise.append(judged(i, "P0", t, label, order))
        summarize("P0", p0, noise)
        floor = table["P0"]["rate"]

        for profile in profiles[1:]:
            state = RouteState(profile, route["injection_form"], target_words, exemplar)
            replies, skipped = replay_profile(profile, state)
            if skipped:
                summarize(profile, {}, [], skipped=True)
                continue
            labels = []
            for i, rs in replies.items():
                for t, r in enumerate(rs, start=1):
                    if _mechanical_fail(r):
                        labels.append(judged(i, profile, t, "format-broken", "n/a"))
                        continue
                    if _mechanical_fail(baseline[i]):
                        labels.append(judged(i, profile, t, "judge-error", "n/a"))
                        continue
                    label, order, cost = judge(client, cfg, keys, questions[i], baseline[i].text, r.text, rng, cfg.prices)
                    account(cost)
                    labels.append(judged(i, profile, t, label, order))
            summarize(profile, replies, labels)
        # Amortize the whole sweep's spend over the route's projected monthly volume, equally across candidates.
        amort = spent["usd"] / db.projected_monthly_volume(conn, key, now)
        for prof, row in table.items():
            if prof != "P0" and not row["skipped"]:
                row["cost_per_request"] += amort
    except JudgeUnavailable as exc:
        if sweep_id is not None:
            db.update_sweep_cost(conn, sweep_id, spent["usd"])
        raise SweepAborted(f"judge unavailable: {exc}") from exc
    except BaseException:
        if sweep_id is not None:
            db.update_sweep_cost(conn, sweep_id, spent["usd"])
        raise

    route_cfg = cfg.routes.get(key)
    bar = route_cfg.equivalence_bar if route_cfg and route_cfg.equivalence_bar is not None else cfg.equivalence_bar
    if trials >= 2 and floor is None:  # every noise pair errored: no baseline consistency to compare against
        winner = None
        for prof, row in table.items():
            row["qualifies"], row["reason"] = False, "baseline" if prof == "P0" else "noise floor undefined"
    else:
        winner = pin_rule(table, bar, floor)
    result = {"table": table, "floor": floor, "bar": bar, "target_words": target_words, "exemplar": exemplar,
              "trials": trials, "sample_n": len(items)}
    if sweep_id is not None:
        db.finish_sweep(conn, sweep_id, spent["usd"], json.dumps(result), winner or "P0", now)
        if winner:
            db.set_route_fields(conn, key, target_words=target_words, exemplar=exemplar, last_sweep_id=sweep_id)
            db.set_pin(conn, key, winner)
        else:
            db.set_pin(conn, key, "P0", status="no-savings")
            db.set_route_fields(conn, key, last_sweep_id=sweep_id)
    return SweepOutcome(winner, table, floor, spent["usd"], sweep_id, target_words, exemplar)


RECHECK_MIN_ROWS = 20


@dataclass
class RecheckOutcome:
    judged: int
    rate: float | None
    reverted: bool


def recheck(conn, cfg: Config, route: dict, client, keys: dict, *, n: int = 20, rng: random.Random | None = None,
            now: float | None = None) -> RecheckOutcome:
    rng = rng or random.Random()
    now = time.time() if now is None else now
    key, provider, pinned = route["key"], route["provider"], route["pinned_profile"]
    if pinned == "P0":
        return RecheckOutcome(0, None, False)
    pkey = keys.get(provider)
    if not pkey:
        raise SweepAborted(f"no API key for provider {provider!r}")
    judged_n = 0
    try:
        for row in db.recent_bodies(conn, key, pinned, n):
            body = json.loads(row["request_json"])
            body.pop("stream", None)
            live = stored_response_text(provider, row["response_json"])
            p0 = call(client, cfg, provider, body, pkey, cfg.prices)
            if p0.status == 0:  # transport failure: this pair cannot be judged; write nothing (spec §11)
                continue
            if _mechanical_fail(p0) or not live.strip():
                db.add_shadow(conn, key, row["id"], "judge-error", now)
                continue
            label, _, _ = judge(client, cfg, keys, last_user_text(provider, body), p0.text, live, rng, cfg.prices)
            db.add_shadow(conn, key, row["id"], label, now)
            judged_n += 1
    except JudgeUnavailable as exc:
        raise SweepAborted(f"judge unavailable: {exc}") from exc
    rate, total = db.shadow_rate(conn, key)
    route_cfg = cfg.routes.get(key)
    bar = route_cfg.equivalence_bar if route_cfg and route_cfg.equivalence_bar is not None else cfg.equivalence_bar
    reverted = False
    if total >= RECHECK_MIN_ROWS and rate is not None and rate < bar:
        db.set_pin(conn, key, "P0", status="reverted")
        reverted = True
    return RecheckOutcome(judged_n, rate, reverted)
