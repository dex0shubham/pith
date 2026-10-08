"""Per-route report: baseline vs pinned profile, $/1k requests, savings. Spec §3."""
import html

from optimizer import db

# $/M tokens (input, output). Claude from the Anthropic pricing docs (2026-09); OpenAI entries added as customers need them.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0), "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _usd_per_1k(model: str, avg_in, avg_out):
    p = PRICES.get(model)
    if not p or avg_in is None or avg_out is None:
        return None
    return (avg_in * p[0] + avg_out * p[1]) / 1e6 * 1000


def route_rows(conn) -> list[dict]:
    stats = {}
    for s in db.route_stats(conn):
        stats.setdefault(s["route_key"], {})[s["profile"]] = s
    rows = []
    for key in [r["key"] for r in conn.execute("SELECT key FROM routes ORDER BY last_seen DESC")]:
        route = db.get_route(conn, key)
        base = stats.get(key, {}).get("P0", {})
        pinned = stats.get(key, {}).get(route["pinned_profile"], {}) if route["pinned_profile"] != "P0" else {}
        b_usd = _usd_per_1k(route["model"], base.get("avg_input"), base.get("avg_output"))
        p_usd = _usd_per_1k(route["model"], pinned.get("avg_input"), pinned.get("avg_output"))
        rows.append({
            "key": key, "provider": route["provider"], "model": route["model"], "name": route["name"],
            "status": route["status"], "pinned_profile": route["pinned_profile"],
            "baseline_n": base.get("n", 0), "baseline_avg_output": base.get("avg_output"),
            "pinned_n": pinned.get("n", 0), "pinned_avg_output": pinned.get("avg_output"),
            "baseline_usd_per_1k": b_usd, "pinned_usd_per_1k": p_usd,
            "estimated_savings_pct": (100 * (1 - p_usd / b_usd)) if b_usd and p_usd else None,
            "last_seen": route["last_seen"],
        })
    return rows


COLS = ("key", "name", "model", "status", "pinned_profile", "baseline_n", "baseline_avg_output", "pinned_n",
        "pinned_avg_output", "baseline_usd_per_1k", "pinned_usd_per_1k", "estimated_savings_pct")


def render_html(rows: list[dict]) -> str:
    def cell(v):
        return "" if v is None else (f"{v:.2f}" if isinstance(v, float) else html.escape(str(v)))
    head = "".join(f"<th>{c}</th>" for c in COLS)
    body = "".join("<tr>" + "".join(f"<td>{cell(r[c])}</td>" for c in COLS) + "</tr>" for r in rows)
    return (f"<!doctype html><title>Output optimizer report</title>"
            f"<style>body{{font-family:system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}</style>"
            f"<h1>Routes</h1><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
