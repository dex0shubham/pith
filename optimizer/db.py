"""SQLite state: schema from spec §10 plus the few queries the proxy and report need."""
import json
import sqlite3
import time
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS routes(
  key TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, system_hash TEXT NOT NULL,
  name TEXT, status TEXT NOT NULL DEFAULT 'observing', pinned_profile TEXT NOT NULL DEFAULT 'P0',
  injection_form TEXT NOT NULL DEFAULT 'system', target_words INTEGER NOT NULL DEFAULT 20,
  exemplar TEXT, eligible INTEGER, rejections INTEGER NOT NULL DEFAULT 0,
  first_seen REAL NOT NULL, last_seen REAL NOT NULL, last_sweep_id INTEGER);
CREATE TABLE IF NOT EXISTS requests(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, route_key TEXT NOT NULL, profile TEXT NOT NULL,
  input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER, cache_create INTEGER,
  estimated INTEGER NOT NULL DEFAULT 0, stop_reason TEXT, latency_ms INTEGER, body_ref INTEGER);
CREATE INDEX IF NOT EXISTS requests_route ON requests(route_key, profile);
CREATE TABLE IF NOT EXISTS bodies(
  id INTEGER PRIMARY KEY, request_json TEXT NOT NULL, response_json TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS samples(
  id INTEGER PRIMARY KEY, route_key TEXT NOT NULL, created_at REAL NOT NULL, item_ids_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sweeps(
  id INTEGER PRIMARY KEY, route_key TEXT NOT NULL, sample_id INTEGER NOT NULL, judge_model TEXT,
  judge_prompt_version TEXT, started_at REAL, finished_at REAL, cost_usd REAL, result_json TEXT, winner TEXT);
CREATE TABLE IF NOT EXISTS judgments(
  id INTEGER PRIMARY KEY, sweep_id INTEGER NOT NULL, item_id INTEGER NOT NULL, profile TEXT NOT NULL,
  trial INTEGER NOT NULL, label TEXT NOT NULL, order_ab TEXT);
CREATE TABLE IF NOT EXISTS shadow(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, route_key TEXT NOT NULL, request_id INTEGER, label TEXT NOT NULL);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # note: WAL+NORMAL only; batching/thread offload if p50 overhead exceeds 5 ms
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


TEXT_STOPS = ("end_turn", "stop", "completed")
ROUTE_FIELDS = ("status", "eligible", "target_words", "exemplar", "last_sweep_id")


def upsert_route(conn, key, provider, model, system_hash, name=None, now=None):
    now = time.time() if now is None else now
    # A changed prompt/tools hash or a changed model means the pinned profile was tuned for a different route: unpin, keylessly.
    conn.execute(
        "INSERT INTO routes(key, provider, model, system_hash, name, first_seen, last_seen) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET last_seen=excluded.last_seen, model=excluded.model, "
        "system_hash=excluded.system_hash, "
        "pinned_profile=CASE WHEN routes.system_hash != excluded.system_hash OR routes.model != excluded.model THEN 'P0' ELSE routes.pinned_profile END, "
        "status=CASE WHEN routes.system_hash != excluded.system_hash OR routes.model != excluded.model THEN 'observing' ELSE routes.status END",
        (key, provider, model, system_hash, name, now, now))
    conn.commit()


def get_route(conn, key):
    row = conn.execute("SELECT * FROM routes WHERE key=?", (key,)).fetchone()
    return dict(row) if row else None


def set_pin(conn, key, profile, status="pinned"):
    conn.execute("UPDATE routes SET pinned_profile=?, status=?, rejections=0 WHERE key=?", (profile, status, key))
    conn.commit()


def set_injection_form(conn, key, form):
    conn.execute("UPDATE routes SET injection_form=? WHERE key=?", (form, key))
    conn.commit()


def bump_rejection(conn, key) -> int:
    conn.execute("UPDATE routes SET rejections=rejections+1 WHERE key=?", (key,))
    conn.commit()
    return conn.execute("SELECT rejections FROM routes WHERE key=?", (key,)).fetchone()[0]


def record_request(conn, *, ts, route_key, profile, input_tokens, output_tokens, cache_read, cache_create,
                   estimated, stop_reason, latency_ms, body_ref=None) -> int:
    cur = conn.execute(
        "INSERT INTO requests(ts, route_key, profile, input_tokens, output_tokens, cache_read, cache_create, "
        "estimated, stop_reason, latency_ms, body_ref) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (ts, route_key, profile, input_tokens, output_tokens, cache_read, cache_create, int(bool(estimated)),
         stop_reason, latency_ms, body_ref))
    conn.commit()
    return cur.lastrowid


def store_body(conn, request_json: str, response_json: str, expires_at: float) -> int:
    cur = conn.execute("INSERT INTO bodies(request_json, response_json, expires_at) VALUES(?,?,?)",
                       (request_json, response_json, expires_at))
    conn.commit()
    return cur.lastrowid


def purge_expired(conn, now: float) -> int:
    cur = conn.execute("DELETE FROM bodies WHERE expires_at <= ?", (now,))
    conn.commit()
    return cur.rowcount


def route_stats(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT route_key, profile, COUNT(*) AS n, AVG(output_tokens) AS avg_output, AVG(input_tokens) AS avg_input, "
        "AVG(cache_read) AS avg_cache_read, "
        "SUM(CASE WHEN stop_reason IN ('max_tokens','length','max_output_tokens') THEN 1 ELSE 0 END) AS max_tokens_stops "
        "FROM requests GROUP BY route_key, profile").fetchall()
    return [dict(r) for r in rows]


def route_request_stats(conn, key) -> dict:
    total, sampled, text = conn.execute(
        "SELECT COUNT(*), SUM(body_ref IS NOT NULL), SUM(stop_reason IN (?,?,?)) FROM requests "
        "WHERE route_key=? AND profile='P0'", (*TEXT_STOPS, key)).fetchone()
    return {"p0_total": total, "p0_sampled": sampled or 0, "text_frac": (text or 0) / total if total else None}


def sample_candidates(conn, key, limit=500) -> list[dict]:
    rows = conn.execute(
        "SELECT r.id, r.output_tokens, b.request_json FROM requests r JOIN bodies b ON b.id = r.body_ref "
        "WHERE r.route_key=? AND r.profile='P0' AND r.stop_reason IN (?,?,?) ORDER BY r.ts DESC LIMIT ?",
        (key, *TEXT_STOPS, limit)).fetchall()
    return [dict(r) for r in rows]


def create_sample(conn, key, item_ids, now=None) -> int:
    cur = conn.execute("INSERT INTO samples(route_key, created_at, item_ids_json) VALUES(?,?,?)",
                       (key, time.time() if now is None else now, json.dumps(list(item_ids))))
    conn.commit()
    return cur.lastrowid


def create_sweep(conn, key, sample_id, judge_model, judge_prompt_version, now=None) -> int:
    cur = conn.execute(
        "INSERT INTO sweeps(route_key, sample_id, judge_model, judge_prompt_version, started_at, cost_usd) "
        "VALUES(?,?,?,?,?,0)", (key, sample_id, judge_model, judge_prompt_version, time.time() if now is None else now))
    conn.commit()
    return cur.lastrowid


def update_sweep_cost(conn, sweep_id, cost_usd):
    conn.execute("UPDATE sweeps SET cost_usd=? WHERE id=?", (cost_usd, sweep_id))
    conn.commit()


def finish_sweep(conn, sweep_id, cost_usd, result_json, winner, now=None):
    conn.execute("UPDATE sweeps SET cost_usd=?, result_json=?, winner=?, finished_at=? WHERE id=?",
                 (cost_usd, result_json, winner, time.time() if now is None else now, sweep_id))
    conn.commit()


def add_judgment(conn, sweep_id, item_id, profile, trial, label, order_ab):
    conn.execute("INSERT INTO judgments(sweep_id, item_id, profile, trial, label, order_ab) VALUES(?,?,?,?,?,?)",
                 (sweep_id, item_id, profile, trial, label, order_ab))
    conn.commit()


def month_sweep_cost(conn, now) -> float:
    """Sweep spend since the start of the UTC month containing `now`, not counting anything after `now`."""
    start = datetime.fromtimestamp(now, timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM sweeps WHERE started_at >= ? AND started_at <= ?",
                       (start, now)).fetchone()
    return float(row[0])


def set_route_fields(conn, key, **fields):
    bad = set(fields) - set(ROUTE_FIELDS)
    if bad:
        raise ValueError(f"not settable here: {sorted(bad)}")
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE routes SET {sets} WHERE key=?", (*fields.values(), key))
    conn.commit()


def sweeps_for_route(conn, key) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM sweeps WHERE route_key=? ORDER BY id DESC", (key,))]


def recent_bodies(conn, key, profile, n) -> list[dict]:
    rows = conn.execute(
        "SELECT r.id, b.request_json, b.response_json FROM requests r JOIN bodies b ON b.id = r.body_ref "
        "WHERE r.route_key=? AND r.profile=? ORDER BY r.ts DESC LIMIT ?", (key, profile, n)).fetchall()
    return [dict(r) for r in rows]


def add_shadow(conn, key, request_id, label, now=None):
    conn.execute("INSERT INTO shadow(ts, route_key, request_id, label) VALUES(?,?,?,?)",
                 (time.time() if now is None else now, key, request_id, label))
    conn.commit()


def shadow_rate(conn, key, window=100):
    labels = [r[0] for r in conn.execute(
        "SELECT label FROM shadow WHERE route_key=? ORDER BY id DESC LIMIT ?", (key, window))]
    judged = [l for l in labels if l != "judge-error"]
    rate = (sum(l == "equivalent" for l in judged) / len(judged)) if judged else None
    return rate, len(labels)


def projected_monthly_volume(conn, key, now) -> int:
    n = conn.execute("SELECT COUNT(*) FROM requests WHERE route_key=? AND ts >= ?", (key, now - 7 * 86400)).fetchone()[0]
    return max(1000, round(n * 30 / 7))
