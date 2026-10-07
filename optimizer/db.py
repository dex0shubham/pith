"""SQLite state: schema from spec §10 plus the few queries the proxy and report need."""
import sqlite3
import time

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


def upsert_route(conn, key, provider, model, system_hash, name=None, now=None):
    now = time.time() if now is None else now
    conn.execute(
        "INSERT INTO routes(key, provider, model, system_hash, name, first_seen, last_seen) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET last_seen=excluded.last_seen",
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
