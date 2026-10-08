import argparse
import logging
import os
import sys

import httpx

from optimizer import db
from optimizer.config import load_config
from optimizer.sweep import BudgetRefused, NoPrice, NothingToSample, SweepAborted, SweepOutcome, eligible_routes, recheck, run_sweep

ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def keys_from_env(env) -> dict:
    return {p: env[var] for p, var in ENV_KEYS.items() if env.get(var)}


def format_table(route_key: str, out: SweepOutcome) -> str:
    lines = [f"route {route_key}: winner: {out.winner or 'none (no-savings)'}  floor={out.floor}  cost=${out.cost_usd:.4f}  "
             f"target_words={out.target_words}"]
    lines.append(f"  {'profile':8} {'rate':>6} {'$/req':>10} {'out_tok':>8} {'stops':>5}  verdict")
    for prof, row in out.table.items():
        rate = "-" if row.get("rate") is None else f"{row['rate']:.2f}"
        lines.append(f"  {prof:8} {rate:>6} {row.get('cost_per_request', 0):>10.6f} {row.get('mean_output', 0):>8.1f} "
                     f"{row.get('stops', 0):>5}  {row.get('reason', '')}")
    return "\n".join(lines)


def _sweep(args, cfg, conn, client, env) -> int:
    keys = keys_from_env(env)
    if cfg.judge_provider not in keys:
        print(f"no judge key: set {ENV_KEYS[cfg.judge_provider]}")
        return 2
    routes = eligible_routes(conn, cfg, sample_n=args.sample, only=args.route)
    if not routes:
        print(f"no eligible routes in {cfg.db_path}" + (f" (route {args.route!r} not eligible or unknown)" if args.route else ""))
        return 0
    rc = 0
    remaining = args.budget_usd  # per-RUN ceiling: each swept route draws it down
    for route in routes:
        if route["provider"] not in keys:
            print(f"route {route['key']}: skipped, set {ENV_KEYS[route['provider']]}")
            rc = 2
            continue
        try:
            out = run_sweep(conn, cfg, route, client, keys, trials=args.trials, sample_n=args.sample,
                            dry_run=args.dry_run, budget_usd=remaining)
        except (BudgetRefused, NothingToSample) as e:
            print(f"route {route['key']}: refused — {e}")
            rc = 2
            continue
        except NoPrice as e:
            print(f"route {route['key']}: refused — {e}")
            rc = 2
            continue
        except SweepAborted as e:
            print(f"route {route['key']}: aborted — {e}")
            return 1
        if remaining is not None:
            remaining = max(0.0, remaining - out.cost_usd)
        print(format_table(route["key"], out) + ("  [dry-run: nothing written]" if args.dry_run else ""))
    return rc


def _recheck(args, cfg, conn, client, env) -> int:
    keys = keys_from_env(env)
    if cfg.judge_provider not in keys:
        print(f"no judge key: set {ENV_KEYS[cfg.judge_provider]}")
        return 2
    rows = conn.execute("SELECT key FROM routes WHERE pinned_profile != 'P0'" + (" AND key=?" if args.route else ""),
                        ((args.route,) if args.route else ())).fetchall()
    for (key,) in rows:
        route = db.get_route(conn, key)
        if route["provider"] not in keys:
            print(f"route {key}: skipped, set {ENV_KEYS[route['provider']]}")
            continue
        try:
            out = recheck(conn, cfg, route, client, keys, n=args.n)
        except SweepAborted as e:
            print(f"route {key}: aborted — {e}")
            return 1
        print(f"route {key}: judged={out.judged} rate={out.rate} reverted={out.reverted}")
    return 0


def main(argv=None, env=None, conn=None, client=None) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(prog="optimizer")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("serve", "sweep", "recheck"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None, help="path to optimizer.toml (env OPTIMIZER_* overrides it)")
        if name in ("sweep", "recheck"):
            p.add_argument("--route", default=None)
        if name == "sweep":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--budget-usd", type=float, default=None)
            p.add_argument("--trials", type=int, default=3)
            p.add_argument("--sample", type=int, default=50)
        if name == "recheck":
            p.add_argument("--n", type=int, default=20)
    args = ap.parse_args(argv)
    cmd = args.cmd or "serve"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(getattr(args, "config", None), env)
    conn = conn or db.connect(cfg.db_path)
    if cmd == "serve":
        import uvicorn
        from optimizer.proxy import create_app
        host, _, port = cfg.listen.rpartition(":")
        uvicorn.run(create_app(cfg, conn), host=host or "0.0.0.0", port=int(port), log_level="info")
        return 0
    client = client or httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
    return _sweep(args, cfg, conn, client, env) if cmd == "sweep" else _recheck(args, cfg, conn, client, env)


if __name__ == "__main__":
    sys.exit(main())
