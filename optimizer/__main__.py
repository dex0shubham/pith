import argparse
import logging

import uvicorn

from optimizer import db
from optimizer.config import load_config
from optimizer.proxy import create_app


def main():
    ap = argparse.ArgumentParser(prog="optimizer")
    ap.add_argument("--config", default=None, help="path to optimizer.toml (env OPTIMIZER_* overrides it)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(args.config)
    host, _, port = cfg.listen.rpartition(":")
    uvicorn.run(create_app(cfg, db.connect(cfg.db_path)), host=host or "0.0.0.0", port=int(port), log_level="info")


if __name__ == "__main__":
    main()
