"""Load optimizer.toml and OPTIMIZER_* environment overrides into a Config."""
import os
import tomllib
from dataclasses import dataclass, field, fields
from typing import Mapping


@dataclass(frozen=True)
class RouteConfig:
    enabled: bool = True
    equivalence_bar: float | None = None


@dataclass
class Config:
    listen: str = "0.0.0.0:8787"
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"
    db_path: str = "./optimizer.db"
    sample_rate: float = 0.05
    retention_days: int = 14
    sweep_budget_usd_month: float = 0.0
    equivalence_bar: float = 0.95
    judge_model: str = "claude-sonnet-5-5"
    judge_provider: str = "anthropic"
    enabled: bool = True
    routes: dict[str, RouteConfig] = field(default_factory=dict)
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)  # $/M tokens (input, output)


def _coerce(kind, raw: str):
    if kind is bool:
        return raw.strip().lower() not in ("0", "false", "no", "off", "")
    return kind(raw)


def load_config(path: str | None = None, env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    data: dict = {}
    if path:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    cfg = Config()
    scalar = {f.name: f.type for f in fields(Config) if f.name not in ("routes", "prices")}
    for name, kind in scalar.items():
        if name in data:
            setattr(cfg, name, kind(data[name]))
        raw = env.get(f"OPTIMIZER_{name.upper()}")
        if raw is not None:
            setattr(cfg, name, _coerce(kind, raw))
    for key, rc in (data.get("routes") or {}).items():
        cfg.routes[key] = RouteConfig(enabled=bool(rc.get("enabled", True)),
                                      equivalence_bar=rc.get("equivalence_bar"))
    for model, p in (data.get("prices") or {}).items():
        cfg.prices[model] = (float(p["input"]), float(p["output"]))
    return cfg
