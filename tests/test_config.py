from optimizer.config import Config, RouteConfig, load_config


def test_defaults_match_spec():
    c = load_config(None, env={})
    assert c.listen == "0.0.0.0:8787"
    assert c.anthropic_upstream == "https://api.anthropic.com"
    assert c.openai_upstream == "https://api.openai.com"
    assert c.db_path == "./optimizer.db"
    assert c.sample_rate == 0.05
    assert c.shadow_rate == 0.02
    assert c.retention_days == 14
    assert c.sweep_budget_usd_month == 0
    assert c.equivalence_bar == 0.95
    assert c.judge_model == "claude-sonnet-5-5"
    assert c.judge_provider == "anthropic"
    assert c.enabled is True
    assert c.routes == {}


def test_toml_and_route_overrides(tmp_path):
    p = tmp_path / "optimizer.toml"
    p.write_text(
        'listen = "127.0.0.1:9000"\nsample_rate = 0.5\n'
        '[routes."anthropic:claude-opus-5-5:abc"]\nenabled = false\nequivalence_bar = 0.97\n'
    )
    c = load_config(str(p), env={})
    assert c.listen == "127.0.0.1:9000"
    assert c.sample_rate == 0.5
    assert c.routes["anthropic:claude-opus-5-5:abc"] == RouteConfig(enabled=False, equivalence_bar=0.97)


def test_env_overrides_toml(tmp_path):
    p = tmp_path / "optimizer.toml"
    p.write_text('db_path = "/from/toml.db"\n')
    c = load_config(str(p), env={"OPTIMIZER_DB_PATH": "/from/env.db", "OPTIMIZER_ENABLED": "0",
                                 "OPTIMIZER_RETENTION_DAYS": "3"})
    assert c.db_path == "/from/env.db"
    assert c.enabled is False
    assert c.retention_days == 3


def test_unknown_toml_key_is_ignored(tmp_path):
    p = tmp_path / "optimizer.toml"
    p.write_text('not_a_field = 1\n')
    assert isinstance(load_config(str(p), env={}), Config)
