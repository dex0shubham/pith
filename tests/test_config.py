from pith.config import Config, RouteConfig, load_config


def test_defaults_match_spec():
    c = load_config(None, env={})
    assert c.listen == "0.0.0.0:8787"
    assert c.anthropic_upstream == "https://api.anthropic.com"
    assert c.openai_upstream == "https://api.openai.com"
    assert c.db_path == "./pith.db"
    assert c.sample_rate == 0.05
    assert c.prices == {}
    assert not hasattr(c, "shadow_rate")
    assert c.retention_days == 14
    assert c.sweep_budget_usd_month == 0
    assert c.equivalence_bar == 0.95
    assert c.judge_model == "claude-sonnet-5-5"
    assert c.judge_provider == "anthropic"
    assert c.enabled is True
    assert c.routes == {}
    assert c.portkey_upstream == "http://localhost:8787" and c.webhook_token == "" and c.portkey_headers == {}


def test_toml_and_route_overrides(tmp_path):
    p = tmp_path / "pith.toml"
    p.write_text(
        'listen = "127.0.0.1:9000"\nsample_rate = 0.5\n'
        '[routes."anthropic:claude-opus-5-5:abc"]\nenabled = false\nequivalence_bar = 0.97\n'
    )
    c = load_config(str(p), env={})
    assert c.listen == "127.0.0.1:9000"
    assert c.sample_rate == 0.5
    assert c.routes["anthropic:claude-opus-5-5:abc"] == RouteConfig(enabled=False, equivalence_bar=0.97)


def test_env_overrides_toml(tmp_path):
    p = tmp_path / "pith.toml"
    p.write_text('db_path = "/from/toml.db"\n')
    c = load_config(str(p), env={"OPTIMIZER_DB_PATH": "/from/env.db", "OPTIMIZER_ENABLED": "0",
                                 "OPTIMIZER_RETENTION_DAYS": "3"})
    assert c.db_path == "/from/env.db"
    assert c.enabled is False
    assert c.retention_days == 3


def test_unknown_toml_key_is_ignored(tmp_path):
    p = tmp_path / "pith.toml"
    p.write_text('not_a_field = 1\n')
    assert isinstance(load_config(str(p), env={}), Config)


def test_prices_table_parsed(tmp_path):
    p = tmp_path / "pith.toml"
    p.write_text('[prices."gpt-5"]\ninput = 1.25\noutput = 10\n[prices."my-model"]\ninput = 0.5\noutput = 2.5\n')
    c = load_config(str(p), env={})
    assert c.prices == {"gpt-5": (1.25, 10.0), "my-model": (0.5, 2.5)}


def test_litellm_upstream_default_and_env_override():
    assert Config().litellm_upstream == "http://localhost:4000"
    assert load_config(None, env={"OPTIMIZER_LITELLM_UPSTREAM": "http://l:4000"}).litellm_upstream == "http://l:4000"


def test_portkey_headers_table_token_and_env_override(tmp_path):
    p = tmp_path / "pith.toml"
    p.write_text('webhook_token = "t0k"\n[portkey_headers]\n"x-portkey-provider" = "openai"\n"x-portkey-config" = "pc-1"\n"X-Portkey-Metadata" = "{}"\n')
    c = load_config(str(p), env={"OPTIMIZER_PORTKEY_UPSTREAM": "http://gw:8787"})
    assert c.webhook_token == "t0k" and c.portkey_upstream == "http://gw:8787"
    assert c.portkey_headers == {"x-portkey-provider": "openai", "x-portkey-config": "pc-1", "x-portkey-metadata": "{}"}
    assert load_config(None, env={"OPTIMIZER_WEBHOOK_TOKEN": "env"}).webhook_token == "env"
