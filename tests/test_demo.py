import io
import json
import re
import time

import httpx
import pytest

from pith import db
from pith.__main__ import main
from pith.demo import mock_upstream, run_demo
from pith.rewrite import RouteState, apply_profile

SUMMARY = re.compile(r"^pinned P[234] \(synthetic\): [-+]\d+% output tokens at equivalence \d\.\d\d "
                     r"\(noise floor \d\.\d\d\); no request data left this machine$", re.MULTILINE)


def _row(path, sql):
    conn = db.connect(path)
    try:
        return tuple(conn.execute(sql).fetchone())
    finally:
        conn.close()


def test_run_demo_pins_a_profile_keylessly(tmp_path):
    path = str(tmp_path / "demo.db")
    out = io.StringIO()
    t0 = time.monotonic()
    assert run_demo(out=out, db_path=path) == 0
    assert time.monotonic() - t0 < 20
    text = out.getvalue()
    assert SUMMARY.search(text), text
    assert "recorded 40 requests, route demo observing" in text
    assert text.splitlines()[0].startswith("pith demo (synthetic):") and "no request data leaves this machine" in text
    assert text.isascii()
    pinned, status = _row(path, "SELECT pinned_profile, status FROM routes WHERE key='demo'")
    assert pinned != "P0" and status == "pinned"


def test_rerun_on_same_db_refuses_without_writing(tmp_path):
    path = str(tmp_path / "demo.db")
    assert run_demo(out=io.StringIO(), db_path=path) == 0
    before = _row(path, "SELECT COUNT(*) FROM requests")
    out = io.StringIO()
    assert run_demo(out=out, db_path=path) == 2
    assert out.getvalue().strip() == "database already has a demo route; use a fresh file"
    assert _row(path, "SELECT COUNT(*) FROM requests") == before


def test_db_with_other_routes_refuses_without_writing(tmp_path):
    path = str(tmp_path / "real.db")
    conn = db.connect(path)
    db.upsert_route(conn, "support", "anthropic", "claude-haiku-4-5", "h")
    conn.close()
    out = io.StringIO()
    assert run_demo(out=out, db_path=path) == 2
    assert "use a fresh file" in out.getvalue()
    assert _row(path, "SELECT COUNT(*) FROM routes") == (1,)
    assert _row(path, "SELECT COUNT(*) FROM requests") == (0,)


def test_same_seed_same_transcript():
    a, b = io.StringIO(), io.StringIO()
    assert run_demo(out=a, items=12) == run_demo(out=b, items=12) == 0
    assert a.getvalue() == b.getvalue()


@pytest.mark.parametrize("form", ["system", "user_text"])
def test_mock_detects_both_shape_forms(form):
    handler = mock_upstream(0)
    body = {"model": "claude-haiku-4-5", "max_tokens": 400, "system": "s",
            "messages": [{"role": "user", "content": "Does the blue jacket come in XL? (ticket #1001)"}]}

    def out_tokens(b):
        req = httpx.Request("POST", "https://x/v1/messages", content=json.dumps(b))
        return json.loads(handler(req).content)["usage"]["output_tokens"]

    assert 120 <= out_tokens(body) <= 180
    shaped = apply_profile("anthropic", body, RouteState("P2", form, 75))
    assert shaped != body
    assert 30 <= out_tokens(shaped) <= 60


def test_demo_subcommand(capsys):
    assert main(["demo", "--items", "12"]) == 0
    assert SUMMARY.search(capsys.readouterr().out)


@pytest.mark.parametrize("flag", ["--items", "--trials"])
def test_demo_rejects_non_positive_counts(flag, capsys):
    with pytest.raises(SystemExit) as e:
        main(["demo", flag, "0"])
    assert e.value.code == 2
    assert "must be an integer >= 1" in capsys.readouterr().err
