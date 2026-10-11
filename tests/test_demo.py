import io
import re
import time

from pith import db
from pith.__main__ import main
from pith.demo import run_demo

SUMMARY = re.compile(r"^pinned P[234]: −\d+% output tokens at equivalence \d\.\d\d \(noise floor \d\.\d\d\); "
                     r"nothing left this machine$", re.MULTILINE)


def test_run_demo_pins_a_profile_keylessly(tmp_path):
    path = str(tmp_path / "demo.db")
    out = io.StringIO()
    t0 = time.monotonic()
    assert run_demo(out=out, db_path=path) == 0
    assert time.monotonic() - t0 < 20
    text = out.getvalue()
    assert SUMMARY.search(text), text
    assert "recorded 40 requests, route demo observing" in text
    route = db.get_route(db.connect(path), "demo")
    assert route["pinned_profile"] != "P0" and route["status"] == "pinned"


def test_demo_subcommand(capsys):
    assert main(["demo", "--items", "12"]) == 0
    assert SUMMARY.search(capsys.readouterr().out)
