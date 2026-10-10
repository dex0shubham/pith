# Plan 6: Evidence and Release Implementation Plan

**Goal:** Make the benchmark honest (a sweep judges identical effective requests once and labels duplicates), make the standalone proxy's missing authentication loud, run the three gateway live tests on a schedule, and pin installs to a tagged release.

**Architecture:** `sweep.run_sweep` canonicalises each profile's rewritten bodies and, when a profile would send exactly what an earlier profile sent, records it as an alias row instead of replaying and judging it again. The README's live-results section says plainly that P2 and P4 were the same request on Haiku. `proxy.create_app` warns at startup when bound to a non-loopback interface. A new scheduled workflow installs pinned LiteLLM, Headroom and Node and runs the three live mock tests. Install lines point at the `v0.1.0` tag (created after merge).

**Tech Stack:** Python 3.12, pytest, GitHub Actions. No new dependency.

**Spec:** this plan is its own spec; it follows `docs/design/specs/2026-10-07-control-plane-design.md` for the sweep's contract (a profile that changes nothing is `skipped`, never billed) and extends it with: a profile whose effective requests equal an earlier profile's is an alias, never billed, never judged, never pinned.

## Global Constraints

- Python ≥ 3.12; code under `pith/`, tests under `tests/`; pytest only; the suite must stay warning-free under `.venv/bin/pytest -q -W error`. Run everything with `.venv/bin/pytest` / `.venv/bin/python`.
- Alias detection happens before anything is billed for that profile; the alias row is `{"skipped": True, "alias_of": "<profile>", "qualifies": False, "reason": "same request as <profile> on this model"}`; aliases never win `pin_rule`; `estimate_cost` is unchanged (it stays a conservative upper bound).
- The system-role fallback inside `replay_profile` still restarts the profile; the canonical key recorded for a profile is the bodies it finally replayed.
- Existing tests in `tests/test_sweep.py` pass with at most the `seed_route`/`_sweep_setup` helpers gaining a `model` parameter.
- Commit messages: plain conventional commits, no trailers, no co-author lines, no tool or model names anywhere in the repo.
- Branch `evidence-and-release`.

---

### Task 1: Sweep aliases identical effective requests; README live-results correction

**Files:**
- Modify: `pith/sweep.py` (`run_sweep`: `replay_profile` returns the bodies; alias check; post-`pin_rule` labelling), `README.md` ("## Live results"), `tests/test_sweep.py`

**Interfaces:**
- Produces: `replay_profile(profile, state) -> tuple[dict[int, list[Reply]], str | None, dict[int, dict]]` where the middle value is `None` (replayed), `"skipped"` (no-op for every item) or the name of the earlier profile whose bodies were identical; `run_sweep` table rows for aliases as in Global Constraints; `tests/test_sweep.py::seed_route(conn, key="k", n=50, text=True, bodies=True, status=None, model="claude-opus-5-5")` and `_sweep_setup(n=50, model="claude-opus-5-5")`.

- [ ] **Step 1: Write the failing test**

In `tests/test_sweep.py`, give `seed_route` a `model="claude-opus-5-5"` parameter used for both the `upsert_route` call and the stored bodies' `"model"`, and give `_sweep_setup` a `model="claude-opus-5-5"` parameter passed through to `seed_route`. Append:

```python
def test_run_sweep_judges_identical_requests_once():
    # claude-haiku-4-5 has no effort parameter, so P4's effective request is byte-identical to P2's.
    conn, cfg, route = _sweep_setup(model="claude-haiku-4-5")
    script = Script()
    out = run_sweep(conn, cfg, route, httpx.Client(transport=httpx.MockTransport(script)), {"anthropic": "k"},
                    trials=2, sample_n=10, rng=random.Random(0), now=time.time())
    table = out.table
    assert table["P1"]["skipped"] is True and table["P1"]["reason"] == "skipped"
    assert table["P4"] == {"skipped": True, "alias_of": "P2", "qualifies": False, "reason": "same request as P2 on this model"}
    assert out.winner == "P2" and db.get_route(conn, "k")["pinned_profile"] == "P2"
    assert conn.execute("SELECT COUNT(*) FROM judgments WHERE profile='P4'").fetchone()[0] == 0
    shaped_calls = [b for b in script.calls if not (b.get("system") and "You compare two answers" in b["system"])
                    and any(isinstance(m.get("content"), list) for m in b["messages"])]
    assert len(shaped_calls) == 2 * 10 * 2  # P2 and P3 only (P3 differs by its exemplar), trials × items each
    stored = json.loads(db.sweeps_for_route(conn, "k")[0]["result_json"])["table"]
    assert stored["P4"]["alias_of"] == "P2"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_sweep.py::test_run_sweep_judges_identical_requests_once -q`
Expected: FAIL (`table["P4"]` is a full replayed row, and P4 judgments exist).

- [ ] **Step 3: Implement**

In `pith/sweep.py`, `run_sweep`:

1. Add `seen: dict[str, str] = {}` next to `spent`.
2. In `replay_profile`, after the existing no-op check inside `if state is not None:`, add:

```python
            canon = json.dumps(bodies, sort_keys=True, default=str)
            if canon in seen:
                return {}, seen[canon], bodies  # identical effective request: judged once, under the first profile
```

   Change the no-op return to `return {}, "skipped", bodies`, the fallback-restart return to `return replay_profile(profile, state)` (unchanged), and the final return to `return replies, None, bodies`. Update the docstring: "Returns {item_id: [reply per trial]}, None | "skipped" | the profile this one duplicates, and the bodies used."
3. `p0, _, _ = replay_profile("P0", None)`.
4. In the profiles loop:

```python
            replies, dup, bodies = replay_profile(profile, state)
            if dup == "skipped":
                summarize(profile, {}, {}, skipped=True)
                continue
            if dup:
                table[profile] = {"skipped": True, "alias_of": dup}
                continue
            seen[json.dumps(bodies, sort_keys=True, default=str)] = profile
```

5. After `winner = pin_rule(table, bar, floor)` (both branches of the `if trials >= 2 and floor is None` block), add:

```python
    for row in table.values():
        if row.get("alias_of"):
            row["qualifies"], row["reason"] = False, f"same request as {row['alias_of']} on this model"
```

`pin_rule` already treats alias rows as skipped (it reads `row.get("skipped")` before any other key) and never selects them; `format_table` and the amortisation loop read only `.get`/`"skipped"` on these rows.

- [ ] **Step 4: Correct the README live-results section**

Replace the P4 row of the table with:

```markdown
| P4 | effort down + shape | 0.933 ± 0.046 | 69 | $0.000562 | pinned, but see below |
```

and replace the paragraph that begins "The pinned profile matches the unconstrained model's agreement with itself exactly" with:

```markdown
One correction to how this table was first read: `claude-haiku-4-5` has no `effort` parameter, so P4's request was
byte-identical to P2's, and the sweep judged the same request twice. The gap between their rates (0.833 ± 0.069 versus
0.933 ± 0.046) is judge noise on 30 items, not evidence that combining effort and shape helps. What was pinned is the
P2 instruction, at 56% fewer output tokens and a judged equivalence that matched the unconstrained model's agreement
with itself in this run. The sweep now judges identical effective requests once and reports the duplicate as an alias
(`same request as P2 on this model`). Treat the number as one 30-item benchmark; a larger run with a holdout is the
next step, and `recheck` re-judges live traffic after a pin and reverts on drift.
```

Keep the cache-safety and "what the earlier failing runs showed" paragraphs.

- [ ] **Step 5: Run the whole suite and commit**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass.

```bash
git add pith/sweep.py tests/test_sweep.py README.md
git commit -m "fix: sweep judges identical effective requests once; README corrects the P2/P4 reading"
```

---

### Task 2: Proxy startup warning, scheduled integration workflow, pinned install lines

**Files:**
- Modify: `pith/proxy.py` (`create_app`), `tests/test_proxy.py`, `README.md` (install lines; "## Tests")
- Create: `.github/workflows/integrations.yml`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy.py` (add `import logging` to the imports):

```python
def test_non_loopback_bind_warns_about_missing_authentication(caplog):
    with caplog.at_level(logging.WARNING, logger="pith"):
        make(Config(sample_rate=0))  # default listen 0.0.0.0:8787
        assert "no authentication" in caplog.text
        caplog.clear()
        make(Config(sample_rate=0, listen="127.0.0.1:8787"))
        make(Config(sample_rate=0, listen="localhost:9000"))
        assert "no authentication" not in caplog.text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_proxy.py::test_non_loopback_bind_warns_about_missing_authentication -q`
Expected: FAIL (`"no authentication" in caplog.text` is false).

- [ ] **Step 3: Implement**

In `pith/proxy.py`, `create_app`, next to the existing Portkey warning, add:

```python
    if not config.listen.startswith(("127.0.0.1:", "localhost:")):
        log.warning("proxy has no authentication; listen=%s — bind it to a private interface or loopback", config.listen)
```

Create `.github/workflows/integrations.yml`:

```yaml
name: integrations

on:
  schedule:
    - cron: "0 6 * * 1"
  workflow_dispatch:

jobs:
  live:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      OPTIMIZER_LITELLM_LIVE: "1"
      OPTIMIZER_HEADROOM_LIVE: "1"
      OPTIMIZER_PORTKEY_LIVE: "1"
      HEADROOM_BEACON: "off"
      DO_NOT_TRACK: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
      - run: python -m pip install -e '.[dev]' 'litellm[proxy]==1.104.2' headroom-ai==0.40.0 uvicorn
      - run: python -m pip install -e .   # re-register pith's entry points after Headroom's install
      - run: python -m pytest -q -rs tests/live/test_litellm_mock.py tests/live/test_headroom_mock.py tests/live/test_portkey_mock.py
```

In `README.md`, change both `pip install git+https://github.com/dex0shubham/pith` lines to `pip install git+https://github.com/dex0shubham/pith@v0.1.0`, and in "## Run" add before the `python3 -m venv` line a sentence: "Releases are git tags; install a pinned one with `pip install git+https://github.com/dex0shubham/pith@v0.1.0`." In "## Tests", after the Portkey live-test paragraph, add:

```markdown
The three gateway live tests also run weekly in CI (the `integrations` workflow, also runnable by hand) against pinned
LiteLLM 1.104.2, Headroom 0.40.0 and Portkey gateway 1.15.2, so an upstream change cannot break them silently.
```

- [ ] **Step 4: Run the suite and commit**

Run: `.venv/bin/pytest -q -W error`
Expected: all pass. Validate the workflow file parses: `.venv/bin/python -c "import tomllib" >/dev/null; python3 -c "import yaml" 2>/dev/null || true` (if PyYAML is absent, skip; GitHub validates on push).

```bash
git add pith/proxy.py tests/test_proxy.py .github/workflows/integrations.yml README.md
git commit -m "feat: warn on unauthenticated non-loopback bind; weekly integration CI; pinned install lines"
```

---

## Self-review

- Alias contract (never billed, never judged, never pinned) → Task 1 code and test; the README correction → Task 1 Step 4.
- Security boundary → Task 2 warning and test. Continuous integration tests → Task 2 workflow. Versioned install → Task 2 README; the `v0.1.0` tag is created after merge.
- Names consistent: `replay_profile` three-tuple, `seen`, `alias_of`, `integrations` workflow, `v0.1.0`.
