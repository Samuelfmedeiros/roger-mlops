#!/usr/bin/env python3
"""Roger critic unit suite — simulated observations against the REAL critic.

Two compatible shapes:
- script:   python tests/test_critic.py   -> 9 labeled PASS/FAIL lines
            + "=== n/9 PASS ===", exit 1 on any FAIL (the report Roger's
            gate parses).
- discover: python -m unittest discover -s tests  -> same cases as unittest
            methods (test_roger_*), failures surfaced as unittest failures.

Cases cover: healthy, LR coma, gradient explosion, rising loss, dead
process, wedged log, phantom self-report (ghost), truncated checkpoint,
and malformed request fail-open. No GPU, no network, no training process.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROGER = os.path.join(os.path.dirname(HERE), "roger", "critic.py")

HEALTHY = {"steps": [[40, 10.48, 120, 4e-5, 28, 4.12], [45, 10.46, 100, 4.5e-5, 28, 4.12],
                     [50, 10.4375, 202, 5e-5, 29, 4.12], [55, 10.3984, 96, 5e-5, 28, 4.12],
                     [60, 10.4141, 137, 5e-5, 30, 4.12]],
           "ad_grad": 2.4e-2, "log_age": 60, "alive": True, "ckpt_ok": True, "nan": False}


def _mutate(**kw):
    d = json.loads(json.dumps(HEALTHY))
    d.update(kw)
    return d


def ask_critic(req: dict) -> dict:
    """Run the real critic binary against a request file, return its JSON verdict."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        p = f.name
    try:
        r = subprocess.run([sys.executable, ROGER, "--request", p],
                           capture_output=True, timeout=60)
    finally:
        os.unlink(p)
    out = (r.stdout or b"").decode("utf-8", "replace")
    return json.loads(out)  # raises on non-JSON — that IS a failure


class TestRogerCritic(unittest.TestCase):
    """discover-compatible: one method per scenario."""

    def _check(self, name, sim, expect_action):
        v = ask_critic({"id": f"test_{name}", "sim": sim})
        self.assertEqual(v.get("action"), expect_action,
                         f"{name}: got {v.get('action')} hard={v.get('hard')} "
                         f"findings={v.get('findings')}")
        return v

    def test_roger_healthy_continue(self):
        self._check("healthy_continue", HEALTHY, "continue")

    def test_roger_lr_coma_escalate(self):
        self._check("lr_coma_escalate",
                    _mutate(steps=[[55, 10.8125, 1.5, 1e-6, 40, 4.12],
                                   [60, 10.8125, 1.28, 1e-6, 41, 4.12]]),
                    "escalate")

    def test_roger_grad_explosion_escalate(self):
        self._check("grad_explosion_escalate",
                    _mutate(steps=[[45, 10.45, 600, 4.5e-5, 28, 4.12],
                                   [50, 10.46, 1500, 5e-5, 29, 4.12],
                                   [55, 10.44, 1200, 5e-5, 28, 4.12]]),
                    "escalate")

    def test_roger_loss_rising_investigate(self):
        self._check("loss_rising_investigate",
                    _mutate(steps=[[20, 10.30, 90, 2e-5, 29, 4.12],
                                   [25, 10.32, 110, 2.5e-5, 28, 4.12],
                                   [30, 10.36, 95, 3e-5, 29, 4.12],
                                   [35, 10.42, 105, 3.5e-5, 28, 4.12],
                                   [40, 10.50, 130, 4e-5, 29, 4.12]]),
                    "investigate")

    def test_roger_dead_process_escalate(self):
        self._check("dead_process_escalate", _mutate(alive=False), "escalate")

    def test_roger_wedge_escalate(self):
        self._check("wedge_escalate", _mutate(log_age=2400), "escalate")

    def test_roger_ghost_report_blocked(self):
        # Phantom self-report: request claims step 999, log only reaches 60.
        v = ask_critic({"id": "test_ghost", "step": 999, "sim": HEALTHY})
        self.assertEqual(v.get("action"), "escalate")
        self.assertTrue(any("phantom" in h for h in v.get("hard", [])),
                        f"no phantom finding: {v.get('hard')}")

    def test_roger_truncated_ckpt_investigate(self):
        # corrupted checkpoint costs 25 points but must NOT block the run
        self._check("truncated_ckpt_investigate", _mutate(ckpt_ok=False),
                    "investigate")

    def test_roger_malformed_request_fail_open(self):
        # A half-written request file must fail OPEN as continue, never page
        # someone. Exercises the daemon's handle_file path directly.
        sys.path.insert(0, os.path.dirname(ROGER))
        os.environ.setdefault("ROGER_HOME", tempfile.mkdtemp(prefix="roger_home_"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("critic_probe", ROGER)
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)
        hdir = tempfile.mkdtemp(prefix="roger_spool_")
        bad = os.path.join(hdir, "request_malformed.json")
        with open(bad, "w") as f:
            f.write("{not json")
        rt.LOG = os.path.join(hdir, "absent.log")   # do not escalate on missing log
        rt.PERSIST = os.path.join(hdir, "no_ckpt")
        rt.REQ, rt.RESP, rt.PROC = hdir, hdir, hdir
        rt.LEDGER = os.path.join(hdir, "ledger.jsonl")
        rt.handle_file(bad)
        v = json.load(open(os.path.join(hdir, "response_malformed.json")))
        self.assertEqual(v.get("action"), "continue")
        self.assertIn("fail-open", v.get("note") or "")


# ── script mode: labeled output the gate parses ─────────────────────────

def _run_script() -> int:
    results = []

    def case(name, sim, expect):
        try:
            v = ask_critic({"id": f"test_{name}", "sim": sim})
        except Exception as e:
            results.append((name, "FAIL", f"non-JSON stdout: {e}"))
            return
        ok = v.get("action") == expect
        results.append((name, "PASS" if ok else "FAIL",
                        f"action={v.get('action')} score={v.get('score')} "
                        f"hard={v.get('hard')} findings={v.get('findings')}"))

    case("healthy_continue", HEALTHY, "continue")
    case("lr_coma_escalate", _mutate(
        steps=[[55, 10.8125, 1.5, 1e-6, 40, 4.12], [60, 10.8125, 1.28, 1e-6, 41, 4.12]]),
        "escalate")
    case("grad_explosion_escalate", _mutate(
        steps=[[45, 10.45, 600, 4.5e-5, 28, 4.12], [50, 10.46, 1500, 5e-5, 29, 4.12],
               [55, 10.44, 1200, 5e-5, 28, 4.12]]), "escalate")
    case("loss_rising_investigate", _mutate(
        steps=[[20, 10.30, 90, 2e-5, 29, 4.12], [25, 10.32, 110, 2.5e-5, 28, 4.12],
               [30, 10.36, 95, 3e-5, 29, 4.12], [35, 10.42, 105, 3.5e-5, 28, 4.12],
               [40, 10.50, 130, 4e-5, 29, 4.12]]), "investigate")
    case("dead_process_escalate", _mutate(alive=False), "escalate")
    case("wedge_escalate", _mutate(log_age=2400), "escalate")

    try:
        v = ask_critic({"id": "test_ghost", "step": 999, "sim": HEALTHY})
        ok = v.get("action") == "escalate" and any("phantom" in h for h in v.get("hard", []))
        results.append(("ghost_report_blocked", "PASS" if ok else "FAIL",
                        f"action={v.get('action')} hard={v.get('hard')}"))
    except Exception as e:
        results.append(("ghost_report_blocked", "FAIL", str(e)))

    case("truncated_ckpt_investigate", _mutate(ckpt_ok=False), "investigate")

    # malformed fail-open via unittest method (same body, no duplication drift)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRogerCritic)
    names = {"test_roger_malformed_request_fail_open": "malformed_request_fail_open"}
    for t in suite:
        if t._testMethodName in names:
            res = unittest.TestResult()
            t.run(res)
            ok = res.wasSuccessful()
            results.append((names[t._testMethodName], "PASS" if ok else "FAIL",
                            "action=continue note=fail-open" if ok
                            else str(res.failures or res.errors)[:200]))

    fails = [r for r in results if r[1] == "FAIL"]
    for n, s, d in results:
        print(f"{s}  {n}: {d}")
    print(f"=== {len(results) - len(fails)}/{len(results)} PASS ===")
    return 1 if fails else 0


if __name__ == "__main__":
    if "-m" in " ".join(sys.argv) or "--unittest" in sys.argv or any(
            a.startswith("TestRoger") for a in sys.argv[1:]):
        unittest.main()
    sys.exit(_run_script())
