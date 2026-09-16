#!/usr/bin/env python3
"""Roger Tatu unit suite — simulated observations against the REAL critic.

Each case feeds a synthetic `sim` block (the same shape a live critic would
extract from the log) and asserts the ACTION the critic must take. This is
the test to run after any edit to roger_tatu.py: 8 scenarios, no GPU, no
network, no training process required.

Run:  python tests/test_roger_tatu.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROGER = os.path.join(os.path.dirname(HERE), "tatu", "roger_tatu.py")

results = []


def run_case(name, sim, expect_action):
    req = {"id": f"test_{name}", "sim": sim}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        p = f.name
    try:
        r = subprocess.run([sys.executable, ROGER, "--request", p],
                           capture_output=True, timeout=60)
    finally:
        os.unlink(p)
    out = (r.stdout or b"").decode("utf-8", "replace")
    try:
        v = json.loads(out)
    except Exception:
        results.append((name, "FAIL", f"non-JSON stdout: {out[:200]}"))
        return
    ok = v.get("action") == expect_action
    results.append((name, "PASS" if ok else "FAIL",
                    f"action={v.get('action')} score={v.get('score')} "
                    f"hard={v.get('hard')} findings={v.get('findings')}"))


HEALTHY = {"steps": [[40, 10.48, 120, 4e-5, 28, 4.12], [45, 10.46, 100, 4.5e-5, 28, 4.12],
                     [50, 10.4375, 202, 5e-5, 29, 4.12], [55, 10.3984, 96, 5e-5, 28, 4.12],
                     [60, 10.4141, 137, 5e-5, 30, 4.12]],
           "ad_grad": 2.4e-2, "log_age": 60, "alive": True, "ckpt_ok": True, "nan": False}

run_case("healthy_continue", HEALTHY, "continue")

coma = json.loads(json.dumps(HEALTHY))
coma["steps"] = [[55, 10.8125, 1.5, 1e-6, 40, 4.12], [60, 10.8125, 1.28, 1e-6, 41, 4.12]]
run_case("lr_coma_escalate", coma, "escalate")

boom = json.loads(json.dumps(HEALTHY))
boom["steps"] = [[45, 10.45, 600, 4.5e-5, 28, 4.12], [50, 10.46, 1500, 5e-5, 29, 4.12],
                 [55, 10.44, 1200, 5e-5, 28, 4.12]]
run_case("grad_explosion_escalate", boom, "escalate")

rising = json.loads(json.dumps(HEALTHY))
rising["steps"] = [[20, 10.30, 90, 2e-5, 29, 4.12], [25, 10.32, 110, 2.5e-5, 28, 4.12],
                   [30, 10.36, 95, 3e-5, 29, 4.12], [35, 10.42, 105, 3.5e-5, 28, 4.12],
                   [40, 10.50, 130, 4e-5, 29, 4.12]]
run_case("loss_rising_investigate", rising, "investigate")

dead = json.loads(json.dumps(HEALTHY))
dead["alive"] = False
run_case("dead_process_escalate", dead, "escalate")

wedge = json.loads(json.dumps(HEALTHY))
wedge["log_age"] = 2400
run_case("wedge_escalate", wedge, "escalate")

# Phantom self-report: request claims step 999, log only reaches 60.
ghost = {"id": "test_ghost", "step": 999, "sim": HEALTHY}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(ghost, f)
    gp = f.name
try:
    r = subprocess.run([sys.executable, ROGER, "--request", gp],
                       capture_output=True, timeout=60)
finally:
    os.unlink(gp)
v = json.loads((r.stdout or b"").decode("utf-8", "replace"))
ok = v.get("action") == "escalate" and any("phantom" in h for h in v.get("hard", []))
results.append(("ghost_report_blocked", "PASS" if ok else "FAIL",
                f"action={v.get('action')} hard={v.get('hard')}"))

# Corrupted checkpoint must cost 25 points but NOT block (investigate, not escalate)
badck = json.loads(json.dumps(HEALTHY))
badck["ckpt_ok"] = False
run_case("truncated_ckpt_investigate", badck, "investigate")

# A malformed request must fail OPEN as continue — evaluating an empty req
# would escalate on "no parsable steps" and a half-written file would page
# someone for nothing. Exercises the daemon's handle_file path directly.
sys.path.insert(0, os.path.dirname(ROGER))
os.environ.setdefault("TATU_HOME", tempfile.mkdtemp(prefix="roger_home_"))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("roger_tatu", ROGER)
_rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rt)
_hdir = tempfile.mkdtemp(prefix="roger_spool_")
_bad = os.path.join(_hdir, "request_malformed.json")
with open(_bad, "w") as f:
    f.write("{not json")
_rt.LOG = os.path.join(_hdir, "absent.log")   # prove we do not escalate on a missing log
_rt.PERSIST = os.path.join(_hdir, "no_ckpt")
_rt.REQ, _rt.RESP, _rt.PROC = _hdir, _hdir, _hdir
_rt.LEDGER = os.path.join(_hdir, "ledger.jsonl")
_rt.handle_file(_bad)
v = json.load(open(os.path.join(_hdir, "response_malformed.json")))
ok = v.get("action") == "continue" and "fail-open" in (v.get("note") or "")
results.append(("malformed_request_fail_open", "PASS" if ok else "FAIL",
                f"action={v.get('action')} note={v.get('note')}"))

fails = [r for r in results if r[1] == "FAIL"]
for n, s, d in results:
    print(f"{s}  {n}: {d}")
print(f"=== {len(results) - len(fails)}/{len(results)} PASS ===")
sys.exit(1 if fails else 0)
