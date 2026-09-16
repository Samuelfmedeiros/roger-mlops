#!/usr/bin/env python3
"""IPC client for the Roger critic.

Drops a request into the spool, waits for the daemon's response, and —
critically — FAILS OPEN: a timeout prints a `continue` verdict instead of
blocking the caller. The training loop must never wait on its watchdog.

Usage from the training harness (Python):

    from roger.roger_client import ask
    verdict = ask(step=1200, loss=3.91, grad=140, lr=2.1e-5,
                  vram=9.8, tok=430)
    if verdict["action"] != "continue":
        ...  # log loudly; do not crash

CLI:
    python -m roger.roger_client --step 600 --loss 4.2 --grad 120 \\
        --lr 2.5e-5 --vram 9.9 --tok 430 --question "checkpoint 600"

Environment: ROGER_ROOT (spool root), plus ROGER_TIMEOUT for the wait.
"""
import argparse
import json
import os
import sys
import time

try:
    from . import config as C
except ImportError:  # pragma: no cover - executed as a plain script
    import config as C

BASE = os.path.expanduser(C.cfg("ROOT", os.path.expanduser("~/.roger/roger")))
if not BASE.endswith("roger"):
    BASE = os.path.join(BASE, "roger")
DEFAULT_TIMEOUT = int(C.cfg("TIMEOUT", "90"))


def submit(step, loss, grad=0, lr=0, vram=0, tok=0, question="",
           timeout=None, sim=None):
    """Write a request, poll for the response. Returns the verdict dict."""
    timeout = timeout or DEFAULT_TIMEOUT
    reqs = os.path.join(BASE, "requests")
    resps = os.path.join(BASE, "responses")
    os.makedirs(reqs, exist_ok=True)
    os.makedirs(resps, exist_ok=True)

    rid = str(int(time.time() * 1000))
    req = {"id": rid, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step,
           "loss": loss, "grad": grad, "lr": lr, "vram": vram, "tok": tok,
           "question": question}
    if sim:
        req["sim"] = sim
    rp = os.path.join(reqs, f"request_{rid}.json")
    with open(rp + ".tmp", "w") as f:          # atomic publish
        json.dump(req, f, ensure_ascii=False, indent=1)
    os.replace(rp + ".tmp", rp)

    deadline = time.time() + timeout
    resp_path = os.path.join(resps, f"response_{rid}.json")
    while time.time() < deadline:
        if os.path.exists(resp_path):
            try:
                with open(resp_path) as f:
                    return json.load(f)
            except Exception:
                pass  # daemon may be mid-write; keep polling
        time.sleep(2)
    return {"request_id": rid, "action": "continue", "fail_open": True,
            "note": f"timeout {timeout}s waiting for Roger response"}


# Backwards-friendly alias
ask = submit


def main(argv):
    ap = argparse.ArgumentParser(description="Roger IPC client")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--loss", type=float, required=True)
    ap.add_argument("--grad", type=float, default=0)
    ap.add_argument("--lr", type=float, default=0)
    ap.add_argument("--vram", type=float, default=0)
    ap.add_argument("--tok", type=int, default=0)
    ap.add_argument("--question", default="")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--sim", help="JSON simulation block (tests)")
    a = ap.parse_args(argv)
    v = submit(a.step, a.loss, a.grad, a.lr, a.vram, a.tok, a.question,
               a.timeout, json.loads(a.sim) if a.sim else None)
    print(json.dumps(v, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
