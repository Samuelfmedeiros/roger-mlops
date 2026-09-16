#!/usr/bin/env python3
"""Minimal trainer wired to the Roger critic.

Shows the integration contract in ~40 lines: the critic is advisory, it must
never raise into the training loop, and it must never be able to wedge a step.

Run:  python examples/train_with_roger.py --steps 20 --interval 2
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roger import roger_client  # noqa: E402


def fake_step(step, total):
    """Stand-in for one optimizer step: returns the telemetry you would log."""
    lr = 5e-5 * 0.5 * (1 + math.cos(math.pi * step / total))
    return {"loss": round(10.5 - 3.5 * (step / total) + random.uniform(-0.02, 0.02), 4),
            "grad": round(random.uniform(80, 200), 1), "lr": lr,
            "vram_gb": round(9.6 + random.uniform(0, .2), 2),
            "tok_s": random.randint(410, 440)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--interval", type=float, default=0.2, help="simulated step time")
    ap.add_argument("--every", type=int, default=5, help="ask Roger every N steps")
    ap.add_argument("--log", default="train.log")
    a = ap.parse_args()

    for step in range(1, a.steps + 1):
        t = fake_step(step, a.steps)
        line = (f"Step {step}/{a.steps} | Loss: {t['loss']:.4f} | Grad: {t['grad']} "
                f"| LR: {t['lr']:.2e} | {t['tok_s']} tok/s | VRAM: {t['vram_gb']} GB")
        print(line)
        with open(a.log, "a") as f:          # append-only: the log is the evidence
            f.write(line + "\n")
        time.sleep(a.interval)

        if step % a.every:
            continue

        # Advisory, fail-open. The client returns `continue` on timeout, so a
        # dead daemon can never block the run. Catch anything anyway.
        try:
            v = roger_client.submit(step=step, loss=t["loss"], grad=t["grad"],
                                    lr=t["lr"], vram=t["vram_gb"], tok=t["tok_s"],
                                    question=f"checkpoint {step}", timeout=30)
        except Exception as e:
            print(f"  [roger] unavailable ({e}) -> continuing")
            continue

        action = v.get("action", "continue")
        print(f"  [roger] score={v.get('score')} action={action}"
              f"{' (fail-open)' if v.get('fail_open') else ''}")
        for finding in (v.get("findings") or []) + (v.get("hard") or []):
            print(f"           - {finding}")
        if action == "escalate":
            # Escalate means stop-and-page-a-human, not stop-and-die: checkpoint
            # first so the next launch is cheap, then exit for the watchdog.
            print("  [roger] ESCALATE -> saving checkpoint and exiting")
            return 3
    print("run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
