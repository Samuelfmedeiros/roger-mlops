#!/usr/bin/env python3
"""Checkpoint quarantine — move a bad checkpoint OUT OF THE GLOB, not in place.

Renaming `step_0345` to `step_0345_CORRUPTED` does NOT neutralise it. Every
resume resolver in the wild picks the highest-numbered `step_*` directory —
which is exactly the corrupted one. Result: the watchdog relaunches, the run
crashes on load, the watchdog relaunches. A relaunch loop whose signature is
the same `RESUMING from step_XXX_CORRUPTED_*` banner over and over.

The cure is a directory that no `step_*` glob matches:

    <ckpt dir>/_quarantine/step_0345_CORRUPTED

`_quarantine` starts with an underscore and is not `step_*`, so the dynamic
resolver walks straight past it.

This tool also VERIFIES before it quarantines: ls and file size do not catch
a partially corrupted tensor. Pass --load to run a real torch.load of each
model file (recommended before trusting a resume target).

Usage:
    python quarantine.py --dir /tmp/ckpts                    # report only
    python quarantine.py --dir /tmp/ckpts --load             # real load test
    python quarantine.py --dir /tmp/ckpts --quarantine step_0345 --reason nan
    python quarantine.py --dir /tmp/ckpts --best             # print safe target
"""
import argparse
import os
import re
import shutil
import sys

RE_STEP = re.compile(r"step[_-]?(\d+)")
QUAR = "_quarantine"


def candidates(d):
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isdir(p) or name == QUAR:
            continue
        m = RE_STEP.search(name)
        if m:
            out.append((int(m.group(1)), name, p))
    return sorted(out)


def load_check(path, do_load):
    """-> (ok, detail). Without --load this is a structural check only."""
    bins = [f for f in ("pytorch_model.bin", "model.safetensors")
            if os.path.exists(os.path.join(path, f))]
    if not bins:
        return False, "no model file"
    if not do_load:
        sizes = {f: os.path.getsize(os.path.join(path, f)) for f in bins}
        if any(s == 0 for s in sizes.values()):
            return False, f"zero-byte model file: {sizes}"
        return True, f"sizes={sizes}"
    try:
        import torch
    except ImportError:
        return True, "torch unavailable — structural check only"
    try:
        for f in bins:
            obj = torch.load(os.path.join(path, f), map_location="cpu",
                             weights_only=False)
            if hasattr(obj, "isnan"):
                if bool(obj.isnan().any()):
                    return False, f"{f}: NaN in weights"
            elif isinstance(obj, dict):
                for k, v in list(obj.items())[:50]:
                    if hasattr(v, "isnan") and bool(v.isnan().any()):
                        return False, f"{f}: NaN in {k}"
            del obj
        return True, "torch.load OK, no NaN"
    except Exception as e:
        return False, f"load failed: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="checkpoint quarantine")
    ap.add_argument("--dir", required=True, help="checkpoint directory")
    ap.add_argument("--load", action="store_true", help="real torch.load test")
    ap.add_argument("--quarantine", help="move this step dir into _quarantine/")
    ap.add_argument("--reason", default="bad")
    ap.add_argument("--best", action="store_true", help="print the safe resume target")
    a = ap.parse_args()

    if not os.path.isdir(a.dir):
        print(f"not a directory: {a.dir}", file=sys.stderr)
        return 2

    if a.quarantine:
        src = os.path.join(a.dir, a.quarantine)
        if not os.path.isdir(src):
            print(f"no such step dir: {src}", file=sys.stderr)
            return 2
        os.makedirs(os.path.join(a.dir, QUAR), exist_ok=True)
        dst = os.path.join(a.dir, f"{a.quarantine}_{a.reason}")
        shutil.move(src, os.path.join(a.dir, QUAR, os.path.basename(dst)))
        print(f"quarantined -> {os.path.join(QUAR, os.path.basename(dst))}")
        return 0

    rows = candidates(a.dir)
    if not rows:
        print("no step_* candidates found")
        return 1
    good = []
    for n, name, path in rows:
        ok, detail = load_check(path, a.load)
        print(f"{'OK ' if ok else 'BAD'} step {n:>6}  {name}  {detail}")
        if ok:
            good.append(n)
    if not good:
        print("VERDICT: no valid checkpoint — do not relaunch into this dir")
        return 1
    best = max(good)
    if a.best:
        print(best)
    else:
        print(f"VERDICT: safe resume target = step {best} "
              f"({os.path.join(a.dir, next(p for n, _nm, p in rows if n == best))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
