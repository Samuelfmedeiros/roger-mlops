#!/usr/bin/env python3
"""GPU throughput referee — the only probe that catches a desiccated GPU-PV channel.

A WSL guest can be fully healthy at the process level and still execute
matmuls at 2% of the card's rate: py-spy shows frames advancing,
nvidia-smi shows 100% utilisation at full clocks, and a binary
`torch.cuda.synchronize()` test passes in under a second. Utilisation is a
sample between mini-kernel bursts; latency is not throughput.

What actually separates the states is a timed matmul against a reference:

    healthy   ~4.7-4.9 TFLOPS (RTX 3060 FP32 in this rig)
    degraded   0.3-1.1 TFLOPS  -> cure with `wsl --shutdown`, re-bench

🔴 Bench under contention is INCONCLUSIVE: with a wedged trainer still
holding the channel, a concurrent bench measured 1.31 — a middle band that
proves nothing. Either kill the trainer first, or read the number as noise.

Run:  python3 cuda_tflops_bench.py [--out /tmp/bench.json] [--size 4096]
                                 [--iters 20] [--min-tflops 2.0]
Exit: 0 healthy / 1 degraded / 2 error — so it composes in shell chains.
"""
import argparse
import json
import sys
import time


def bench(size=4096, iters=20):
    import torch  # imported here so `--help` works without torch installed
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible")
    dev = "cuda"
    out = {"device": torch.cuda.get_device_name(0)}

    t0 = time.time()
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)
    torch.cuda.synchronize()
    out["alloc_s"] = round(time.time() - t0, 2)

    t1 = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t1
    out["matmul_s"] = round(dt, 3)
    out["tflops"] = round(2 * size ** 3 * iters / dt / 1e12, 1)
    out["vram_alloc_mib"] = torch.cuda.memory_allocated() >> 20
    del a, b, c
    torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="CUDA throughput referee")
    ap.add_argument("--out", default="", help="write JSON verdict here")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--min-tflops", type=float, default=2.0,
                    help="below this the channel is considered degraded")
    a = ap.parse_args(argv)
    try:
        r = bench(a.size, a.iters)
        r["verdict"] = "OK" if r["tflops"] >= a.min_tflops else "DEGRADED"
        rc = 0 if r["verdict"] == "OK" else 1
    except Exception as e:  # an unmeasurable channel must not read as healthy
        r = {"verdict": f"ERROR: {e}"}
        rc = 2
    line = json.dumps(r)
    print(line)
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(line)
        import os
        os.replace(tmp, a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
