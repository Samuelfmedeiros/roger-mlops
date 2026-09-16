#!/usr/bin/env python3
"""GPU-PV channel watchdog — the cure for the failure that looks like health.

Runs on the WINDOWS host (the side that owns the VM), watches a trainer
inside WSL, and can perform the only cure that works on a desiccated
GPU-PV channel: `wsl --shutdown`, re-bench, relaunch.

Why a separate watchdog from night_watch.py:

    night_watch catches "process dead" and "process wedged". This one
    catches the third state, the one that fools every other probe: the
    process is ALIVE, py-spy shows frames advancing, nvidia-smi reports
    97-100% utilisation at full clocks — and the GPU-PV channel is running
    at ~2% of its throughput. A 12-minute step takes 2h15. A binary CUDA
    test (`torch.cuda.synchronize() < 1s`) PASSES in this state.

    The only thing that detects it is throughput. So this watchdog is a
    throughput referee: log staleness raises the question, a TFLOPS bench
    answers it, and `wsl --shutdown` is the medicine.

Evidence chain from the incident that produced these thresholds:
    * dmesg signature precedes degradation:
      `dxg dxgkio_escape: Ioctl failed: -22`
    * degraded channel bench: 0.3 TFLOPS (healthy: 4.7) — 15x
    * checkpoint load dropped from ~20min to ~90s after the cure
    * bench under contention is INCONCLUSIVE (wedged trainer hogging the
      channel gave 1.31 = a middle band that proves nothing). Kill the
      trainer before benching, or treat the number as noise.

Host-side decoding pitfall (why `out_text` exists): wsl.exe on a pt-BR (or
any localized) Windows does not always emit WSAETIMEDOUT — it can return a
*translated* message, sometimes with rc=0. Never classify on error text;
classify on the sentinel you appended.

Run:  python -m roger.gpu_cure            (foreground loop)
Env:  ROGER_TRAIN_LOG, ROGER_PROC, ROGER_BENCH, ROGER_BENCH_OUT, ROGER_LAUNCHER,
      ROGER_STALE_S, ROGER_CADENCE_S, ROGER_BENCH_MIN, ROGER_INTERVAL_S
"""
import json
import os
import subprocess
import sys
import time

try:
    from . import config as C
except ImportError:  # pragma: no cover - executed as a plain script
    import config as C

# --- tuning knobs (defaults from measured cadence, not guesswork) ---------
STALE_S = int(C.cfg("STALE_S", "3600"))        # 1h without a step = deep stall
CADENCE_S = int(C.cfg("CADENCE_S", "2700"))    # 45min = slow cadence -> bench arbitrates
BENCH_MIN = float(C.cfg("BENCH_MIN", "2.0"))   # healthy 4.7-4.9; degraded 0.3-1.1
INTERVAL = int(C.cfg("INTERVAL_S", "1200"))    # 20min cycle
POWER_IDLE_W = float(C.cfg("POWER_IDLE_W", "15"))
SETTLE_S = int(C.cfg("SETTLE_S", "300"))       # let a fresh launch stabilise

# Host-visible path of the training log (drvfs mounts are readable from the
# host even when the wsl.exe command channel is wedged — that asymmetry is
# the whole reason this watchdog lives on the host).
TRAIN_LOG = C.cfg("TRAIN_LOG")
PROC = C.cfg("PROC", "run_train")
BENCH = C.cfg("BENCH")                       # guest-readable path to bench script
BENCH_OUT = C.cfg("BENCH_OUT") or C.tmp_file("roger_cudabench.json")
LAUNCHER = C.cfg("LAUNCHER")
WLOG = C.tmp_file("roger_gpu_cure.log")
ILOG = C.tmp_file("roger_gpu_incidents.log")


def log(msg, to_incident=False):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(WLOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if to_incident:
            with open(ILOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass


def out_text(r):
    """Decode wsl.exe output. It is UTF-16LE when the service talks, UTF-8
    when bash does, and NUL-strewn when it is half-dead."""
    b = r.stdout or b""
    if b[:2] == b"\xff\xfe":
        s = b.decode("utf-16-le", errors="ignore")
    else:
        s = b.decode("utf-8", errors="ignore")
    return s.replace("\x00", "")


def wsl(cmd, timeout=90):
    try:
        r = subprocess.run(["wsl.exe", "-e", "bash", "-c", cmd],
                           capture_output=True, timeout=timeout)
        return r
    except Exception as e:
        log(f"wsl probe raised {type(e).__name__}: {e}")
        return None


def training_alive():
    """Sentinel-based liveness. Bracket pattern so the probe never matches
    its own command line (a probe that finds itself reports a ghost)."""
    r = wsl(f"pgrep -f '[{PROC[:1]}]{PROC[1:]}' >/dev/null 2>&1 && echo VIVO || echo MORTO")
    if r is None:
        return None
    t = out_text(r)
    if "VIVO" in t:
        return True
    return False if "MORTO" in t else None


def log_age_s():
    """Prefer the HOST view of the log (drvfs/NTFS is readable when the
    command channel is not). Fall back to the guest's `stat`."""
    if TRAIN_LOG and os.path.exists(TRAIN_LOG):
        try:
            return max(0.0, time.time() - os.path.getmtime(TRAIN_LOG))
        except OSError:
            pass
    r = wsl(f"stat -c %Y {TRAIN_LOG} 2>/dev/null; date +%s")
    if r is None:
        return None
    parts = out_text(r).split()
    if len(parts) >= 2:
        try:
            return int(parts[1]) - int(parts[0])
        except ValueError:
            return None
    return None


def gpu_power_w():
    """Power drawn, read on the HOST. A 'busy' GPU at 8W is doing nothing."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=30)
        return float(out_text(r).strip().splitlines()[0].replace(",", "."))
    except Exception:
        return None


def run_bench():
    """Throughput referee. Returns TFLOPS or None (None = not measured)."""
    if not BENCH:
        log("ROGER_BENCH not set — cannot arbitrate with throughput")
        return None
    try:
        wsl(f"timeout 120 python3 {BENCH} --out {BENCH_OUT}", timeout=150)
        with open(BENCH_OUT, encoding="utf-8") as f:
            return json.load(f).get("tflops", 0.0)
    except Exception as e:
        log(f"bench errored: {e}")
        return None


def relaunch():
    """Detached + </dev/null: a foreground launch through a timeout kills the
    trainer when the timeout fires (the launcher execs python, so the timer
    hits training itself)."""
    wsl(f"setsid nohup bash {LAUNCHER} >/dev/null 2>&1 </dev/null &", timeout=30)
    log("relaunch dispatched (launcher's own flock arbitrates duplicates)")


def cure(reason):
    """Surgical host reset of the GPU-PV channel.

    🔴 `wsl --shutdown` KILLS EVERYTHING in the VM — including this
    trainer. That is acceptable only because the launcher is resume-safe.
    Never wire this to a run without checkpoints and a flock'd launcher.
    """
    log(f"=== CURE ({reason}): resetting GPU-PV channel ===", to_incident=True)
    wsl(f"pkill -f '[{PROC[:1]}]{PROC[1:]}'", timeout=60)
    time.sleep(3)
    try:
        subprocess.run(["wsl.exe", "--shutdown"], capture_output=True, timeout=120)
    except Exception as e:
        log(f"wsl --shutdown raised {e} — continuing to boot wait")
    for _ in range(6):
        time.sleep(15)
        r = wsl("echo BOOT_OK", timeout=60)
        if r is not None and "BOOT_OK" in out_text(r):
            break
    time.sleep(25)  # driver/NVML needs a moment after the distro is back
    t = run_bench()
    log(f"post-reset bench: {t} TFLOPS", to_incident=True)
    if t is not None and t < BENCH_MIN:
        log("channel STILL degraded after reset — not relaunching into a broken GPU; "
            "needs host-level attention (driver reboot / TDR / VRAM pressure)",
            to_incident=True)
        print(f"🔴 GPU-PV channel still degraded after `wsl --shutdown` "
              f"({t} TFLOPS < {BENCH_MIN}). Relaunch would die slowly again. "
              f"Check Windows-side VRAM pressure and the driver. Trail: {ILOG}")
        return False
    relaunch()
    return True


def tick():
    alive = training_alive()
    if alive is None:
        log("liveness NOT MEASURED (channel blind) — no decision this tick")
        return
    if not alive:
        log("trainer absent (a keepalive/relauncher owns this case)")
        return

    age = log_age_s()
    if age is None:
        log("log age unavailable — no decision this tick")
        return

    if age > STALE_S:
        t = run_bench()
        log(f"stale {int(age)//60}min -> bench {t} TFLOPS")
        if t is None or t < BENCH_MIN:
            cure("deep stall + bench degraded/unmeasured")
        else:
            log("stale but channel HEALTHY -> no cure (bench arbitrated; the stall is "
                "inside the process, see night_watch zombie logic)")
    elif age > CADENCE_S:
        gp = gpu_power_w()
        if gp is not None and gp < POWER_IDLE_W:
            log(f"slow cadence ({int(age)//60}min) with GPU at {gp}W = idle spin -> cure")
            cure("low power during cadence stall")
            return
        t = run_bench()
        log(f"slow cadence ({int(age)//60}min, GPU {gp}W) -> bench {t} TFLOPS")
        if t is not None and t < BENCH_MIN:
            cure("degraded channel under load")
        else:
            log("slow but channel healthy (legitimate cold start?) — waiting")
    else:
        log(f"ok: trainer alive, log age {int(age)//60}min")


def main():
    if not TRAIN_LOG:
        print("ROGER_TRAIN_LOG is not set — nothing to watch.")
        return 2
    log(f"=== gpu-cure watchdog up (stale {STALE_S//60}min, cadence {CADENCE_S//60}min, "
        f"bench<{BENCH_MIN} TFLOPS arbitrates, cycle {INTERVAL//60}min) ===")
    time.sleep(SETTLE_S)  # do not judge a run that just booted
    while True:
        try:
            tick()
        except Exception as e:
            log(f"cycle error: {e}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main() or 0)
