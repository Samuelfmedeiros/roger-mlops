#!/usr/bin/env python3
"""Night watch — the watchdog that actually RESURRECTS a training run.

Why this exists: most watchdogs only alert. On a Windows host running the
trainer inside WSL, the failure modes that matter are silent and there is
nobody awake at 03:00. This script closes that gap. Live = silence.
Dead without an end banner = relaunch resume-safe. Wedged = kill the
process group and relaunch. And every decision requires a POSITIVE
measurement — an unreadable probe is never evidence of death.

The five guards, each earned by a real incident:

1. SENTINEL MEASUREMENT (`medida`) — a `wsl.exe` call that times out under
   load returns empty. Empty is not "dead". Every probe appends a sentinel
   and the answer is only trusted when the sentinel comes back.
2. THREE-STATE LOGIC — live / dead / NOT MEASURED. A not-measured tick
   takes no action at all.
3. ZOMBIE DETECTION — a spinning CPU does not prove life. A process stuck
   in a frozen Python stack burns CPU for hours with zero progress, so
   after THRESH_ZOMBIE_MIN of silence we hash the py-spy dump twice;
   identical = zombie = kill + relaunch. If the stack can't be measured at
   all, ZOMBIE_BLIND_MAX_MIN caps how long blindness buys silence.
4. STEP REGRESSION GUARD — a boot that wipes /tmp plus a launcher that
   falls back to scratch loses hundreds of steps and looks perfectly
   healthy. Compare the current step to the maximum ever seen.
5. CUDA CRASH-LOOP GUARD — a GPU link error kills the run, a resume-safe
   launcher brings it back from the same checkpoint, and the loop repeats
   forever. Count resumes per step: 3 on the same step means the host is
   broken, not the process.

Stdout is the alert channel (a cron `no_agent` job delivers it to chat);
silence means healthy. Local logging goes to a dedicated timestamped file.

Environment: TATU_LOG, TATU_PROC, TATU_END_BANNER, TATU_LAUNCHER,
TATU_LOCKFILE, TATU_PYSPY, TATU_TMP, plus the TATU_* tuning knobs below.
"""
import json
import os
import re
import subprocess
import sys
import time

try:
    from . import config as C
except ImportError:  # pragma: no cover - executed as a plain script
    import config as C

LOG = C.cfg("LOG")
PROC = C.cfg("PROC", "run_train")
END_BANNER = C.cfg("END_BANNER", r"CONCLU|DONE")
LAUNCHER = C.cfg("LAUNCHER")
LOCKFILE = C.cfg("LOCKFILE") or (os.path.dirname(LOG) + "/.train.lock")
PYPY = C.cfg("PYSPY", os.path.expanduser("~/.local/bin/py-spy"))

# Tunables (documented defaults chosen from real run cadence, not theory).
THRESH_WEDGE_MIN = int(C.cfg("WEDGE_MIN", "60"))     # log silence before wedge talk
THRESH_ZOMBIE_MIN = int(C.cfg("ZOMBIE_MIN", "90"))   # after this, CPU alone is not enough
ZOMBIE_BLIND_MAX_MIN = int(C.cfg("ZOMBIE_BLIND_MIN", "240"))
MAX_STEP_DROP = int(C.cfg("STEP_REGRESS_MAX", "60"))  # legit resume loses <=10
ALERT_GAP_MIN = int(C.cfg("ALERT_GAP_MIN", "60"))     # per-category anti-spam
CRASH_LOOP_RESUMES = int(C.cfg("CRASH_LOOP", "3"))
RELAUNCH_CONFIRM_S = int(C.cfg("RELAUNCH_CONFIRM_S", "45"))

WLOG = C.tmp_file("tatu_night_watch.log")
ILOG = C.tmp_file("tatu_train_incidents.log")
ALERT_STATE = C.tmp_file("tatu_alert_state.txt")
MAXSTEP_STATE = C.tmp_file("tatu_max_step.txt")
WATCH_STATE = C.tmp_file("tatu_watch_state.json")

SENTINEL = "__TATU_MEASURED__"
RE_STEP = re.compile(r"Step\s+(\d+)/(\d+)")


# ── probing ───────────────────────────────────────────────────────────────
def _run(cmd, timeout):
    r = subprocess.run(C.wsl_cmd(cmd), capture_output=True, timeout=timeout)
    return ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", "replace")


def wsl(cmd, timeout=60, tries=3, gap_sleep=8):
    """Raw best-effort probe. Empty string means the bridge is blind."""
    for i in range(tries):
        try:
            out = _run(cmd, timeout)
            if out:
                return out
        except Exception:
            pass
        if i < tries - 1:
            time.sleep(gap_sleep)
    return ""


def medida(cmd, timeout=90, tries=3, gap_sleep=8):
    """Measurement with a sentinel -> (text, measured: bool).

    🔴 Without the sentinel a pgrep that times out under load returns empty
    and empty gets read as death. This exact bug announced 'DEAD /
    RELAUNCHED' for a process that had been alive the whole time.
    """
    for i in range(tries):
        try:
            out = _run(f"{cmd}; echo {SENTINEL}", timeout)
            if SENTINEL in out:
                return out.split(SENTINEL)[0], True
        except Exception:
            pass
        if i < tries - 1:
            time.sleep(gap_sleep)
    return "", False


def bridge_alive():
    return SENTINEL in wsl(f"echo {SENTINEL}", timeout=45, tries=2)


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    try:
        with open(WLOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def incident(msg):
    """Dedicated, timestamped incident log — the paper trail for a fix."""
    try:
        with open(ILOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%F %T')} {msg}\n")
    except Exception:
        pass


# ── state probes ──────────────────────────────────────────────────────────
def alive():
    """-> (pid|None, measured: bool). measured=False is NOT death."""
    out, ok = medida(f"pgrep -f '[{PROC[:1]}]{PROC[1:]}' | head -1", timeout=90)
    if not ok:
        return None, False
    pids = [p.strip() for p in out.splitlines() if p.strip().isdigit()]
    return (pids[0] if pids else None), True


def log_age_min():
    out, ok = medida(f"stat -c %Y {LOG} 2>/dev/null; date +%s", timeout=60)
    if not ok:
        return None
    nums = [l for l in out.splitlines() if l.strip().isdigit()]
    if len(nums) < 2:
        return None
    return max(0.0, (int(nums[1]) - int(nums[0])) / 60.0)


def cpu_advancing(pid, gap=45):
    """Did utime+stime move in `gap` seconds? None = not measured."""
    out, ok = medida(
        f"awk '{{print $14+$15}}' /proc/{pid}/stat; sleep {gap}; "
        f"awk '{{print $14+$15}}' /proc/{pid}/stat", timeout=gap + 60)
    if not ok:
        return None
    nums = [l.strip() for l in out.splitlines() if l.strip().isdigit()]
    if len(nums) < 2:
        return None
    return int(nums[1]) > int(nums[0])


def stack_frozen(pid, samples=2, gap=12):
    """True = frozen stack (zombie) | False = moving | None = not measured.

    A zombie with a spinning CPU is the hardest failure to spot: state=R,
    utime climbing, IO zero, py-spy always on the same line. Only the hash
    of the MainThread section across two samples separates it from real work.
    """
    hashes = []
    for i in range(samples):
        out, ok = medida(
            f"sudo -n {PYPY} dump --pid {pid} 2>/dev/null | "
            f"sed -n '/MainThread/,/^Thread/p' | md5sum",
            timeout=90, tries=2, gap_sleep=6)
        if not ok:
            return None
        h = out.strip().split()[0] if out.strip() else ""
        if not h:
            return None
        hashes.append(h)
        if i < samples - 1:
            time.sleep(gap)
    return len(set(hashes)) == 1


def gpu_ok():
    """True = CUDA visible from the guest, False = NVML blocked, None = blind.

    When NVML is blocked, relaunching is useless — the process dies in
    seconds and the watchdog turns into an alarm machine. Cure is on the
    host (`wsl --shutdown`), which is a human decision while other tenants
    share the VM.
    """
    out, ok = medida("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1",
                     timeout=90)
    if not ok:
        return None
    return "NVIDIA" in out.upper()


def lock_holder_alive():
    """Is the flock owner still there? Guard against phantom relaunches."""
    if not LOCKFILE:
        return None
    out, ok = medida(f'p=$(cat {LOCKFILE} 2>/dev/null); '
                     f'[ -n "$p" ] && kill -0 $p 2>/dev/null && echo VIVO || echo NAO', timeout=60)
    if not ok:
        return None
    return "VIVO" in out


def current_step():
    out, ok = medida(f"grep -aoE 'Step [0-9]+/[0-9]+' {LOG} | tail -1", timeout=60)
    if not ok:
        return None
    m = RE_STEP.search(out or "")
    return int(m.group(1)) if m else None


def run_finished():
    """Positive evidence of legitimate completion (banner or final step)."""
    out, ok = medida(f"grep -aE '{END_BANNER}' {LOG} | tail -2", timeout=60)
    if ok and out.strip():
        return True, out.strip().replace("\n", " | ")[:200]
    return False, ""


# ── alert anti-spam (per category) ────────────────────────────────────────
def _alert_path(kind):
    return ALERT_STATE if kind == "gpu" else f"{ALERT_STATE}.{kind}"


def alert_due(kind="gpu"):
    try:
        last = float(open(_alert_path(kind), encoding="utf-8").read().strip())
    except Exception:
        last = 0.0
    return (time.time() - last) > ALERT_GAP_MIN * 60


def mark_alert(kind="gpu"):
    try:
        with open(_alert_path(kind), "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except Exception:
        pass


# ── guards ────────────────────────────────────────────────────────────────
def check_step_regression(step):
    """Alert when the step count FALLS far below the maximum ever observed."""
    if step is None:
        return
    try:
        prev = int(open(MAXSTEP_STATE, encoding="utf-8").read().strip() or "0")
    except Exception:
        prev = 0
    if prev and step < prev - MAX_STEP_DROP:
        incident(f"STEP_REGRESSION current={step} max_observed={prev} drop={prev - step}")
        if alert_due("step_regress"):
            mark_alert("step_regress")
            print(f"🔴 STEP REGRESSION: training is at {step} but already reached {prev} "
                  f"(lost {prev - step}). Likely a restart from scratch — a healthy "
                  f"checkpoint was lost or the launcher found no valid resume.")
        return  # do NOT lower the max: keep alerting until it climbs back
    if step > prev:
        try:
            with open(MAXSTEP_STATE, "w", encoding="utf-8") as f:
                f.write(str(step))
        except Exception:
            pass


def crash_loop_watch():
    """Detect CUDA crash + resume loop on the SAME checkpoint. Returns True if alerted.

    A resume-safe launcher is a good idea until the underlying GPU link is
    broken: then it resurrects the run from the same step forever, and every
    resurrection looks like progress to a naive watchdog.
    """
    try:
        st = json.load(open(WATCH_STATE, encoding="utf-8"))
    except Exception:
        st = {}
    out, ok = medida(
        f"a=$(grep -ac 'CUDA error' {LOG} 2>/dev/null); "
        f"b=$(grep -ac 'Resuming from step' {LOG} 2>/dev/null); "
        f"c=$(grep -a 'Resuming from step' {LOG} | tail -1 | grep -oE '[0-9]+' | tail -1); "
        f'echo "$a $b $c"', timeout=90)
    if not ok:
        return False
    nums = (out or "").split()
    if len(nums) < 2:
        return False
    try:
        cuda_n, resumes = int(nums[0]), int(nums[1])
    except ValueError:
        return False
    step = nums[2] if len(nums) > 2 else "?"
    alerted = False
    if st.get("seeded") and cuda_n > st.get("cuda", 0):
        incident(f"CUDA_CRASH new (resumes {st.get('resumes',0)}->{resumes}, step {step}, total={cuda_n})")
        if alert_due("cuda"):
            mark_alert("cuda")
            print(f"🔴 New 'CUDA error' in the training log (total {cuda_n}). Last resume: "
                  f"step {step} (#{resumes}). The launcher brought it back — confirm it is "
                  f"advancing. If the same step repeats {CRASH_LOOP_RESUMES}x the fault is on "
                  f"the host, not the process. Trail: {os.path.basename(ILOG)}")
            alerted = True
    if (not alerted and st.get("seeded") and step != "?"
            and resumes - st.get("loop_base", 0) >= CRASH_LOOP_RESUMES
            and step == st.get("loop_step")):
        if alert_due("cuda"):
            mark_alert("cuda")
            incident(f"CRASH_LOOP step={step} ({resumes - st.get('loop_base', 0)} resumes in a row)")
            print(f"🔴 CRASH LOOP: {resumes - st.get('loop_base', 0)} resumes on step {step}. "
                  f"Relaunching is not fixing it — needs host intervention "
                  f"(GPU/driver/`wsl --shutdown`).")
            alerted = True
    if not st.get("seeded") or (step != "?" and step != st.get("loop_step")):
        st["loop_base"] = resumes
        st["loop_step"] = step
    st.update(cuda=cuda_n, resumes=resumes, seeded=True)
    try:
        with open(WATCH_STATE, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except Exception:
        pass
    return alerted


# ── the action ────────────────────────────────────────────────────────────
def relaunch():
    """Detached relaunch + positive confirmation. `RELAUNCHED=<pid>` proves nothing.

    🔴 Launching in foreground through a timeout kills the trainer when the
    timeout fires (the launcher `exec`s python, so the timeout hits training
    itself, silently). Always setsid + nohup + </dev/null, then prove life.
    """
    out, ok = medida(f"setsid nohup bash {LAUNCHER} "
                     f">> {os.path.dirname(LOG)}/launch_watch.log 2>&1 < /dev/null & "
                     f"echo RELAUNCHED=$!", timeout=45, tries=2)
    ok = ok and "RELAUNCHED=" in out
    if not ok:
        print(f"RELAUNCH_FAIL: launcher call failed — out={(out or '').strip()[:150]}")
        return False
    time.sleep(RELAUNCH_CONFIRM_S)
    pid2, med2 = alive()
    if pid2:
        return True
    if not med2:
        log("post-relaunch not measured — staying quiet")
        return True
    print("RELAUNCH_FAIL: launcher ran but the trainer did not survive "
          f"{RELAUNCH_CONFIRM_S}s — check {LOG} and nvidia-smi in the guest.")
    return False


def kill_group(pid):
    """Kill the process AND its group — a worker pool survives a bare kill."""
    wsl(f"kill -9 {pid} 2>/dev/null; sleep 3; "
        f"kill -9 -$(ps -o pgid= -p {pid} 2>/dev/null | tr -d ' ') 2>/dev/null; true")


def main():
    if not LAUNCHER:
        print("TATU_LAUNCHER is not set — the watchdog cannot relaunch anything.")
        return 2
    if crash_loop_watch():
        return 0

    pid, measured = alive()

    if pid:
        check_step_regression(current_step())
        age = log_age_min()
        if age is None or age <= THRESH_WEDGE_MIN:
            log(f"OK alive pid={pid} log_age={'?' if age is None else round(age)}min")
            return 0  # silence = healthy

        adv = cpu_advancing(pid)
        if adv is None:
            log(f"log silent {age:.0f}min but CPU not measured — not a wedge")
            return 0

        if adv:
            if age < THRESH_ZOMBIE_MIN:
                log(f"OK alive pid={pid} log_age={round(age)}min cpu=active")
                return 0
            frozen = stack_frozen(pid)
            if frozen is not True:
                log(f"log silent {age:.0f}min cpu=active "
                    f"stack={'moving' if frozen is False else 'not measured'} — not a wedge")
                if frozen is None and age > ZOMBIE_BLIND_MAX_MIN and alert_due("blind"):
                    mark_alert("blind")
                    incident(f"BLIND_ZOMBIE pid={pid} silent={age:.0f}min "
                             f"stack unmeasurable >{ZOMBIE_BLIND_MAX_MIN}min")
                    print(f"🔴 Log silent {age:.0f}min with CPU active but the stack is "
                          f"UNMEASURABLE (py-spy/sudo failing) — same shape as a zombie that "
                          f"ate 5 hours unnoticed. Check py-spy access in the guest manually.")
                return 0
            log(f"ZOMBIE pid={pid} silent={age:.0f}min cpu=active frozen_stack — killing")
        else:
            log(f"WEDGE pid={pid} silent={age:.0f}min cpu=stopped — killing")

        g = gpu_ok()
        if g is False:
            log("WEDGE + GPU inaccessible — alerting instead of a relaunch loop")
            if alert_due():
                mark_alert()
                print("🔴 Trainer WEDGED and the guest GPU is INACCESSIBLE (NVML blocked by "
                      "the OS). Not relaunching — it would die in seconds and spam alerts. "
                      "Cure on the host: `wsl --shutdown`, boot the distro, verify nvidia-smi, "
                      "then run the launcher. (Alerts at most once/hour.)")
            return 0
        if g is None:
            log("WEDGE but GPU not measurable (blind bridge) — no decision this tick")
            return 0

        kill_group(pid)
        time.sleep(5)
        if relaunch():
            print(f"🧟→🟢 Trainer was alive but made no progress (log silent {age:.0f}min, "
                  f"{'CPU stopped' if not adv else 'CPU spinning on a frozen stack — zombie'}). "
                  f"Killed and RELAUNCHED from the last valid checkpoint.")
        return 0

    if not measured:
        log("NOT MEASURED (probe did not complete) — no decision this tick")
        return 0

    # Measured empty, but the flock owner may still be alive: that is a bad
    # measurement, not a death. Without this guard the watchdog relaunches a
    # living run and reports a resurrection that never happened.
    holder = lock_holder_alive()
    if holder is True:
        log("pgrep empty BUT flock owner alive — bad measurement, not death")
        return 0
    if holder is None:
        log("pgrep empty and flock owner NOT MEASURED — no relaunch this tick")
        return 0

    finished, why = run_finished()
    if finished:
        log(f"END detected: {why}")
        print(f"🏁 Training COMPLETED — {why}")
        return 0

    g = gpu_ok()
    if g is None:
        log("GPU not measurable (blind bridge) — no decision this tick")
        return 0
    if g is False:
        log("DEAD + GPU inaccessible (NVML blocked) — relaunch skipped")
        if alert_due():
            mark_alert()
            print("🔴 Trainer DEAD and the guest GPU is INACCESSIBLE (NVML 'GPU access "
                  "blocked'). Relaunching does not help — the process dies in seconds. Cure "
                  "on the Windows host: `wsl --shutdown`, boot, check nvidia-smi, run the "
                  "launcher. (Repeats at most once/hour.)")
        return 0

    log("DEAD with no end banner — relaunching via resume-safe launcher")
    if relaunch():
        print("🔴→🟢 Trainer DIED without an end banner — RELAUNCHED and alive, resumed "
              "from the last valid checkpoint. Watch continues.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
