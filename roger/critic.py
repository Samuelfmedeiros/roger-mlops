#!/usr/bin/env python3
"""Roger — deterministic training critic + file-based IPC daemon.

Two roles, one file:

R1  A deterministic critic that judges a training run against the REAL log,
    never against a self-report. Scores 0-100 and returns one of three
    actions: continue / investigate / escalate.

R2  An IPC server over plain JSON files + flock on a local filesystem.
    No HTTP, no sockets, no drvfs/9p. A request lands in
    ``<root>/requests/request_<id>.json``; the daemon answers in
    ``<root>/responses/response_<id>.json`` and appends an audit line to
    ``<root>/ledger.jsonl``. Processed requests move to ``processed/``.

Golden rule: FAIL-OPEN. Any broken dependency degrades to ``continue``.
The critic must never wedge the run it is watching — a silent critic is a
bug, a blocking critic is an incident.

Environment (see roger/config.py):
    ROGER_ROOT, ROGER_LOG, ROGER_PERSIST, ROGER_PROC,
    ROGER_LR, ROGER_WARMUP, ROGER_TOTAL_STEPS, ROGER_END_LR, ROGER_GATE,
    ROGER_VRAM_MAX, ROGER_GRAD_MAX, ROGER_AD_MAX, ROGER_LOG_MAX_AGE
"""
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile

try:
    from . import config as C
except ImportError:  # executed as a plain script
    import config as C

BASE = os.path.expanduser(C.cfg("ROOT", os.path.expanduser("~/.roger/roger")))
if BASE.endswith("roger"):
    REQ = os.path.join(BASE, "requests")
    RESP = os.path.join(BASE, "responses")
    PROC = os.path.join(BASE, "processed")
    LEDGER = os.path.join(BASE, "ledger.jsonl")
else:
    _r = os.path.join(BASE, "roger")
    REQ = os.path.join(_r, "requests")
    RESP = os.path.join(_r, "responses")
    PROC = os.path.join(_r, "processed")
    LEDGER = os.path.join(_r, "ledger.jsonl")

LOG = C.cfg("LOG", os.path.join(BASE, "train.log"))
PERSIST = C.cfg("PERSIST", "/tmp/roger_ckpt_persist")
PROC_PATTERN = C.cfg("PROC", "run_train")

TRAIN_LR = float(C.cfg("LR", "5e-5"))
WARMUP = int(C.cfg("WARMUP", "50"))
TOTAL = int(C.cfg("TOTAL_STEPS", "2000"))
END_LR = float(C.cfg("END_LR", "5e-8"))
GATE = int(C.cfg("GATE", "85"))
VRAM_MAX = float(C.cfg("VRAM_MAX", "11.5"))
GRAD_MAX = float(C.cfg("GRAD_MAX", "1000"))
AD_MAX = float(C.cfg("AD_MAX", "0.15"))
LOG_MAX_AGE = int(C.cfg("LOG_MAX_AGE", "1800"))

# Two log shapes are supported out of the box: a single-loss line and a
# two-term (pretrain/sft + total) line. Step total is intentionally
# permissive so the same critic works across runs of different length.
RE_STEP_A = re.compile(
    r"Step\s+(\d+)/\d+\s*\|\s*(?:Pretrain )?Loss:\s*([\d.]+)\s*\|\s*Grad:\s*([\d.]+)"
    r"\s*\|\s*LR:\s*([\deE.+-]+)\s*\|\s*(\d+)\s*tok/s\s*\|\s*VRAM:\s*([\d.]+)\s*GB")
RE_STEP_B = re.compile(
    r"Step\s+(\d+)/\d+\s*\|\s*P:\s*([\d.]+)\s*\|\s*S:\s*([\d.]+)\s*\|\s*Grad:\s*([\d.]+)"
    r"\s*\|\s*Total:\s*([\d.]+)\s*\|\s*LR:\s*([\deE.+-]+)\s*\|\s*(\d+)\s*tok/s"
    r"\s*\|\s*VRAM:\s*([\d.]+)\s*GB")
RE_AD = re.compile(r"grads:\s*[A-Za-z_0-9/]+\s*=\s*([\deE.+-]+)")


def sh(cmd, t=30):
    """Best-effort shell helper. Empty string on ANY failure (fail-open)."""
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, timeout=t)
        return (r.stdout or b"").decode("utf-8", "replace")
    except Exception:
        return ""


def read_log_tail(n=400):
    try:
        with open(LOG, errors="replace") as f:
            return f.readlines()[-n:]
    except Exception:
        return []


def parse_steps(lines):
    """-> [(step, loss, grad_total, lr, tok_s, vram_gb)] ascending by file order."""
    out = []
    for ln in lines:
        m = RE_STEP_A.search(ln)
        if m:
            g = m.groups()
            out.append((int(g[0]), float(g[1]), float(g[2]), float(g[3]), int(g[4]), float(g[5])))
            continue
        m = RE_STEP_B.search(ln)
        if m:
            g = m.groups()
            # P: pretrain, S: sft, Grad, Total -> judge the TOTAL
            out.append((int(g[0]), float(g[4]), float(g[3]), float(g[5]), int(g[6]), float(g[7])))
    return out


def expected_lr(step):
    """Linear warmup then cosine decay — the curve the trainer should show."""
    if step <= WARMUP:
        return TRAIN_LR * step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, (TOTAL - WARMUP))
    return END_LR + 0.5 * (TRAIN_LR - END_LR) * (1 + math.cos(math.pi * prog))


def last_ad_grad(lines):
    for ln in reversed(lines):
        m = RE_AD.search(ln)
        if m:
            return float(m.group(1))
    return None


def newest_ckpt():
    import glob
    ds = sorted(glob.glob(os.path.join(PERSIST, "step_*")))
    return ds[-1] if ds else None


def collect_observation():
    """Read reality: log tail, its age, NaN, ckpt integrity, process liveness."""
    lines = read_log_tail()
    steps_d = parse_steps(lines)
    age = sh("stat -c %%Y " + LOG) if os.name != "nt" else ""
    try:
        log_age = max(0.0, time.time() - float(age.strip()))
    except Exception:
        log_age = 9999.0
    ad = last_ad_grad(lines)
    nan = any("nan" in ln.lower() for ln in lines[-30:])
    ckpt_ok = False
    ckdir = newest_ckpt()
    if ckdir:
        try:
            ckpt_ok = zipfile.is_zipfile(os.path.join(ckdir, "pytorch_model.bin"))
        except Exception:
            ckpt_ok = False
    alive = bool(sh("pgrep -f '[%s]' | head -1" % PROC_PATTERN[:1] + PROC_PATTERN[1:]).strip())
    return {"steps": steps_d, "ad": ad, "log_age": log_age, "nan": nan,
            "ckpt_ok": ckpt_ok, "alive": alive}


def evaluate(req):
    """The verdict. `req['sim']` short-circuits observation for tests."""
    findings, hard = [], []
    score = 100
    sim = req.get("sim") or {}
    ts = time.time()

    if sim:
        steps_d = [tuple(x) for x in (sim.get("steps") or [])]
        ad = sim.get("ad_grad")
        log_age = sim.get("log_age", 0)
        nan = sim.get("nan", False)
        ckpt_ok = sim.get("ckpt_ok", True)
        alive = sim.get("alive", True)
    else:
        obs = collect_observation()
        steps_d, ad, log_age = obs["steps"], obs["ad"], obs["log_age"]
        nan, ckpt_ok, alive = obs["nan"], obs["ckpt_ok"], obs["alive"]

    if not alive:
        hard.append("trainer process absent")
    if log_age > LOG_MAX_AGE:
        hard.append(f"log silent for {log_age/60:.0f}min (wedge?)")
    if nan:
        hard.append("NaN in log")
    if len(steps_d) < 2:
        hard.append("no parsable steps in log")
        steps_d = []

    last = steps_d[-1] if steps_d else None
    if last:
        s, loss, grad, lr, tok, vram = last

        # A claimed step ahead of the log is the classic hallucinated report.
        claimed = int(req.get("step", 0) or 0)
        if claimed > s + 2:
            hard.append(f"request claims step {claimed} > log step {s} (phantom self-report)")

        # LR vs the expected schedule — this is the check that catches a run
        # that silently fell into a dead-LR coma (or a decimal-comma typo).
        exp = expected_lr(s)
        if lr <= 0 or abs(lr - exp) / max(exp, 1e-12) > 0.05:
            hard.append(f"LR {lr:.2e} != expected {exp:.2e} at step {s} (coma/explosion?)")

        recent = [d[2] for d in steps_d[-5:]]
        if recent and max(recent) > GRAD_MAX:
            hard.append(f"sustained grad norm {max(recent):.0f} > {GRAD_MAX:.0f}")

        if ad is not None and ad > AD_MAX:
            score -= 15
            findings.append(f"A_log/D grad {ad:.2e} > {AD_MAX:.0e} (guard under tension)")

        if len(steps_d) >= 4 and steps_d[-1][0] - steps_d[0][0] >= 20:
            half = len(steps_d) // 2
            a = sum(d[1] for d in steps_d[:half]) / max(1, half)
            b = sum(d[1] for d in steps_d[half:]) / max(1, len(steps_d) - half)
            if b > a + 0.05:
                score -= 20
                findings.append(f"loss rising across window ({a:.3f} -> {b:.3f})")
            elif abs(b - a) <= 0.015:
                score -= 5
                findings.append(f"loss flat across window ({b:.3f})")

        if tok and tok < 15:
            score -= 10
            findings.append(f"throughput {tok} tok/s degraded (<15)")
        if vram > VRAM_MAX:
            score -= 15
            findings.append(f"VRAM {vram:.2f}GB > {VRAM_MAX}GB")
        if not ckpt_ok:
            score -= 25
            findings.append("newest checkpoint fails zip test (save in flight?)")

    score = max(0, score)
    action = "escalate" if hard else ("continue" if score >= GATE else "investigate")
    tele = {}
    if last:
        tele = {"step": last[0], "loss": last[1], "grad": last[2], "lr": last[3],
                "tok_s": last[4], "vram_gb": last[5],
                "lr_expected": round(expected_lr(last[0]), 8),
                "log_age_min": round(log_age / 60, 1)}
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": req.get("id"),
            "score": score, "action": action, "findings": findings, "hard": hard,
            "telemetry": tele, "roger": "Roger v1"}


def ledger_write(entry):
    try:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        sys.stderr.write(f"ledger failed (fail-open): {e}\n")


def _atomic_write(path, payload):
    """Write via .tmp + os.replace: a reader never sees a half response."""
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def handle_file(path):
    rid = os.path.basename(path)[len("request_"):-len(".json")]
    try:
        with open(path) as f:
            req = json.load(f)
    except Exception as e:
        # An unreadable request must NOT be judged as a dead run: evaluating
        # an empty req would escalate on "no parsable steps" and a caller
        # that wrote a half-file would get a false alarm. Fail open.
        req = None
        verdict = {"request_id": rid, "action": "continue", "score": 0,
                   "hard": [], "findings": [],
                   "note": f"fail-open: unreadable request ({e})",
                   "roger": "Roger v1"}
    if req is not None:
        try:
            verdict = evaluate(req)
        except Exception as e:  # fail-open: never block the run we watch
            verdict = {"request_id": rid, "action": "continue", "score": 0,
                       "hard": [], "findings": [],
                       "note": f"fail-open: critic error {e}",
                       "roger": "Roger v1"}
    verdict["request_id"] = rid
    _atomic_write(os.path.join(RESP, f"response_{rid}.json"), verdict)
    ledger_write({"ts": verdict.get("ts"), "id": rid,
                  "question": ((req or {}).get("question") or "")[:180],
                  "action": verdict["action"], "score": verdict.get("score"),
                  "findings": verdict.get("findings", []), "hard": verdict.get("hard", []),
                  "note": verdict.get("note", "")})
    try:
        os.replace(path, os.path.join(PROC, os.path.basename(path)))
    except Exception:
        pass


def _single_instance_lock():
    """flock where available; msvcrt on Windows; no-op elsewhere."""
    path = os.path.join(BASE, ".lock")
    fh = open(path, "w")
    try:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        try:
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise
        except Exception:
            pass
    except OSError:
        raise
    return fh  # keep the handle alive


def daemon(poll=5):
    for d in (REQ, RESP, PROC):
        os.makedirs(d, exist_ok=True)
    lock = _single_instance_lock()  # noqa: F841 — held for process lifetime
    sys.stderr.write(f"[roger] daemon up on {BASE} (poll {poll}s)\n")
    import glob
    while True:
        try:
            for p in sorted(glob.glob(os.path.join(REQ, "request_*.json"))):
                handle_file(p)
        except Exception as e:
            sys.stderr.write(f"loop failed (fail-open): {e}\n")
        time.sleep(poll)


def main(argv):
    if "--daemon" in argv:
        try:
            daemon()
        except OSError as e:
            # flock refused: a second daemon means two writers on one spool.
            # Exit clean (rc 3) instead of dumping a traceback into the journal.
            print(f"another Roger daemon already holds {BASE} ({e}) — exiting",
                  file=sys.stderr)
            return 3
    elif "--request" in argv:
        p = argv[argv.index("--request") + 1]
        with open(p) as f:
            req = json.load(f)
        print(json.dumps(evaluate(req), ensure_ascii=False, indent=1))
    elif "--once" in argv:
        # one drain pass (useful in cron/supervisor mode, no long-lived daemon)
        import glob
        for d in (REQ, RESP, PROC):
            os.makedirs(d, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(REQ, "request_*.json"))):
            handle_file(p)
    else:
        print("usage: critic.py --daemon | --once | --request <file.json>")


if __name__ == "__main__":
    main(sys.argv)
