# Tatu Edge MLOps

Resilient local MLOps for training on the edge: a Windows host running the
trainer inside WSL, hybrid NVMe/HDD mounts, and a 12GB GPU that is shared
with the desktop.

These tools assume nobody is awake at 03:00. Silence means healthy, and every
alarm has to survive a positive measurement before it fires.

**Model-agnostic.** Nothing here knows about a specific architecture,
tokenizer or dataset. It is pure operational machinery: it watches a training
loop, refuses to trust self-reported progress, and brings the run back from a
checkpoint when the platform under it fails silently.

---

## The problem

Long training runs on Windows + WSL do not fail loudly. They fail in ways
that look exactly like health:

| Failure mode | Why it fools you | What it costs |
|---|---|---|
| **Silent trainer death** | `pgrep` under load can return empty for a process that is alive; a translated `wsl.exe` error can arrive with `rc=0` | a watchdog "resurrects" a run that never died, or misses one that did |
| **CPU zombie** | State `R`, CPU ticking, GPU context open — zero progress, same Python line for hours | 5 hours of a 12GB card doing nothing |
| **Desiccated GPU-PV channel** | `nvidia-smi` reports 97-100% util at full clocks, py-spy shows frames advancing, a binary CUDA test passes in <1s — yet the channel runs at ~2% throughput | a 12-minute step becomes 2h15; a 20-minute checkpoint load becomes 20min |
| **9p/drvfs truncation** | A large tensor read through `/mnt/<drive>` fails with `ENOMEM` under host RAM pressure and leaves a destination whose **size still matches** the resume check | resume loads half a model and NaNs an hour later |
| **Masked OOM / sysmem fallback** | Orphan CUDA contexts from earlier crashes hold VRAM at the driver layer; the trainer quietly spills to system RAM over PCIe and *keeps making progress* | a 3-4x throughput collapse read as "the model is just slower now" |
| **Step regression** | A boot clears `/tmp`, the launcher falls back to a scratch copy, and training resumes hundreds of steps behind — perfectly healthy telemetry | hundreds of silent steps of lost work |
| **Crash loop on the same checkpoint** | A resume-safe launcher is a good idea until the fault is in the host; then it resurrects from the same step forever, and every resurrection looks like progress | an overnight loop of identical deaths |
| **Phantom self-report** | An agent-written report claims `step 999` while the log says `step 60` | a whole run believed complete that never was |
| **Relaunch race** | Two watchdogs relaunch the same crash; two writers hit one checkpoint directory | checkpoint corruption at the next save |

Separately: file-based I/O across the WSL boundary (`drvfs`/`9p`) is slow and
lossy in exactly the situations where you are writing the most data.

## The solution

Three pieces, deliberately boring:

1. **Deterministic critic (`tatu/roger_tatu.py`) — the "Roger Critic".**
   It judges a run against the **real log**, never against a summary of it.
   It recomputes the learning rate the schedule *should* show at the observed
   step and compares it to the logged LR — that single check catches the dead
   LR coma, the cosine-blowup, and the decimal-comma typo (`lr=0,00005`
   parses as `0.0` in a shell). It scores 0-100 and returns exactly one of
   `continue` / `investigate` / `escalate`. It **fails open**: a broken critic
   degrades to `continue`, never to a blocked run.

2. **Asynchronous file IPC over ext4 (`tatu/roger_client.py`).**
   The training loop writes a JSON request into a spool and never blocks on a
   daemon: no HTTP server, no sockets, no shared-memory library, nothing that
   has to survive a reboot of a guest. Requests land in
   `~/.tatu/roger/requests/`, answers in `responses/`, and every decision is
   appended to `ledger.jsonl`. A client timeout yields a `continue` verdict —
   the watchdog can be down and training still cannot wedge.
   This is the same discipline applied to checkpoint I/O: canonical files on
   ext4, the slow mount treated as an advisory mirror, never as the source of
   truth (`tatu/safe_io.py`, `deploy/launch_train.sh`).

3. **Watchdogs that actually cure (`tatu/night_watch.py`, `tatu/gpu_cure.py`).**
   Not alert-only. Dead without an end banner → relaunch resume-safe.
   Wedged → kill the process group, relaunch. Channel degraded → `wsl
   --shutdown`, re-bench, relaunch, and *refuse* to relaunch into a GPU that
   is still broken. Every action requires a positive measurement: an
   unreadable probe is never evidence of death.

## Layout

```
tatu/
  config.py          all paths from env; zero personal paths in code
  roger_tatu.py      deterministic critic + flock IPC daemon (fail-open)
  roger_client.py    submit/poll client with fail-open timeout
  night_watch.py     in-guest watchdog: zombie, wedge, step regression, crash loop
  gpu_cure.py        host watchdog: GPU-PV channel cure (wsl --shutdown)
  safe_io.py         atomic publish, .part + size + trailer verify, mirror
scripts/
  cuda_tflops_bench.py  throughput referee (catches the 2%-channel state)
  quarantine.py      move a bad checkpoint OUT OF THE GLOB; verify before trusting
  host_vram_probe.ps1    per-process VRAM attribution from the Windows side
deploy/
  roger-tatu.service     systemd **user** unit for the critic daemon
  keepalive.service      systemd user unit for the relaunch loop
  keepalive.sh           guest-side 24/7 relaunch loop (completion is parsed, not grepped)
  launch_train.sh        reference launcher: flock + pgrep + VRAM gate + 9p staging
tests/
  test_roger_tatu.py     8 scenarios against the real critic, no GPU needed
docs/
  diagnosis-decision-table.md  which probe answers which question
examples/
  train_with_roger.py    minimal trainer wired to the critic
```

## Quick start

```bash
# 1. critic daemon (inside the Linux/WSL guest)
mkdir -p ~/.tatu/roger
cp tatu/*.py ~/.tatu/roger/
export TATU_HOME=$HOME/.tatu TATU_TRAIN_LOG=$HOME/.tatu/train.log TATU_PROC=run_train
python3 ~/.tatu/roger/roger_tatu.py --daemon &      # or install the systemd user unit

# 2. from the training loop, every N steps
python3 -m tatu.roger_client --step 600 --loss 4.20 --grad 120 \
    --lr 2.5e-5 --vram 9.9 --tok 430 --question "checkpoint 600"

# 3. prove the critic is sane before trusting it
python3 tests/test_roger_tatu.py            # 8/8 PASS, no GPU

# 4. is the GPU channel actually healthy?
python3 scripts/cuda_tflops_bench.py --min-tflops 2.0
```

Wiring the critic into a training loop is ~10 lines and never raises:
see [`examples/train_with_roger.py`](examples/train_with_roger.py).

## Configuration

Every path and threshold comes from the environment — see
[`tatu/config.py`](tatu/config.py) for the full contract. The important ones:

| Variable | Meaning | Default |
|---|---|---|
| `TATU_HOME` | IPC spool + daemon state root | `~/.tatu` |
| `TATU_TRAIN_LOG` | append-only training log the critic reads | `$TATU_HOME/train.log` |
| `TATU_PROC` | cmdline pattern of the trainer process | `run_train` |
| `TATU_PERSIST` | canonical ext4 checkpoint dir | `/tmp/tatu_ckpt_persist` |
| `TATU_MIRROR` | optional second-drive mirror (written last, verified apart) | off |
| `TATU_LAUNCHER` | resume-safe launch script the watchdogs may call | — |
| `TATU_TOTAL_STEPS` / `TATU_LR` / `TATU_WARMUP` / `TATU_END_LR` | the schedule the critic recomputes | `2000 / 5e-5 / 50 / 5e-8` |
| `TATU_GATE` | score threshold for `investigate` | `85` |
| `TATU_BENCH_MIN` | TFLOPS below which the GPU-PV channel is considered degraded | `2.0` |

The log parser accepts two shapes out of the box (single `Loss:` line, and a
two-term `P:/S:/Total:` line) — extend `RE_STEP_*` in `roger_tatu.py` for your
own format. The critic only needs step, loss, grad norm, LR, tok/s and VRAM.

## Design rules these files follow

1. **Fail open.** A watchdog that can block training is worse than no watchdog.
2. **Measure or stay quiet.** Every destructive action needs a sentinel-backed
   positive measurement. Empty output means the probe failed, not that the
   process died.
3. **Judge reality, not reports.** The log, the checkpoint bytes and the
   process table are evidence; a claimed step number is a suspicion.
4. **Never bake in a path.** Person, drive letter, hostname and chat ID go in
   the environment. The code is portable because it is anonymous.
5. **Size is not proof.** A file is valid when its size, trailer and (for a
   resume target) load agree.
6. **One launcher owns the race.** flock *and* pgrep, `flock -n` always, so a
   second relauncher aborts instead of queuing.
7. **Alerts must be rate-limited and attributable.** Per-category cooldown,
   plus a dedicated timestamped incident log so a fix can be audited later.

## What this is not

Not a job scheduler, experiment tracker or hyperparameter search. Not a
cluster tool — it targets one box, one GPU, one long run. Not a substitute for
checkpoint discipline: every tool here assumes your launcher can resume and
your saves are the thing you would want to keep.

## License

MIT — see [LICENSE](LICENSE).
