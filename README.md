<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-6366f1?style=flat-square" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <img src="https://img.shields.io/badge/dependencies-0-22c55e?style=flat-square" alt="Zero dependencies">
  <img src="https://img.shields.io/badge/OS-Windows%2011%20%2B%20WSL2-fuchsia?style=flat-square" alt="Windows + WSL2">
</p>

<h1 align="center">Roger — Resilient Local MLOps</h1>

<p align="center"><strong>Keeps long unattended GPU training runs honest on WSL2 and edge hardware</strong> — a deterministic critic, watchdogs that act, and crash-safe checkpoint I/O. Zero dependencies, model-agnostic.</p>

> 🌐 **English** · [🇧🇷 Português](README.pt-BR.md)

Long unattended GPU training on a Windows + WSL2 workstation fails where nobody is watching: the trainer stays alive while its log stops, the NVML bridge dies on sleep/hibernate, a checkpoint resumes past a truncated save, and an agent loop happily inherits yesterday's perfect score. None of it raises an error at 03:00 — the run just quietly gets worse.

Roger is the guard layer that turns each of those into a **handled event**:

- a **deterministic critic** that scores the run against the curves you declared (LR vs schedule, gradient norm, loss velocity, tokens/s, VRAM, log staleness) and files every verdict with its evidence in an append-only ledger;
- **watchdogs that act** — kill a wedged trainer and relaunch it from the last good checkpoint, under a relaunch budget with a crash-loop breaker;
- a **GPU-PV bridge cure** that performs the surgical WSL2 restart the guest can never request for itself;
- **crash-safe checkpoint I/O** built for 9p/drvfs: a save either exists complete or does not exist — the half-written state that corrupts resumes is removed from the possible;
- **orchestrator hygiene** that keeps agentic loops honest: untrusted tool output quarantined as data, per-campaign state fingerprinting, regression and inconclusive-round sentinels.

Zero dependencies. Model-agnostic. Every guard carries a unit test written against the incident that produced it. If you train on your own hardware, this is the reliability layer your scheduler should have shipped with.

## The failure modes, and the guard that kills each

| Symptom (silent unless guarded) | Guard | Module |
|---|---|---|
| NVML bridge `found a PCI device but no GPUs found` after sleep/hibernate; guest looks healthy forever | Host-side cure: probe via the bridge, surgical `wsl --shutdown` + relaunch, arbitration of real TFLOPS before trusting the channel | `roger/gpu_cure.py`, `scripts/cuda_tflops_bench.py` |
| Trainer process alive but wedged: 100% CPU, log frozen, GPU 0 | Zombie-kill with relaunch budget; step-regression detection; crash-loop breaker | `roger/night_watch.py` |
| Host reports "2 GB used by WSL" — OOM masked in shared memory | Host truth probe: enumerate VM processes, sum WS, attribute per-process VRAM | `scripts/host_vram_probe.ps1` |
| Checkpoint truncated in flight on 9p/drvfs and resumes past garbage bytes | `.part` + fsync + read-back verify + `os.replace`; quarantine outside the resolver glob | `roger/safe_io.py`, `scripts/quarantine.py` |
| Critic grades via HTTP/JSON: fragile, auth walls, timeouts | File IPC on ext4 with flock, one-shot mode, fail-open verdicts, append-only ledger | `roger/critic.py`, `roger/roger_client.py` |
| An agent loop inherits yesterday's 100/100 because a state file outlived the campaign | Campaign fingerprint (git HEAD) + state reconciliation + regression detection | `roger/hygiene.py` |
| Audited repo's tool output injects instructions into your agent | Untrusted-data quarantine (fences, control chars, truncation) | `roger/hygiene.py` |

## Architecture

```
 trainer (any framework, any machine)
   |  atomic + verified checkpoints ....... roger/safe_io.py
   |  ask "should I keep going?" (non-blocking, 90s fail-open)
   v
 Roger critic daemon  <── file IPC, flock on ext4, never 9p/HTTP
   |  deterministic score 0-100 + verdict {continue|watch|investigate|escalate}
   |  ledger.jsonl (append-only evidence)
   v
 night_watch (guest) — zombie/step-regression/crash-loop -> kill + relaunch
 gpu_cure   (host)  — NVML/dxgkrnl bridge cure, surgical WSL restart
 keepalive (guest) — resume-safe relaunch loop (systemd --user)
```

The critic **never blocks training** — every dependency fails open with an alert. Roger's opinion is advisory by contract: it gates the *campaign*, not the *epoch*.

## Installation

No dependencies. Python 3.10+ on each side of the bridge.

```bash
git clone <this-repo> && cd roger-mlops
cp .env.example ~/.config/roger.env   # edit paths to your layout
export ROGER_HOME=$HOME/.roger ROGER_TRAIN_LOG=$HOME/.roger/train.log
python3 -m unittest discover -s tests -v   # all green before you trust it
```

Systemd user units live in `deploy/`:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/roger.service deploy/keepalive.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now roger keepalive
```

## Commands

| Command | What it does |
|---|---|
| `python3 roger/critic.py --daemon` | Run the critic daemon (file IPC, flock'd spool) |
| `python3 roger/critic.py --once` | Drain pending requests once (cron-friendly, no daemon) |
| `python3 roger/roger_client.py ask --project NAME --log PATH [--dry-run]` | Ask for a verdict; prints `VERDICT=`/`SCORE=`; fail-open |
| `python3 roger/night_watch.py --once` | One zombie/regression/loop check; kills + relaunches |
| `python3 roger/gpu_cure.py --probe-only` | Classify the GPU bridge state without curing |
| `python3 roger/hygiene.py` | Self-test of every orchestrator guard |
| `python3 scripts/quarantine.py move CKPT_DIR CKPT` | Move a suspect checkpoint out of the resolver glob |
| `python3 scripts/cuda_tflops_bench.py` | Measure real TFLOPS (the GPU-throughput referee) |
| `pwsh scripts/host_vram_probe.ps1` | Host-side truth: WSL VM working set + per-process VRAM |
| `bash deploy/launch_train.sh` | Reference resume-safe launcher (flock + pgrep double guard) |

## Critic contract

Requests and verdicts are JSON across `request_*.json` files (schema in
`roger/critic.py`); the deterministic checks cover the curves a human
would eyeball at 03:00 — LR vs the cosine schedule you declared, gradient
norm blowups, loss velocity, field drift (e.g. `|A_log|`), tok/s, VRAM,
log staleness, process presence — each with weights and a gate. If you plug
a critic into your own loop, it must print `SCORE=<n>` / `GAPS=<a|b>` /
`DETAIL=<...>`; `hygiene.parse_critic_output()` refuses to grade a critic
that broke the contract instead of silently scoring 0.

## Why files, not sockets

On a hybrid-storage laptop under training load, the Windows↔WSL bridges lie:
directory metadata reads raise I/O errors, `wsl.exe` output arrives empty,
ports on mirrored localhost belong to `wslrelay` even when the service is
dead, and `df` through 9p reports cached numbers. A request written to ext4
with flock and fsync either exists complete or does not exist — there is no
third state. That property is the whole design.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Covers: verdict weights and gate, malformed-request fail-open, duplicate
daemon refusal, quarantine bypass attempts, inherited-grade campaign
archival, flapping-gap regression, `0 passed` environment rounds,
checkpoint-verify against truncation, and the launcher's done-marker parse.

## License

MIT — see [LICENSE](LICENSE).
