# Diagnosis decision table — which probe answers which question

The recurring lesson from every incident behind this repo: **a single probe
never settles a WSL/GPU question.** Utilisation, process existence and log
staleness each lie in at least one real failure mode. This table is the
shortest path from symptom to the measurement that actually discriminates.

All commands assume the trainer pattern is `$PROC` (bracket it:
`pgrep -f "[${PROC:0:1}]${PROC:1}"` — an unbracketed probe matches its own
command line and reports a ghost PID).

## Symptom → referee

| Symptom | Do NOT trust | The measurement that decides | Reading |
|---|---|---|---|
| Log silent > stale threshold, process present | `nvidia-smi` util (100% util happens in every state below) | CPU delta via `/proc/$P/stat`: `awk '{print $14+$15}'` ×2 with 30s gap | delta=0 → **wedge** (kill + relaunch); delta ≈ 1 core/30s → **grind** (wait) |
| CPU delta > 0 but zero progress for hours | CPU activity (a zombie spins) | `py-spy dump` ×2, 15-20s apart, hash the MainThread section — compare the FULL dump, never a slice | identical → **zombie** (kill); frames move → real work |
| Everything looks healthy but steps take 10x longer | binary `torch.cuda.synchronize()` test (passes at 2% throughput) | `scripts/cuda_tflops_bench.py` from a SEPARATE process | ≥4.7 TFLOPS healthy (3060 FP32); 0.3-1.1 degraded → `wsl --shutdown` cure; mid-band under load = inconclusive, bench after killing |
| `wsl.exe` calls return empty / rc=0 garbage | the emptiness itself (timeout under load returns empty; localized Windows emits translated text, sometimes rc=0) | sentinel: append `; echo __MEASURED__` and require it back | no sentinel → **not measured**; take NO destructive action this tick |
| pgrep empty, want to relaunch | pgrep alone (false negatives under thrash) | flock holder: `p=$(cat $LOCK); kill -0 $p` + `ps aux --sort=-rss \| head` + `/proc/$P/cmdline` | holder alive → bad measurement, do not relaunch |
| GPU util 100% but tok/s 3-4x below history | one power.draw sample (data-loading gaps read low) | sustained: multiple probes + step cadence + host `\GPU Process Memory\Shared Usage` (`scripts/host_vram_probe.ps1`) | Shared 1.5-2GB → sysmem fallback; power 60W/170W sustained → PCIe-bound, not computing |
| VRAM ~12GB, no visible owner | `--query-compute-apps` inside guest (`[N/A]` PIDs, GUI decoys) | host counters per PID; the WSL owner is `vmwp.exe` — that IS your guest context, not a thief | confirm with in-guest list; if only the trainer, footprint is real |
| Resume hangs on first forward, fresh runs fine | "the checkpoint is corrupt" (it may be fine on disk) | two causes: (A) `cp` through 9p under <3GB host free RAM truncates SILENTLY — hash natively vs in-guest; (B) `map_location=device` in resume inflates VRAM ~2GB | (A) stage with `dd bs=8M` to ext4, verify sizes; (B) load on CPU, `del`, `empty_cache()` → success signature: ~6GB VRAM + step lands <2min |
| Same `Resuming from step N` banner repeating | the banner itself (it prints every resurrection) | resume count per step (`roger/night_watch.py::crash_loop_watch`) | ≥3 on the same step → host fault, not process fault; stop relaunching |
| Step number DROPPED, run otherwise healthy | "it's progressing, look at the telemetry" | max-step-ever-seen vs current (`check_step_regression`) | drop > 1 legit resume window → boot wiped staging or launcher fell back to scratch |
| A report says `step 999`, log says `step 60` | the report, always | the critic compares claimed vs parsed (`critic.py`) | claims > log → phantom self-report → escalate |
| VM gone: UTF-16-spaced output, exit 127 | WSAETIMEDOUT alone (intermittent under GPU load with a LIVE VM) | host side: `ls -lt /mnt/<drive>/<outdir>` — ckpt mtimes advancing = VM alive, command channel wedged | boot time fresh (`uptime -s`) AND mtimes frozen → VM dead; shutdown + relaunch is then safe |

## Thresholds are per-phase, not global

- Stale threshold ≈ **3-4x the slowest phase's step cadence**. Co-training
  dual-stream steps ran 12-25 min (first post-resume: 47min) on a 3060 — a
  20-min threshold false-alarms every single step of that phase.
- `--grad-accum N` multiplies wall time per logged step by N.
- The first post-resume step is always fast (short step) and always lies about
  ETA. Recalculate cadence from step 2 onward. Dual-stream measured 5-11
  tok/s as its NATURAL rate where pretrain ran ~38 — never compare phases.

## Order of operations before any kill/relaunch

1. Measure liveness with a sentinel (never empty = dead).
2. If alive: CPU delta → py-spy (full dumps) → bench, in that order.
3. Check you are not racing another relauncher: read `/tmp/<train>_*.log`,
   `ls -t` checkpoint dirs on every candidate disk, look for `*_CORRUPTED*`
   renames — another session may have already diagnosed and migrated.
   Any sign of a parallel session → do not relaunch; report instead.
4. Kill the process GROUP (a worker pool survives a bare `kill $pid`).
5. Relaunch detached (`setsid nohup ... </dev/null`), then PROVE the new
   process survived ~45s. `RELAUNCHED=<pid>` on stdout proves nothing.
6. Log the decision with a timestamp — the incident trail is what turns
   the next 03:00 page into a five-minute diagnosis.
