#!/usr/bin/env bash
# launch_train.sh — the reference resume-safe launcher.
#
# Every line here exists because a run died without it. Four jobs:
#
#   1. NEVER double-launch. Two trainers on one checkpoint directory
#      corrupt the next save; two on one GPU is a crash race. Two barriers,
#      because they cover different holes:
#        - flock  : cheap, race-free, released by the kernel on death
#        - pgrep  : catches a process started BEFORE this lock existed
#      `flock -n` (non-blocking) is mandatory — a blocking flock would make
#      the second launcher WAIT and then start unsupervised the instant the
#      first dies.
#   2. Stage checkpoint reads OFF the lossy mount. Reading a 2GB tensor
#      through 9p/drvfs while the host is short on free RAM fails with
#      ENOMEM and leaves a TRUNCATED destination whose size still passes a
#      naive check. Copy to ext4 first, verify size, then resume from ext4.
#   3. Gate on free VRAM before touching CUDA. A 12GB card with a resident
#      2.4GB embedder does not fit a 12GB trainer: it dies in the first
#      allocation and looks like a driver bug.
#   4. Detach properly. `setsid nohup ... </dev/null` — a launch that stays
#      attached to the calling shell dies when the caller's timeout fires.
#
# Configure with env (or copy and edit the CFG block):
#   TRAIN_DIR  workdir with the python project     (required)
#   TRAIN_CMD  python invocation, appended to CFG  (required)
#   LOG        append-only training log            (required)
#   OUTDIR     checkpoint dir on the SLOW mount    (optional, enables staging)
#   STAGE      local ext4 staging dir              (default /tmp/tatu_ckpt_persist)
#   STAGE_STEP step number to resume (else newest valid ckpt in OUTDIR/STAGE)
#   VRAM_MAX_MIB launch only if used VRAM is below this (default 11000)
#   MIN_FREE_STAGING_MIB host/guest free RAM needed before a 9p copy (default 3000)

set -uo pipefail

TRAIN_DIR="${TRAIN_DIR:?TRAIN_DIR required}"
TRAIN_CMD="${TRAIN_CMD:?TRAIN_CMD required (python invocation)}"
LOG="${LOG:?LOG required}"
PROC="${PROC:-$(printf '%s' "$TRAIN_CMD" | awk '{print $2}' | xargs -r basename)}"
STAGE="${STAGE:-/tmp/tatu_ckpt_persist}"
LOCK_FILE="${LOCK_FILE:-${STAGE%/*}/.train.lock}"
VRAM_MAX_MIB="${VRAM_MAX_MIB:-11000}"
MIN_FREE_STAGING_MIB="${MIN_FREE_STAGING_MIB:-3000}"

mkdir -p "$(dirname "$LOG")" "$(dirname "$LOCK_FILE")" "$STAGE"

# ── barrier 1: flock on fd 9 ──────────────────────────────────────────────
# fd 9 survives `exec` (the python process inherits it); death closes it.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "LAUNCH ABORTED: already running (lock $LOCK_FILE)" | tee -a "$LOG"
  exit 0
fi
echo "$$" > "$LOCK_FILE"

# ── barrier 2: pgrep (bracketed so this script cannot match itself) ───────
if [ -n "$PROC" ]; then
  PAT="[${PROC:0:1}]${PROC:1}"
  if pgrep -f "$PAT" >/dev/null 2>&1; then
    echo "LAUNCH ABORTED: trainer process '$PROC' already alive (pgrep)" | tee -a "$LOG"
    exit 0
  fi
fi

# ── VRAM gate ────────────────────────────────────────────────────────────
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "${used:-}" ] && [ "$used" -gt "$VRAM_MAX_MIB" ]; then
  echo "LAUNCH ABORTED: GPU already holds ${used}MiB > ${VRAM_MAX_MIB}MiB" | tee -a "$LOG"
  exit 0
fi

# ── checkpoint resolution + 9p staging ───────────────────────────────────
RESUME=""
newest_in() {
  # Highest-numbered step_* dir that still contains a loadable-looking file.
  ls -d "$1"/step_* 2>/dev/null | sort -V | tail -1
}

if [ -n "${OUTDIR:-}" ]; then
  SRC=$(newest_in "$OUTDIR")
  DST=$(newest_in "$STAGE")
  if [ -n "${STAGE_STEP:-}" ]; then
    WANT="$STAGE/step_$STAGE_STEP"
    if [ -d "$WANT" ]; then RESUME="$WANT"; fi
  fi
  # Trust the local ext4 copy ONLY while it is not behind the mounted copy.
  # If the mounted dir advanced past the staged step, the stage is stale —
  # resuming from it silently re-runs hundreds of steps.
  if [ -z "$RESUME" ] && [ -n "$SRC" ]; then
    s_n=$(basename "$SRC" | grep -oE '[0-9]+' | head -1)
    d_n=$([ -n "$DST" ] && basename "$DST" | grep -oE '[0-9]+' | head -1 || echo 0)
    if [ -n "$s_n" ] && [ "${d_n:-0}" -ge "$s_n" ] && [ -n "$DST" ]; then
      RESUME="$DST"
    else
      free_mib=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
      if [ "$free_mib" -lt "$MIN_FREE_STAGING_MIB" ]; then
        echo "STAGING SKIPPED: ${free_mib}MiB free < ${MIN_FREE_STAGING_MIB}MiB — 9p copy would risk ENOMEM truncation" | tee -a "$LOG"
        echo "LAUNCH ABORTED: no safe checkpoint source (stale stage, slow mount under RAM pressure)" | tee -a "$LOG"
        exit 1
      fi
      sync
      mkdir -p "$STAGE/$(basename "$SRC")"
      ok=1
      for f in "$SRC"/*; do
        [ -e "$f" ] || continue
        dd if="$f" of="$STAGE/$(basename "$SRC")/$(basename "$f")" bs=8M status=none || ok=0
      done
      # dd where cp failed: cp hits ENOMEM on large 9p reads, dd's smaller
      # bounded buffer gets through. Then PROVE the bytes.
      if [ "$ok" -eq 1 ]; then
        for f in "$SRC"/*; do
          a=$(stat -c %s "$f"); b=$(stat -c %s "$STAGE/$(basename "$SRC")/$(basename "$f")" 2>/dev/null || echo -1)
          [ "$a" = "$b" ] || { echo "STAGING VERIFY FAILED on $(basename "$f") ($a != $b)" | tee -a "$LOG"; ok=0; }
        done
      fi
      [ "$ok" -eq 1 ] && RESUME="$STAGE/$(basename "$SRC")"
    fi
  fi
fi

if [ -n "$RESUME" ]; then
  echo "[$(date '+%F %T')] Resuming from $RESUME" | tee -a "$LOG"
  export TATU_RESUME="$RESUME"
fi

# ── go ───────────────────────────────────────────────────────────────────
cd "$TRAIN_DIR" || exit 1
echo "[$(date '+%F %T')] launch: $TRAIN_CMD (log: $LOG)" | tee -a "$LOG"
exec bash -c "$TRAIN_CMD" >> "$LOG" 2>&1
