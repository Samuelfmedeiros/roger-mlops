#!/usr/bin/env bash
# keepalive.sh — guest-side 24/7 relaunch loop for a resume-safe trainer.
#
# v2 logic learned the hard way: "finished" must be PARSED, never grepped.
# v1 grepped for a completion word and matched the launcher's own dry-run
# banner ("Dry-run complete. Ready for real training!") — the vigilia exited
# at step 200/2000 while the run was alive, and nobody noticed for hours.
#
# Usage:
#   LOG=/tmp/train/train.log TOTAL=2000 PROC=run_train ./keepalive.sh
# Installed as a systemd user unit (see deploy/keepalive.service) or under
# any supervisor. Ctrl-C friendly: it is a loop, not a daemon.

set -u

LOG="${LOG:?set LOG to the append-only training log}"
PROC="${PROC:-run_train}"
LAUNCHER="${LAUNCHER:?set LAUNCHER to the resume-safe launch script}"
INTERVAL="${INTERVAL:-600}"
RECHECK="${RECHECK:-300}"

# Bracket trick: prevents pgrep from matching this script's own cmdline.
PAT="[${PROC:0:1}]${PROC:1}"

echo "[keepalive] $(date '+%F %T') watching $LOG (proc=$PROC)"
while true; do
  if pgrep -f "$PAT" >/dev/null 2>&1; then
    sleep "$INTERVAL"
    continue
  fi

  # Dead — but is it DONE? Parse the last "Step X/N" and compare numbers.
  LAST=$(grep -aoE 'Step [0-9]+/[0-9]+' "$LOG" 2>/dev/null | tail -1)
  CUR=$(echo "$LAST"  | grep -oE '[0-9]+' | head -1)
  TOT=$(echo "$LAST"  | grep -oE '[0-9]+' | tail -1)
  if [ -n "${CUR:-}" ] && [ -n "${TOT:-}" ] && [ "$CUR" -ge "$TOT" ]; then
    echo "[keepalive] $(date '+%F %T') run finished (step $CUR/$TOT) — exiting"
    exit 0
  fi

  echo "[keepalive] $(date '+%F %T') trainer dead (last step: ${LAST:-none}) — relaunching"
  bash "$LAUNCHER"
  RC=$?
  echo "[keepalive] $(date '+%F %T') launcher rc=$RC — re-evaluating in ${RECHECK}s"
  sleep "$RECHECK"
done
