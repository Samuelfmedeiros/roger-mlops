"""Shared configuration for the Roger toolbelt.

Every path is derived from the environment — the code carries no personal
usernames, drive letters or machine names. Set the ROGER_* variables (or a
.env file read by your supervisor) and the whole fleet follows.

Environment contract
--------------------
ROGER_HOME       root for IPC spool/state      (default: ~/.roger)
ROGER_TRAIN_LOG  append-only training log      (default: $ROGER_HOME/train.log)
ROGER_PERSIST    canonical ext4 checkpoint dir (default: /tmp/roger_ckpt_persist)
ROGER_MIRROR     optional second-drive mirror  (default: empty = disabled)
ROGER_PROC       cmdline pattern of the trainer process (default: run_train)
ROGER_END_BANNER regex marking legitimate completion (default: CONCLU|DONE)
ROGER_TMP        scratch/state dir for watchdogs (default: system temp)
"""
import os
import sys
import tempfile


def _home() -> str:
    return os.environ.get("ROGER_HOME") or os.path.expanduser("~/.roger")


def cfg(name: str, default: str = "") -> str:
    v = os.environ.get("ROGER_" + name)
    if v:
        return v
    if name == "ROOT":
        return _home()
    if name == "LOG":
        return os.path.join(_home(), "train.log")
    if name == "PERSIST":
        return "/tmp/roger_ckpt_persist"
    if name == "PROC":
        return "run_train"
    if name == "END_BANNER":
        return r"CONCLU|DONE"
    if name == "PYSPY":
        return os.path.expanduser("~/.local/bin/py-spy")
    if name == "TMP":
        return tempfile.gettempdir()
    return default


# Windows side of the fleet writes state/log files; POSIX side reads the log.
_IS_WIN = os.name == "nt"


def tmp_file(name: str) -> str:
    """Path for a watchdog's local state/log file (system temp on either side)."""
    return os.path.join(cfg("TMP") or tempfile.gettempdir(), name)


def wsl_cmd(*argv: str) -> list:
    """Build a `wsl.exe -e bash -c ...` invocation that is locale-proof.

    On pt-BR (or any non-English) Windows, wsl.exe may emit *translated*
    error strings — sometimes even with rc=0 — instead of WSAETIMEDOUT.
    Never match on the error text; match on the sentinel (see cfg users).
    """
    cmd = " ".join(argv) if len(argv) > 1 else argv[0]
    if _IS_WIN:
        return ["wsl", "-e", "bash", "-c", cmd]
    return ["bash", "-c", cmd]
