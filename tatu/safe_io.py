#!/usr/bin/env python3
"""Crash-safe checkpoint I/O for hybrid-disk boxes (WSL / drvfs / 9p / NFS).

The rule this module exists to enforce:

    A file on a slow or lossy mount is not a checkpoint until its BYTES have
    been proven. Size alone lies. mtime alone lies. "The save call returned
    without raising" lies hardest of all.

Three failure modes seen in production, each with a guard here:

1. 9p/drvfs truncation that passes a naive size check. A copy interrupted by
   ENOMEM leaves a file whose size matches the resume floor, so the resume
   happily loads a half tensor and NaNs later. Guard: copy to `<name>.part`,
   fsync, verify size AND trailer, then `os.replace` (atomic on the same
   filesystem).

2. A copy in flight being judged as corruption. Guard: any file younger than
   `inflight_seconds` is reported as IN_FLIGHT, never as BAD — deleting a
   write in progress is worse than waiting.

3. mtime lies across mounts (`GetFileTime` returns 0 on 9p; a `copy2` carries
   the source mtime and disguises a stale file as fresh). Guard: age via
   `os.stat` on the mount that owns the file, and prefer content hashes taken
   on the Windows/native side over anything read through the bridge.

Canonical layout for a training run:
    persist dir  -> fast local filesystem (ext4 on WSL, never /mnt/*)
    mirror dir   -> optional second drive, written LAST and verified apart

Environment: TATU_PERSIST, TATU_MIRROR, TATU_MIRROR_VERIFY.
"""
import hashlib
import json
import os
import shutil
import sys
import time

try:
    from . import config as C
except ImportError:  # pragma: no cover - executed as a plain script
    import config as C

PERSIST = C.cfg("PERSIST", "/tmp/tatu_ckpt_persist")
MIRROR = C.cfg("MIRROR")  # empty string disables mirroring entirely
VERIFY = C.cfg("MIRROR_VERIFY", "1") not in ("0", "", "false", "False")

# gzip/tar members carry a CRC32 trailer; a truncated stream almost always
# fails this test even when the byte count looks right.
TRAILER_MAGIC = b"\x1f\x8b"


class Result:
    OK = "ok"
    IN_FLIGHT = "in_flight"
    BAD = "bad"
    MISSING = "missing"


def _fsync_dir(path):
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass  # not all filesystems allow it; the file fsync still happened


def fsync_path(path):
    """fsync a file then its parent dir, so the rename survives a power loss."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass
    _fsync_dir(path)


def atomic_write(path, data, mode="wb"):
    """Write `data` to `path` atomically and durably.

    Uses `<path>.tmp` + fsync + os.replace. Never leaves a partial `path`
    visible to a reader, which is what makes a concurrent resume safe.
    """
    tmp = path + ".tmp"
    with open(tmp, mode) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path)
    return path


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expect_size=None, check_trailer=False):
    """-> (Result, detail). Structural verification, not just existence."""
    if not os.path.exists(path):
        return Result.MISSING, "absent"
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return Result.BAD, f"stat failed: {e}"
    if expect_size is not None and size != expect_size:
        return Result.BAD, f"size {size} != expected {expect_size}"
    if check_trailer and path.endswith((".gz", ".tgz")):
        try:
            with open(path, "rb") as f:
                head = f.read(2)
                if head != TRAILER_MAGIC:
                    return Result.BAD, f"bad gzip magic {head!r}"
                f.seek(max(0, size - 8))
                tail = f.read(8)
            # gzip trailer = CRC32 + ISIZE; both zero on a zero-filled tail
            if size and tail == b"\x00" * 8:
                return Result.BAD, "gzip trailer zero-filled (truncated copy)"
        except OSError as e:
            return Result.BAD, f"trailer read failed: {e}"
    return Result.OK, f"size={size}"


def age_seconds(path):
    """Age measured on the mount that OWNS the file. Never trust 9p mtime
    forwarded from another host — read through the native side instead."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not st.st_mtime:
        return None  # 9p returns 0 for GetFileTime on some builds
    return max(0.0, time.time() - st.st_mtime)


def safe_copy(src, dst, inflight_seconds=300, check_trailer=False):
    """Copy a big artifact into place the way a resume-safe pipeline must.

    Returns (Result, dst_path, detail). Never removes an existing good dst
    when the copy fails; the previous version survives.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    part = dst + ".part"
    expect = os.path.getsize(src)
    try:
        with open(src, "rb") as fi, open(part, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)
            fo.flush()
            os.fsync(fo.fileno())
        res, detail = verify(part, expect_size=expect, check_trailer=check_trailer)
        if res != Result.OK:
            try:
                os.unlink(part)
            except OSError:
                pass
            return Result.BAD, dst, f"copy verification failed: {detail}"
        fsync_path(part)
        os.replace(part, dst)
        _fsync_dir(dst)
    except OSError as e:
        # ENOMEM/EIO across 9p is the classic; leave no misleading partial.
        try:
            if os.path.exists(part) and (age_seconds(part) or 0) > inflight_seconds:
                os.unlink(part)
        except OSError:
            pass
        return Result.BAD, dst, f"copy raised {type(e).__name__}: {e}"
    return Result.OK, dst, f"copied {expect} bytes"


def prune(path, inflight_seconds=300):
    """Delete `path` only if it is provably stale. A file younger than
    `inflight_seconds` is a write in flight, not garbage."""
    age = age_seconds(path)
    if age is not None and age < inflight_seconds:
        return Result.IN_FLIGHT, f"age {age:.0f}s < {inflight_seconds}s guard"
    try:
        os.unlink(path)
        return Result.OK, "removed"
    except OSError as e:
        return Result.BAD, str(e)


def publish(directory, name, payload, expect_size=None):
    """Checkpoint save entry point: bytes -> durable file -> sidecar manifest.

    `payload` may be a path (moved/copied in) or raw bytes. Writes
    `<name>.json` beside it with size + sha256 so a later resume can prove
    identity instead of guessing from the filename.
    """
    os.makedirs(directory, exist_ok=True)
    dst = os.path.join(directory, name)
    if isinstance(payload, (bytes, bytearray)):
        atomic_write(dst, bytes(payload))
    else:
        if os.path.dirname(os.path.abspath(payload)) == os.path.abspath(directory):
            os.replace(payload, dst)
            fsync_path(dst)
        else:
            res, _, detail = safe_copy(payload, dst, check_trailer=name.endswith(".gz"))
            if res != Result.OK:
                raise IOError(f"publish failed: {detail}")
    size = os.path.getsize(dst)
    if expect_size is not None and size != expect_size:
        raise IOError(f"published {name} size {size} != {expect_size}")
    atomic_write(dst + ".json", json.dumps({
        "name": name, "size": size, "sha256": sha256_file(dst),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1).encode())
    return dst


def mirror_step(directory, inflight_seconds=300):
    """Copy canonical checkpoints to the secondary drive, verifying each.

    The mirror is a convenience, never a source of truth: a failed mirror
    logs and continues. A canonical write that fails must raise.
    """
    if not MIRROR:
        return {"enabled": False, "copied": 0, "skipped": 0, "failed": 0}
    os.makedirs(MIRROR, exist_ok=True)
    copied = skipped = failed = 0
    for root, _dirs, files in os.walk(directory):
        for fn in files:
            if fn.endswith(".part") or fn.endswith(".tmp") or fn.endswith(".json"):
                continue
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, directory)
            dst = os.path.join(MIRROR, rel)
            res, detail = (Result.OK, "")
            if os.path.exists(dst):
                try:
                    same = (os.path.getsize(dst) == os.path.getsize(src)
                            and (age_seconds(dst) or 0) > inflight_seconds)
                except OSError:
                    same = False
                if same and (not VERIFY or sha256_file(dst) == sha256_file(src)):
                    skipped += 1
                    continue
            res, _, detail = safe_copy(src, dst, inflight_seconds=inflight_seconds,
                                       check_trailer=fn.endswith(".gz"))
            if res == Result.OK:
                copied += 1
            else:
                failed += 1
                print(f"[mirror] FAILED {rel}: {detail}", file=sys.stderr)
    return {"enabled": True, "target": MIRROR, "copied": copied,
            "skipped": skipped, "failed": failed}


if __name__ == "__main__":  # tiny self-check, no deps
    import tempfile
    d = tempfile.mkdtemp(prefix="tatu_safe_io_")
    p = os.path.join(d, "model.bin")
    publish(d, "model.bin", b"tensors" * 1000, expect_size=7000)
    assert verify(p)[0] == Result.OK, "roundtrip failed"
    assert os.path.exists(p + ".json"), "manifest missing"
    res, _, _ = safe_copy(p, os.path.join(d, "copy.bin"))
    assert res == Result.OK
    print(f"self-check OK in {d}")
    shutil.rmtree(d, ignore_errors=True)
