"""
checkers_scr -- drive Checkers' compressed checkpoints through SCR.

Checkers reduces *how many bytes* a checkpoint costs; SCR decides *where those
bytes live, how they survive a node failure, and when they reach the parallel
file system*.  This module is the (thin) glue: it owns the SCR output/restart
window and hands Checkers a directory to write into.

Why a Python shim is sufficient
-------------------------------
SCR's ``SCR_Route_file`` takes the *basename* of the logical file and prepends
the dataset's cache directory (scr.c:scr_route_file).  So every file a rank
routes for one dataset lands in the *same* directory, with its name preserved.
Checkers already writes a fixed, deterministic file set into a single
directory::

    <dir>/rank_%03d_<stream>.cch      one per compressed stream
    <dir>/rank_%03d_manifest.json     always written

Those two facts line up exactly: route one file to learn the directory, let
Checkers write its whole set there, then register what it actually wrote.  No
C++ change is needed.

Two rules the implementation depends on
---------------------------------------
1. ``route_file`` must always be given the *logical* (PFS-side) path.  SCR
   aborts if the path is not under ``SCR_PREFIX`` (scr.c:3285), and the routed
   cache path lives in /dev/shm, which is not.
2. Never route a file that will not be written.  ``scr_complete_output``
   stats every file in the filemap and marks the whole dataset invalid if one
   is unreadable.  Hence the probe uses the manifest, which Checkers writes
   unconditionally.

Call sequence (output)
----------------------
    SCR_Start_output(name)        collective
    SCR_Route_file(manifest)      local  -> tells us the cache directory
    checkers compress + write     local  (GPU codec, D2H, zstd, file write)
    SCR_Route_file(each file)     local  -> registers what was written
    SCR_Complete_output(valid)    collective

With ``blocking=False`` the window stays open across the gap between
``compress_and_save`` and ``checkpoint_join``; no other SCR call may happen in
between.  That is enforced here rather than left to discover at runtime.

If SCR is unavailable or ``enabled=False``, every entry point degrades to
plain Checkers writing under ``<prefix>/<name>``, so the same script runs
either way.
"""

from __future__ import annotations

import glob
import json
import os
import struct

import checkers_py

__all__ = [
    "init",
    "initialize_context",
    "compress_and_save",
    "checkpoint_join",
    "need_checkpoint",
    "should_exit",
    "have_restart",
    "restart_from_checkpoint",
    "finalize",
    "is_enabled",
    "last_routed_dir",
]

# Set by init(); None means "SCR not in use, fall back to plain Checkers".
_scr = None

_S = {
    "enabled": False,
    "prefix": None,      # SCR_PREFIX -- logical checkpoint root on the PFS
    "rank": 0,           # must match the rank Checkers puts in its filenames
    "open_name": None,   # dataset name of the currently open output window
    "routed_dir": None,  # where Checkers was told to write for that window
    "logical_dir": None, # <prefix>/<name>, the PFS-side name of that window
    "min_snr_db": None,  # optional validity gate, see compress_and_save()
    "singleton": False,  # one independent SCR instance per rank (no PMI)
    "expected_tensors": None,  # optional {stream: n} or int, checked by _validate
    "noscr_dir": None,   # SCR off: where the pending non-blocking save writes
}

# Defaults tuned for a node with /dev/shm as the only cache tier.  Every one of
# these can be overridden through init(**config) or an SCR config file.
#
# Two of these are not the SCR defaults, and both matter:
#   SCR_CACHE_BYPASS defaults to 1 in SCR 3.x, which writes straight through to
#     the parallel file system and skips the cache entirely -- that would make
#     the whole cache/redundancy tier a no-op.  0 turns caching on.
#   Backgrounding the flush needs BOTH of these -- they are different layers:
#     SCR_FLUSH_ASYNC (default 0) picks the async code path over scr_flush_sync
#       (scr.c: `if (scr_flush_async) scr_flush_async_start() else
#       scr_flush_sync()`).
#     SCR_FLUSH_TYPE (default "SYNC") is the AXL *transfer* used inside that
#       flush; PTHREAD copies on background threads.  Valid values: DEFAULT,
#       NATIVE, PTHREAD, SYNC, DATAWARP, BBAPI.
#   Note bypass mode ignores both and always calls scr_flush_sync, which is why
#   SCR_CACHE_BYPASS=0 has to come first for any of this to matter.
_DEFAULT_CONFIG = {
    "SCR_CACHE_BASE": "/dev/shm",
    "SCR_CACHE_BYPASS": "0",    # <- actually use the cache tier
    "SCR_CACHE_SIZE": "1",      # datasets kept in cache; /dev/shm is precious
    "SCR_COPY_TYPE": "XOR",     # erasure code across nodes
    "SCR_CRC_ON_COPY": "1",     # detect corruption while redundancy can fix it
    "SCR_FLUSH": "1",           # flush every checkpoint to the PFS
    "SCR_FLUSH_ASYNC": "1",     # ...off the critical path
    "SCR_FLUSH_TYPE": "PTHREAD",  # ...using background copy threads
}


# --------------------------------------------------------------------------- #
#  internals
# --------------------------------------------------------------------------- #
def _detect_job_id():
    """Find this job's id from the launcher's environment.

    SCR reads $FLUX_JOB_ID only when built with -DSCR_RESOURCE_MANAGER=FLUX
    (scr_env.c wraps each launcher in an #ifdef).  A build configured for
    SLURM -- which is what is installed here -- ignores Flux entirely and
    falls back to the literal string "defjobid", so every job on a node shares
    /dev/shm/<user>/scr.defjobid for both cache and control state.  Two
    concurrent jobs would then collide there.

    SCR_JOB_ID takes precedence over the compiled-in detection (scr.c:780), so
    setting it explicitly works regardless of how the library was built.
    """
    for var in ("FLUX_JOB_ID", "SLURM_JOBID", "SLURM_JOB_ID",
                "LSB_JOBID", "PBS_JOBID"):
        value = os.environ.get(var)
        if value:
            return value

    # Under `--launcher pdsh` the remote shell is clean: pdsh exports only what
    # is listed in .deepspeed_env, so none of the above survive and every run
    # would share one job id -- and therefore one /dev/shm/<user>/scr.<id>/.
    # Two concurrent jobs would then scribble on each other's control state,
    # and have_restart() could hand a fresh run the *previous* job's checkpoint.
    # MASTER_PORT is the per-run unique token every DeepSpeed launcher exports
    # to every process (launch.py), and two jobs cannot share one anyway
    # without torch.distributed itself colliding.
    port = os.environ.get("MASTER_PORT")
    if port:
        return "ds" + port

    return None


def _resolve_rank(model_engine):
    """Mirror api.cpp:extract_rank exactly.

    Checkers names its files from ``model_engine.global_rank``, falling back to
    $RANK, falling back to -1 which it then clamps to 0.  We must land on the
    same number or the probe path will not match what gets written.
    """
    rank = getattr(model_engine, "global_rank", None)
    if rank is None:
        env = os.environ.get("RANK")
        rank = int(env) if env is not None else -1
    rank = int(rank)
    return 0 if rank < 0 else rank


def _manifest_name():
    return "rank_{:03d}_manifest.json".format(_S["rank"])


def _rank_glob():
    return "rank_{:03d}_*".format(_S["rank"])


def _require_closed(op):
    if _S["open_name"] is not None:
        raise RuntimeError(
            "checkers_scr: cannot call {}() while the SCR output window for "
            "'{}' is still open -- call checkpoint_join() first".format(
                op, _S["open_name"]
            )
        )


def _route_probe(logical_dir):
    """Open-window helper: learn the directory SCR wants this dataset written to.

    Routes the manifest -- the one file Checkers always writes -- so the probe
    never leaves a phantom entry in the filemap.
    """
    probe_logical = os.path.join(logical_dir, _manifest_name())
    routed = _scr.route_file(probe_logical)
    if not routed:
        raise RuntimeError("checkers_scr: SCR_Route_file returned nothing for "
                           + probe_logical)
    return os.path.dirname(routed)


def _register_written(routed_dir, logical_dir):
    """Register every file this rank actually wrote, by its logical name.

    Returns the list of routed paths on disk.
    """
    written = sorted(glob.glob(os.path.join(routed_dir, _rank_glob())))
    for path in written:
        # Rule 1: route the logical name, not the routed one.
        _scr.route_file(os.path.join(logical_dir, os.path.basename(path)))
    return written


# .cch layout (include/checkers/CheckpointFormat.hpp): CchHeader is 88 bytes,
# each CchIndexEntry 432, its data/aux offset+size at byte 344.
_CCH_HDR = struct.Struct("<8I d 6Q")
_CCH_IDX = 432
_CCH_MAGIC = 0x31484343


def _cch_read(path):
    """-> (n_tensors, size the header + index imply).  Raises on a bad header."""
    with open(path, "rb") as f:
        raw = f.read(_CCH_HDR.size)
        if len(raw) != _CCH_HDR.size:
            raise ValueError("header truncated")
        h = _CCH_HDR.unpack(raw)
        magic, n_tensors, data_off = h[0], h[6], h[10]
        if magic != _CCH_MAGIC:
            raise ValueError("bad magic 0x%08x" % magic)
        idx = f.read(n_tensors * _CCH_IDX)
    if len(idx) != n_tensors * _CCH_IDX:
        raise ValueError("index truncated")
    end = 0
    for k in range(n_tensors):
        do, ds, ao, asz = struct.unpack_from("<4Q", idx, k * _CCH_IDX + 344)
        end = max(end, do + ds, ao + asz)
    return n_tensors, data_off + end


def _expected_for(stream):
    exp = _S["expected_tensors"]
    if isinstance(exp, dict):
        return exp.get(stream)
    return exp


def _validate(metrics, written):
    """Decide the per-rank ``valid`` flag handed to SCR_Complete_output.

    SCR ORs this across ranks: if any rank reports invalid, the dataset is
    rejected and SCR falls back to an older checkpoint.  That makes it the
    right place to enforce "the error bound was actually met".

    Since 2026-11-05 it also checks the files themselves (existence and
    non-zero size were not enough -- a short write and a stream missing
    tensors both passed):
      * the manifest is readable and lists every .cch this rank wrote;
      * each .cch is exactly as long as its own header + index imply;
      * each .cch's tensor count equals the manifest's n_tensors and
        n_expected (the tensors collected for that stream), and the
        caller's expected count if init(expected_tensors=...) gave one.
    """
    if metrics is None:
        return False, "compress_and_save returned no metrics"
    if not written:
        return False, "no checkpoint files were written"

    for path in written:
        if not os.path.isfile(path):
            return False, "missing file " + path
        if os.path.getsize(path) == 0:
            return False, "empty file " + path

    mpaths = [p for p in written if p.endswith("_manifest.json")]
    if len(mpaths) != 1:
        return False, "expected one manifest, found %d" % len(mpaths)
    try:
        with open(mpaths[0]) as f:
            streams = json.load(f)["streams"]
    except Exception as exc:
        return False, "unreadable manifest %s (%s)" % (mpaths[0], exc)
    cch = {os.path.basename(p): p for p in written if p.endswith(".cch")}
    listed = {v.get("file") for v in streams.values()}
    for extra in sorted(set(cch) - listed):
        return False, "%s is not in the manifest" % extra
    for name, st in sorted(streams.items()):
        path = cch.get(st.get("file"))
        if path is None:
            return False, "stream %s: file %s missing" % (name, st.get("file"))
        try:
            n_hdr, need = _cch_read(path)
        except Exception as exc:
            return False, "%s: unreadable header (%s)" % (path, exc)
        size = os.path.getsize(path)
        if size != need:
            return False, "%s is %d bytes, its header implies %d (%s)" % (
                path, size, need, "truncated" if size < need else "too long")
        for what, want in (("manifest n_tensors", st.get("n_tensors")),
                           ("manifest n_expected", st.get("n_expected")),
                           ("expected_tensors", _expected_for(name))):
            if want is not None and n_hdr != want:
                return False, "stream %s holds %d tensors, %s is %d" % (name, n_hdr, what, want)

    overall = metrics.get("overall", {})
    if overall.get("n_tensors", 0) == 0:
        return False, "no tensors compressed"

    floor = _S["min_snr_db"]
    if floor is not None and not overall.get("lossless", True):
        snr = overall.get("snr_db")
        if isinstance(snr, (int, float)) and snr < floor:
            return False, "snr {:.2f} dB below floor {:.2f} dB".format(snr, floor)

    return True, ""


def _check_noscr(metrics, out_dir):
    """SCR off: the same checks as _validate, but a failure raises (there is
    no SCR valid flag to hand it to)."""
    written = sorted(glob.glob(os.path.join(out_dir, _rank_glob())))
    valid, why = _validate(metrics, written)
    if not valid:
        raise RuntimeError("checkers_scr: checkpoint in %s failed validation on rank %d: %s"
                           % (out_dir, _S["rank"], why))
    metrics["validated"] = True
    return metrics


def _close_output(metrics):
    """Register written files, validate, and close the window.  Collective."""
    routed_dir = _S["routed_dir"]
    logical_dir = _S["logical_dir"]
    name = _S["open_name"]

    try:
        written = _register_written(routed_dir, logical_dir)
        valid, why = _validate(metrics, written)
        all_valid = _scr.complete_output(valid)
    finally:
        _S["open_name"] = None
        _S["routed_dir"] = None
        _S["logical_dir"] = None

    if metrics is not None:
        metrics["scr"] = {
            "name": name,
            "dir": routed_dir,
            "files": [os.path.basename(p) for p in written],
            "valid": valid,
            "all_ranks_valid": bool(all_valid),
            "reason": why,
        }
    return metrics


# --------------------------------------------------------------------------- #
#  lifecycle
# --------------------------------------------------------------------------- #
def init(prefix, enabled=True, min_snr_db=None, expected_tensors=None, **config):
    """Initialize SCR.  Collective over MPI_COMM_WORLD.

    Call once, after MPI and torch.distributed are both up and before the
    first checkpoint.

    prefix      logical checkpoint root on the parallel file system.  Becomes
                SCR_PREFIX; every checkpoint is named <prefix>/<name>/.
    enabled     False runs the plain Checkers path with no SCR involvement.
    min_snr_db  if set, a lossy checkpoint whose measured SNR falls below this
                is reported to SCR as invalid rather than silently kept.
    expected_tensors  optional tensor count per stream, an int (same for every
                stream, e.g. 291 for a LLaMA-3-8B ZeRO-3 shard) or a dict
                {stream: n}; a saved stream with any other count is invalid.
                The manifest's own n_expected is always checked.
    **config    extra SCR_* settings, e.g. SCR_COPY_TYPE="PARTNER".  These
                must be applied before SCR_Init, which is why they live here.
    """
    global _scr

    _S["prefix"] = os.path.abspath(prefix)
    _S["min_snr_db"] = min_snr_db
    _S["expected_tensors"] = expected_tensors

    if not enabled:
        _S["enabled"] = False
        return False

    try:
        import scr as _scr_mod
    except ImportError as exc:
        raise ImportError(
            "checkers_scr: could not import the SCR python module ({}).  Add "
            "<scr-install>/share/scr/python to PYTHONPATH, and make sure "
            "mpi4py is imported before this call.".format(exc)
        )

    _scr = _scr_mod

    settings = dict(_DEFAULT_CONFIG)
    job_id = _detect_job_id()

    # ---- singleton mode -------------------------------------------------- #
    # A launcher with no PMI server (pdsh, plain ssh) gives every process its
    # own 1-rank MPI_COMM_WORLD.  SCR still works there -- it just forces the
    # copy type to SINGLE, because parity needs >= 2 ranks in a set -- but the
    # N processes would collide, since each believes it is rank 0:
    #   * /dev/shm/<user>/scr.<SCR_JOB_ID>/   control + cache state
    #   * <prefix>/.scr/index.scr, flush.scr  dataset bookkeeping
    # Giving each rank its own job id AND its own prefix subtree isolates both.
    # The cost is explicit and worth restating: NO cross-node redundancy.
    from mpi4py import MPI

    mpi_size = MPI.COMM_WORLD.Get_size()
    app_world = int(os.environ.get("WORLD_SIZE", "1"))
    app_rank = int(os.environ.get("RANK", "0"))
    _S["singleton"] = (mpi_size == 1 and app_world > 1)

    if _S["singleton"]:
        _S["prefix"] = os.path.join(_S["prefix"], "scr_rank_%03d" % app_rank)
        job_id = "%s_r%03d" % (job_id or "nojob", app_rank)
        if app_rank == 0:
            print("[checkers_scr] SINGLETON MODE: MPI world is 1 rank but the "
                  "app has %d. Each rank runs its own independent SCR instance "
                  "under <prefix>/scr_rank_NNN/. Caching and async flush still "
                  "work; XOR redundancy is downgraded to SINGLE (no cross-node "
                  "parity) because that needs a real multi-rank MPI world."
                  % app_world, flush=True)

    settings["SCR_PREFIX"] = _S["prefix"]
    if job_id:
        settings["SCR_JOB_ID"] = job_id

    settings.update({k: str(v) for k, v in config.items()})
    for key, value in settings.items():
        _scr.config("{}={}".format(key, value))

    _scr.init()
    _S["enabled"] = True
    return True


def initialize_context(model_engine, optimizer=None, **kwargs):
    """Passthrough to checkers_py.initialize_context.

    Also captures this rank's id, because compress_and_save() needs to predict
    the filenames Checkers will produce and model_engine is not passed there.
    """
    _S["rank"] = _resolve_rank(model_engine)
    return checkers_py.initialize_context(model_engine, optimizer, **kwargs)


def finalize():
    """Shut down Checkers and SCR.  Collective."""
    if _S["open_name"] is not None:
        # A window left open would hang the next collective; close it as failed.
        try:
            metrics = checkers_py.checkpoint_join()
        except Exception:
            metrics = None
        _close_output(metrics)

    try:
        # Tolerant: shutdown() is a no-op if initialize_context never ran, but
        # a teardown error must not stop us from closing SCR.
        checkers_py.shutdown()
    except Exception:
        pass
    finally:
        if _S["enabled"]:
            _scr.finalize()
            _S["enabled"] = False


def is_enabled():
    return _S["enabled"]


def last_routed_dir():
    """Directory Checkers was last told to write into (useful for logging)."""
    return _S["routed_dir"]


# --------------------------------------------------------------------------- #
#  output
# --------------------------------------------------------------------------- #
def compress_and_save(name, error_bound, targets="all", blocking=True,
                      flags=None, allow_lossless=False):
    """Compress this rank's shard and hand the files to SCR.

    name         dataset name, e.g. "step_500".  Replaces the old output_dir;
                 the checkpoint is <SCR_PREFIX>/<name>/ on the file system.
    error_bound  float, or per-stream dict -- passed straight to Checkers.
    targets      "all" or a stream list -- passed straight to Checkers.
    blocking     True  -> compress, write, and close the SCR window here.
                 False -> return after the GPU snapshot; the SCR window stays
                          open and you MUST call checkpoint_join() to close it.
    flags        SCR dataset flags; defaults to FLAG_CHECKPOINT.
    allow_lossless  passed straight to Checkers: True lets an explicit 0
                 bound store that stream lossless (raw + zstd).

    Returns the Checkers metrics dict, with an extra "scr" entry once the
    window has closed.
    """
    if not _S["enabled"]:
        out_dir = os.path.join(_S["prefix"], name)
        os.makedirs(out_dir, exist_ok=True)
        metrics = checkers_py.compress_and_save(
            out_dir, error_bound=error_bound, targets=targets,
            blocking=blocking, allow_lossless=allow_lossless,
        )
        if not blocking:
            _S["noscr_dir"] = out_dir      # checked by checkpoint_join()
            return metrics
        return _check_noscr(metrics, out_dir)

    _require_closed("compress_and_save")

    if flags is None:
        flags = _scr.FLAG_CHECKPOINT

    logical_dir = os.path.join(_S["prefix"], name)

    _scr.start_output(name, flags)          # collective
    _S["open_name"] = name
    _S["logical_dir"] = logical_dir

    try:
        routed_dir = _route_probe(logical_dir)
        _S["routed_dir"] = routed_dir
        # SCR has already created this directory; Checkers' create_directories
        # call on it is a harmless no-op.
        metrics = checkers_py.compress_and_save(
            routed_dir, error_bound=error_bound, targets=targets,
            blocking=blocking, allow_lossless=allow_lossless,
        )
    except Exception:
        # Close the window as failed so the collective stays balanced and the
        # next checkpoint is not blocked by a half-open dataset.
        _S["routed_dir"] = _S["routed_dir"] or logical_dir
        _close_output(None)
        raise

    if blocking:
        return _close_output(metrics)

    # Window deliberately left open -- checkpoint_join() closes it.
    return metrics


def checkpoint_join():
    """Wait for a non-blocking compress_and_save, then close the SCR window.

    Collective, because SCR_Complete_output is.
    """
    if not _S["enabled"]:
        metrics = checkers_py.checkpoint_join()
        out_dir, _S["noscr_dir"] = _S["noscr_dir"], None
        return _check_noscr(metrics, out_dir) if out_dir else metrics

    if _S["open_name"] is None:
        raise RuntimeError("checkers_scr: checkpoint_join() with no open "
                           "checkpoint -- did compress_and_save run?")

    try:
        metrics = checkers_py.checkpoint_join()
    except Exception:
        _close_output(None)
        raise

    return _close_output(metrics)


# --------------------------------------------------------------------------- #
#  policy
# --------------------------------------------------------------------------- #
def need_checkpoint():
    """Ask SCR whether it is time to checkpoint.  Collective."""
    if not _S["enabled"]:
        return True
    _require_closed("need_checkpoint")
    return _scr.need_checkpoint()


def should_exit():
    """Ask SCR whether the job should stop early (allocation ending, scr_halt).

    Collective.
    """
    if not _S["enabled"]:
        return False
    _require_closed("should_exit")
    return _scr.should_exit()


# --------------------------------------------------------------------------- #
#  restart
# --------------------------------------------------------------------------- #
def have_restart():
    """Name of a checkpoint SCR has staged for restart, or None.  Collective."""
    if not _S["enabled"]:
        return None
    _require_closed("have_restart")
    return _scr.have_restart()


def restart_from_checkpoint(model_engine, optimizer=None, targets=None,
                            name=None):
    """Reload this rank's shard from an SCR-managed checkpoint.

    With SCR enabled, ``name`` is ignored: SCR decides which checkpoint to
    restart from and may serve it out of cache rather than the file system.
    Without SCR, ``name`` selects <prefix>/<name>/ directly.

    Returns the Checkers restart dict, or None if SCR has nothing to restart
    from.
    """
    _S["rank"] = _resolve_rank(model_engine)

    if not _S["enabled"]:
        if name is None:
            raise ValueError("checkers_scr: name= is required when SCR is off")
        return checkers_py.restart_from_checkpoint(
            model_engine, os.path.join(_S["prefix"], name),
            optimizer=optimizer, targets=targets,
        )

    _require_closed("restart_from_checkpoint")

    loaded = _scr.have_restart()
    if not loaded:
        return None

    _scr.start_restart()                    # collective
    try:
        logical_dir = os.path.join(_S["prefix"], loaded)
        routed = _scr.route_file(os.path.join(logical_dir, _manifest_name()))
        input_dir = os.path.dirname(routed) if routed else logical_dir

        out = checkers_py.restart_from_checkpoint(
            model_engine, input_dir, optimizer=optimizer, targets=targets,
        )
        valid = bool(out) and not out.get("missing")
    except Exception:
        _scr.complete_restart(False)
        raise

    all_valid = _scr.complete_restart(valid)   # collective

    if out is not None:
        out["scr"] = {
            "name": loaded,
            "dir": input_dir,
            "valid": valid,
            "all_ranks_valid": bool(all_valid),
        }
    return out
