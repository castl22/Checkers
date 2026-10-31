"""
Validate the SCR plumbing before wiring it into a training run.

This exercises everything in the integration *except* Checkers itself: MPI and
torch.distributed coexisting in one process, SCR_Init, the route/write/complete
cycle, and the flush to the parallel file system.  It writes dummy files using
Checkers' exact naming convention, so if this passes, the only thing left to go
wrong in the real thing is the compression itself.

Run it the way you run training -- under the same launcher, not torchrun:

    flux run -N2 -n8 python scr_smoke_test.py --prefix /p/lustre5/<you>/scr_test

What it proves, in order:
  1. mpi4py and torch.distributed agree on rank/world (the #1 integration risk)
  2. SCR initializes and picks up /dev/shm as its cache
  3. route_file returns a cache path, and the basename survives routing
  4. complete_output accepts the dataset
  5. the dataset reaches <prefix>/<name>/ on the file system
"""

import argparse
import os
import sys

# MPI must come up before SCR.  Importing mpi4py.MPI calls MPI_Init.
from mpi4py import MPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="SCR_PREFIX -- a directory on the parallel file system")
    ap.add_argument("--name", default="smoke_step_1")
    ap.add_argument("--cache-base", default="/dev/shm")
    ap.add_argument("--bytes", type=int, default=4 << 20,
                    help="dummy payload size per stream per rank")
    ap.add_argument("--skip-torch", action="store_true",
                    help="skip the torch.distributed rank-agreement check")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    def log(msg):
        print("[rank {}] {}".format(mpi_rank, msg), flush=True)

    # ---- 1. rank agreement --------------------------------------------- #
    # Checkers names files from model_engine.global_rank / $RANK, while SCR
    # works off the MPI rank.  If these disagree, ranks silently route each
    # other's files.  This is the check that matters most.
    env_rank = os.environ.get("RANK")
    if env_rank is not None and int(env_rank) != mpi_rank:
        log("FAIL: $RANK={} but MPI rank={}".format(env_rank, mpi_rank))
        comm.Abort(1)

    if not args.skip_torch:
        import torch
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("RANK", str(mpi_rank))
            os.environ.setdefault("WORLD_SIZE", str(mpi_size))
            os.environ.setdefault("MASTER_ADDR",
                                  comm.bcast(MPI.Get_processor_name(), root=0))
            os.environ.setdefault("MASTER_PORT", "29512")
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend,
                                    rank=mpi_rank, world_size=mpi_size)
        if dist.get_rank() != mpi_rank or dist.get_world_size() != mpi_size:
            log("FAIL: torch rank {}/{} != MPI rank {}/{}".format(
                dist.get_rank(), dist.get_world_size(), mpi_rank, mpi_size))
            comm.Abort(1)
        if mpi_rank == 0:
            log("OK 1: torch.distributed and MPI agree ({} ranks)".format(mpi_size))
    elif mpi_rank == 0:
        log("OK 1: skipped torch check")

    # ---- 2. SCR init ---------------------------------------------------- #
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # checkers_scr imports checkers_py, which lives in the build tree.
    build_dir = os.path.join(os.path.dirname(here), "build")
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import checkers_scr  # noqa: E402  (must follow the sys.path tweaks)

    # Import the raw module too, so we can report what SCR decided.
    checkers_scr.init(prefix=args.prefix, SCR_CACHE_BASE=args.cache_base,
                      SCR_DEBUG="1")
    if mpi_rank == 0:
        log("OK 2: SCR initialized, prefix={} cache={}".format(
            args.prefix, args.cache_base))

    import scr  # noqa: E402

    # ---- 3/4. route, write, complete ------------------------------------ #
    # Mimic Checkers' file set exactly: one .cch per stream plus a manifest.
    streams = ["model_state", "master_weights", "exp_avg", "exp_avg_sq"]
    logical_dir = os.path.join(os.path.abspath(args.prefix), args.name)
    manifest = "rank_{:03d}_manifest.json".format(mpi_rank)

    scr.start_output(args.name, scr.FLAG_CHECKPOINT)

    routed_manifest = scr.route_file(os.path.join(logical_dir, manifest))
    cache_dir = os.path.dirname(routed_manifest)

    if os.path.basename(routed_manifest) != manifest:
        log("FAIL: routing changed the basename: {}".format(routed_manifest))
        comm.Abort(1)
    if mpi_rank == 0:
        log("OK 3: routed into {}".format(cache_dir))
        # Distinguish cache mode from bypass mode.  SCR_CACHE_BYPASS defaults
        # to 1, which routes straight to the prefix and silently skips the
        # cache tier -- the checkpoint still works, but every benefit of
        # multi-level checkpointing is gone.  Catch that here rather than
        # wonder later why /dev/shm is empty.
        if cache_dir.startswith(os.path.abspath(args.prefix)):
            log("WARN 3b: BYPASS MODE -- routed to the parallel file system, "
                "not the cache.  Set SCR_CACHE_BYPASS=0 to use {}.".format(
                    args.cache_base))
        elif cache_dir.startswith(args.cache_base):
            log("OK 3b: cache mode active, staging through {}".format(
                args.cache_base))
        else:
            log("NOTE 3b: routed somewhere unexpected: {}".format(cache_dir))

    payload = b"\xa5" * args.bytes
    written = []
    with open(routed_manifest, "w") as fh:
        fh.write('{"rank": %d, "smoke": true}\n' % mpi_rank)
    written.append(routed_manifest)

    for stream in streams:
        base = "rank_{:03d}_{}.cch".format(mpi_rank, stream)
        routed = scr.route_file(os.path.join(logical_dir, base))
        with open(routed, "wb") as fh:
            fh.write(payload)
        written.append(routed)

    all_valid = scr.complete_output(True)
    if not all_valid:
        log("FAIL: complete_output reported some rank invalid")
        comm.Abort(1)
    if mpi_rank == 0:
        log("OK 4: complete_output accepted {} files/rank".format(len(written)))

    # ---- 5. did it reach the file system? -------------------------------- #
    # With SCR_FLUSH_ASYNC the flush may still be in flight; finalize drains it.
    checkers_scr.finalize()
    comm.Barrier()

    if mpi_rank == 0:
        if os.path.isdir(logical_dir):
            found = sorted(os.listdir(logical_dir))
            log("OK 5: {} holds {} files, e.g. {}".format(
                logical_dir, len(found), found[:3]))
        else:
            log("NOTE: {} not present yet -- check SCR_FLUSH settings or the "
                "scr.dataset index under the prefix".format(logical_dir))
        log("smoke test complete")


if __name__ == "__main__":
    main()
