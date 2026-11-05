#!/usr/bin/env python3
"""
test_commit_2rank.py -- the all-ranks commit of an SCR-off save, on 2 GPUs.

    RANK=r LOCAL_RANK=r WORLD_SIZE=2 MASTER_ADDR=... MASTER_PORT=... \
        python test_commit_2rank.py --out DIR          (one process per rank)

  1  both ranks save "ok"          -> both validated, ok/ holds both ranks' files,
                                      no ok.inprogress
  2  rank 1 fails (injected) "bad" -> BOTH ranks raise (rank 0: "NOT committed"),
                                      neither hangs, bad/ does not exist,
                                      bad.inprogress holds rank 0's files only
  3  retry "bad" without the fault -> committed on both

Each rank prints its own verdict; run_save_tests.sh requires both to PASS.
"""
import argparse, glob, os, shutil, sys

os.environ.setdefault("DS_ACCELERATOR", "cuda")
import torch
import deepspeed

sys.path.insert(0, "/usr/WS1/kogiou1/LLM_work/Checkers/build")
sys.path.insert(0, "/usr/WS1/kogiou1/LLM_work/Checkers/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers_py
import checkers_scr as cs
from test_restart_roundtrip import Tiny

EPS = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.manual_seed(0)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    cfg = {"train_micro_batch_size_per_gpu": 4, "gradient_accumulation_steps": 1,
           "bf16": {"enabled": True},
           "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
           "zero_optimization": {"stage": 3}}
    model = Tiny()
    engine, opt, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=cfg)
    g = torch.Generator(device="cpu").manual_seed(1 + rank)
    for _ in range(2):
        x = torch.randn(4, 512, generator=g).to(engine.device, torch.bfloat16)
        engine.backward(engine(x).float().pow(2).mean()); engine.step()
    if rank == 0:
        shutil.rmtree(a.out, ignore_errors=True); os.makedirs(a.out)
    torch.distributed.barrier()
    cs.init(a.out, enabled=False)
    cs.initialize_context(engine, opt, stage_targets="all", group_size=256, compute_streams=4)

    rows, fails = [], []
    def check(ok, what):
        rows.append(("PASS" if ok else "FAIL") + "  " + what)
        if not ok:
            fails.append(what)
    def n_files(d, r):
        return len(glob.glob(os.path.join(a.out, d, "rank_%03d_*" % r)))

    m = cs.compress_and_save("ok", EPS, blocking=True)
    torch.distributed.barrier()
    check(m.get("validated") is True and n_files("ok", 0) == 5 and n_files("ok", 1) == 5 and
          not os.path.exists(os.path.join(a.out, "ok.inprogress")),
          f"1 both ranks commit ok/: files r0={n_files('ok', 0)} r1={n_files('ok', 1)}")

    if rank == 1:
        os.environ["CHECKERS_TEST_FAIL_TENSOR"] = "l3.weight"
    msg = None
    try:
        cs.compress_and_save("bad", EPS, blocking=True)
    except Exception as e:
        msg = str(e)
    os.environ.pop("CHECKERS_TEST_FAIL_TENSOR", None)
    torch.distributed.barrier()
    want = "NOT committed" if rank == 0 else "save FAILED"
    check(msg is not None and want in msg, f"2 rank {rank} raised with {want!r}: {(msg or 'no exception')[:150]!r}")
    check(not os.path.exists(os.path.join(a.out, "bad")) and n_files("bad.inprogress", 0) == 5
          and n_files("bad.inprogress", 1) == 0,
          f"2 bad/ not committed; bad.inprogress files r0={n_files('bad.inprogress', 0)} "
          f"r1={n_files('bad.inprogress', 1)}")

    m = cs.compress_and_save("bad", EPS, blocking=True)
    torch.distributed.barrier()
    check(m.get("validated") is True and n_files("bad", 0) == 5 and n_files("bad", 1) == 5,
          f"3 retry commits bad/: files r0={n_files('bad', 0)} r1={n_files('bad', 1)}")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\n=== 2-rank commit, rank {rank}: {verdict} ({len(rows) - len(fails)}/{len(rows)} checks)")
    for r in rows:
        print("  " + r)
    checkers_py.shutdown()
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
