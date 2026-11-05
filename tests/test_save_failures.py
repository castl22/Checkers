#!/usr/bin/env python3
"""
test_save_failures.py -- a save must fail loudly, never write a short or
incomplete checkpoint and report success (bugs 1-2, 5, 6; 2026-11-05).

    python test_save_failures.py --out DIR

Same tiny one-GPU ZeRO-3 model as test_restart_roundtrip.py; saves go through
checkers_scr with SCR off, the way the experiments run.  Cases:

  A  normal blocking save         -> returns validated=True; every manifest
                                     stream has n_expected == n_tensors
  B  one tensor fails (injected)  -> compress_and_save raises "save FAILED",
                                     nothing written for this rank
  C  same, non-blocking           -> checkpoint_join() raises
  D  a .cch write fails (ENOSPC)  -> raises "write failed" (file is a symlink
                                     to /dev/full)
  E  next save after a failure    -> works (the error does not stick)
  F  _validate on copies of A     -> a truncated .cch, a stream with an index
                                     entry dropped, and a stray .cch are all
                                     invalid; an exact copy is valid
  G  restore of A                 -> complete (the writer changes kept the
                                     format readable)
  H  format of A                  -> every .cch is v3 (header | data | index),
                                     no tensor uses zstd (lossy: BITPACK |
                                     RAW_SIDE; raw: RAW_PASSTHROUGH)
  I  restore of a v2 checkpoint   -> complete (--v2-dir, written by the
                                     previous writer), if given
  J  retry of B without the fault -> commits: B exists, B.inprogress is gone
  K  save to an existing name     -> refuses; the existing checkpoint is untouched
  (B, C, D: the failed save never appears under its real name, only <name>.inprogress)

Exit status 0 = PASS, 1 = FAIL.
"""
import argparse, glob, json, os, shutil, struct, sys

os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 2000))
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
INJ = "CHECKERS_TEST_FAIL_TENSOR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--v2-dir", default=None, help="a v2 checkpoint of this model (case I)")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)

    cfg = {"train_micro_batch_size_per_gpu": 4, "gradient_accumulation_steps": 1,
           "bf16": {"enabled": True},
           "optimizer": {"type": "AdamW", "params": {"lr": 1e-3, "betas": [0.9, 0.999],
                                                     "eps": 1e-8, "weight_decay": 0.1}},
           "zero_optimization": {"stage": 3}}
    model = Tiny()
    engine, opt, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                             config=cfg)
    g = torch.Generator(device="cpu").manual_seed(1)
    for _ in range(2):
        x = torch.randn(4, 512, generator=g).to(engine.device, torch.bfloat16)
        loss = engine(x).float().pow(2).mean()
        engine.backward(loss); engine.step()

    cs.init(a.out, enabled=False)
    cs.initialize_context(engine, opt, stage_targets="all", group_size=256, zstd_level=3,
                          compute_streams=4, measure_snr=False)

    rows, fails = [], []
    def check(ok, what):
        rows.append(("PASS" if ok else "FAIL") + "  " + what)
        if not ok:
            fails.append(what)
    def files(name):
        return sorted(glob.glob(os.path.join(a.out, name, "rank_000_*")))
    def expect_raise(what, fn, must):
        try:
            fn()
        except Exception as e:
            msg = str(e)
            check(must in msg, f"{what}: raised, message has {must!r}: {msg[:160]!r}")
            return msg
        check(False, f"{what}: did NOT raise")
        return None

    # A
    mA = cs.compress_and_save("A", EPS, blocking=True)
    man = json.load(open(os.path.join(a.out, "A", "rank_000_manifest.json")))["streams"]
    check(mA.get("validated") is True, f"A normal save: validated={mA.get('validated')}")
    check(len(man) == 4 and all(s.get("n_expected") == s["n_tensors"] == 4 for s in man.values()),
          "A manifest n_expected == n_tensors == 4 for all 4 streams: "
          + str({k: (v["n_tensors"], v.get("n_expected")) for k, v in man.items()}))

    # B
    os.environ[INJ] = "l2.weight"
    expect_raise("B injected failure, blocking", lambda: cs.compress_and_save("B", EPS, blocking=True),
                 "save FAILED")
    check(files("B.inprogress") == [], f"B nothing written for rank 0: {[os.path.basename(f) for f in files('B.inprogress')]}")
    check(not os.path.exists(os.path.join(a.out, "B")), "B not committed (no B/ folder)")

    # C
    cs.compress_and_save("C", EPS, blocking=False)
    expect_raise("C injected failure, non-blocking -> checkpoint_join", cs.checkpoint_join, "save FAILED")
    check(files("C.inprogress") == [] and not os.path.exists(os.path.join(a.out, "C")),
          "C nothing written for rank 0, not committed")
    del os.environ[INJ]

    # D
    os.makedirs(os.path.join(a.out, "D.inprogress"))
    os.symlink("/dev/full", os.path.join(a.out, "D.inprogress", "rank_000_exp_avg.cch"))
    expect_raise("D write to a full device", lambda: cs.compress_and_save("D", EPS, blocking=True),
                 "write failed")
    check(not os.path.exists(os.path.join(a.out, "D")), "D not committed (no D/ folder)")

    # E
    mE = cs.compress_and_save("E", EPS, blocking=True)
    check(mE.get("validated") is True, "E next save after failures: validated")

    # F
    src = os.path.join(a.out, "A")
    def variant(name, mutate):
        d = os.path.join(a.out, name)
        shutil.copytree(src, d)
        mutate(d)
        return cs._validate(mA, sorted(glob.glob(os.path.join(d, "rank_000_*"))))
    ok, why = variant("F_copy", lambda d: None)
    check(ok, f"F exact copy valid ({why})")
    def trunc(d):
        p = os.path.join(d, "rank_000_model_state.cch")
        os.truncate(p, os.path.getsize(p) - 100)
    ok, why = variant("F_trunc", trunc)
    check(not ok and "truncated" in why, f"F truncated .cch invalid: {why}")
    def drop_one(d):
        # rewrite exp_avg_sq (v3: header | data | index) with its last tensor
        # removed: a self-consistent but short stream
        p = os.path.join(d, "rank_000_exp_avg_sq.cch")
        b = bytes(open(p, "rb").read())
        H = struct.Struct("<8I d 6Q"); h = list(H.unpack_from(b, 0)); n = h[6]
        assert h[1] == 3, "drop_one expects a v3 file"
        idx_off, data_off = h[9], h[10]
        ents = [b[idx_off + 432 * k: idx_off + 432 * (k + 1)] for k in range(n)]
        offs = [struct.unpack_from("<4Q", e, 344) for e in ents]
        k_last = max(range(n), key=lambda k: offs[k][0])
        keep = min(offs[k_last][0], offs[k_last][2])
        h[6] = n - 1; h[9] = data_off + keep
        kept = b"".join(e for k, e in enumerate(ents) if k != k_last)
        open(p, "wb").write(H.pack(*h) + b[data_off:data_off + keep] + kept)
    ok, why = variant("F_drop", drop_one)
    check(not ok and "holds 3 tensors" in why, f"F stream missing a tensor invalid: {why}")
    ok, why = variant("F_stray", lambda d: shutil.copy(os.path.join(d, "rank_000_exp_avg.cch"),
                                                       os.path.join(d, "rank_000_extra.cch")))
    check(not ok and "not in the manifest" in why, f"F stray .cch invalid: {why}")

    # G
    res = cs.restart_from_checkpoint(engine, opt, targets="all", name="A")
    restored = res.get("restored", {}) or {}
    check(not res.get("missing") and not res.get("warnings") and
          all(restored.get(s) == 4 for s in ("model_state", "master_weights", "exp_avg", "exp_avg_sq")),
          f"G restore of A complete: restored={restored} warnings={len(res.get('warnings') or [])}")

    # J
    mJ = cs.compress_and_save("B", EPS, blocking=True)
    check(mJ.get("validated") is True and os.path.isdir(os.path.join(a.out, "B")) and
          not os.path.exists(os.path.join(a.out, "B.inprogress")) and len(files("B")) == 5,
          f"J retry of B commits: B/ has {len(files('B'))} files, "
          f"B.inprogress exists={os.path.exists(os.path.join(a.out, 'B.inprogress'))}")

    # K
    before = {f: os.stat(f).st_mtime_ns for f in files("A")}
    expect_raise("K save to an existing name", lambda: cs.compress_and_save("A", EPS, blocking=True),
                 "already exists")
    check({f: os.stat(f).st_mtime_ns for f in files("A")} == before, "K existing A untouched")

    # H
    H = struct.Struct("<8I d 6Q")
    bad = []
    for p in sorted(glob.glob(os.path.join(a.out, "A", "rank_000_*.cch"))):
        b = open(p, "rb").read(); h = H.unpack_from(b, 0); n = h[6]
        if h[1] != 3 or h[10] != 88 or h[9] + 432 * n != len(b):
            bad.append(f"{os.path.basename(p)} version={h[1]} data_off={h[10]} index_off={h[9]} size={len(b)}")
        for k in range(n):
            fl = b[h[9] + 432 * k + 258]
            if fl & 2 or not (fl == 1 or fl == (4 | 8)):
                bad.append(f"{os.path.basename(p)} entry {k} flags=0x{fl:02x}")
    check(not bad, "H every .cch v3 with the index at the end, no zstd flags: " + ("; ".join(bad) or "ok"))

    # I
    if a.v2_dir:
        hv = H.unpack_from(open(sorted(glob.glob(os.path.join(a.v2_dir, "rank_000_*.cch")))[0], "rb").read(88), 0)
        res = checkers_py.restart_from_checkpoint(engine, a.v2_dir, optimizer=opt, targets="all")
        restored = res.get("restored", {}) or {}
        check(hv[1] == 2 and not res.get("missing") and not res.get("warnings") and
              all(restored.get(s) == 4 for s in ("model_state", "master_weights", "exp_avg", "exp_avg_sq")),
              f"I restore of v2 checkpoint (version {hv[1]}) complete: restored={restored} "
              f"warnings={len(res.get('warnings') or [])}")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\n=== save failures: {verdict} ({len(rows) - len(fails)}/{len(rows)} checks)")
    for r in rows:
        print("  " + r)
    checkers_py.shutdown()
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
