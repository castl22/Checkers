#!/usr/bin/env python3
"""
test_restart_roundtrip.py -- save -> restore round trip through Checkers' real
compress_and_save / restart_from_checkpoint, on one GPU.

    python test_restart_roundtrip.py --measure-snr {0,1} --out DIR

A tiny ZeRO-3 model takes two real optimizer steps so all four streams exist.
Checkers saves them at eps=0.05, every live tensor is then overwritten with
NaN (so anything the restore fails to write is visible), and the restore runs.
Each stream is checked against what its codec promises:

  model_state, master_weights   per-group RMS step   rel_l2 <= REL_PG
  exp_avg_sq                    log-domain           max |v_hat/v - 1| <= eps (+ slack)
  exp_avg                       TT + residual        raw-path tensors bit-exact; TT-path
                                                     tensors no worse than the same codec
                                                     in memory (tt_eq_selftest) + slack
  every stream                  no NaN left, restore report complete

Regression target: with measure_snr=1, codec_pergroup used to pack un-zigzagged
codes, so model_state / master_weights / exp_avg_sq restored as garbage while
measure_snr=0 was fine.  run_restart_roundtrip.sh runs both settings and also
requires the per-group streams' .cch files to be byte-identical between them --
a diagnostics flag must not change what is written.

Exit status 0 = PASS, 1 = FAIL.
"""
import argparse, json, os, sys

os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 2000))
os.environ.setdefault("DS_ACCELERATOR", "cuda")

import torch
import torch.nn as nn
import deepspeed

sys.path.insert(0, "/usr/WS1/kogiou1/LLM_work/Checkers/build")
import checkers_py

EPS = 0.05
REL_PG = 0.06          # per-group RMS bound -> measured ~0.027 on LLaMA shards
LOG_SLACK = 1e-3       # float rounding on top of the hard log-domain bound
TT_SLACK = 0.05        # TT-SVD is randomized; allow this much above the in-memory run


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(512, 512, bias=False)     # 262144 elems: TT path for exp_avg
        self.l2 = nn.Linear(512, 512, bias=False)
        self.l3 = nn.Linear(512, 512, bias=False)
        self.norm = nn.Parameter(torch.ones(512))     # small: raw path for exp_avg

    def forward(self, x):
        h = torch.tanh(self.l1(x)) * self.norm
        return self.l3(torch.tanh(self.l2(h)))


def snapshot(engine, opt):
    st = opt.optimizer.state
    d = {"model_state": {n: p.ds_tensor.detach().float().clone()
                         for n, p in engine.module.named_parameters()},
         "master": [], "exp_avg": [], "exp_avg_sq": [], "layout": []}
    names = {id(p): n for n, p in engine.module.named_parameters()}
    for g, flat in enumerate(opt.fp32_partitioned_groups_flat):
        d["master"].append(flat.detach().clone())
        d["exp_avg"].append(st[flat]["exp_avg"].detach().clone())
        d["exp_avg_sq"].append(st[flat]["exp_avg_sq"].detach().clone())
        off, lay = 0, []
        for p in opt.fp16_groups[g]:
            lay.append((names[id(p)], off, p.ds_tensor.numel()))
            off += p.ds_tensor.numel()
        d["layout"].append(lay)
    return d


def poison(engine, opt):
    st = opt.optimizer.state
    for _, p in engine.module.named_parameters():
        p.ds_tensor.data.fill_(float("nan"))
        if getattr(p, "ds_persist", False):
            p.data.fill_(float("nan"))      # the full copy the forward pass reads
    for flat in opt.fp32_partitioned_groups_flat:
        flat.data.fill_(float("nan"))
        st[flat]["exp_avg"].fill_(float("nan"))
        st[flat]["exp_avg_sq"].fill_(float("nan"))


def rel_l2(a, b):
    a = a.double(); b = b.double()
    return ((a - b).norm() / a.norm().clamp_min(1e-300)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-snr", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.set_device(0)

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

    checkers_py.initialize_context(engine, opt, stage_targets="all", group_size=256,
                                   zstd_level=3, compute_streams=4,
                                   measure_snr=bool(a.measure_snr))
    orig = snapshot(engine, opt)
    os.makedirs(a.out, exist_ok=True)
    checkers_py.compress_and_save(a.out, error_bound=EPS, targets="all", blocking=True)
    torch.cuda.synchronize()

    poison(engine, opt)
    res = checkers_py.restart_from_checkpoint(engine, a.out, optimizer=opt, targets="all")
    torch.cuda.synchronize()
    rest = snapshot(engine, opt)

    fails, rows = [], []
    def check(ok, what):
        rows.append(("PASS" if ok else "FAIL") + "  " + what)
        if not ok:
            fails.append(what)

    restored = res.get("restored", {}) or {}
    check(not res.get("missing") and not res.get("warnings")
          and all(restored.get(s) for s in ("model_state", "master_weights", "exp_avg", "exp_avg_sq")),
          f"restore report complete: restored={restored} missing={res.get('missing')} "
          f"warnings={len(res.get('warnings') or [])}")

    for n, t in orig["model_state"].items():
        r = rest["model_state"][n]
        check(torch.isfinite(r).all().item() and rel_l2(t, r) <= REL_PG,
              f"model_state    {n:10s} rel_l2={rel_l2(t, r):.4f} (<= {REL_PG})")

    # ZeRO-3 persistent params: the forward pass reads the full copy, so after a
    # restore it must equal the RESTORED partition (world size 1: partition =
    # whole param), not whatever was there before (poisoned to NaN above).
    persist = [(n, p) for n, p in engine.module.named_parameters() if getattr(p, "ds_persist", False)]
    check(len(persist) > 0, f"persistent params present: {[n for n, _ in persist]}")
    for n, p in persist:
        full = p.data.detach().flatten().float()
        part = p.ds_tensor.detach().flatten().float()[:full.numel()]
        check(torch.equal(full, part),
              f"persistent     {n:10s} forward copy == restored partition: {torch.equal(full, part)}"
              f" (finite={torch.isfinite(full).all().item()})")
    for gi, lay in enumerate(orig["layout"]):
        for n, off, k in lay:
            sl = slice(off, off + k)
            o, r = orig["master"][gi][sl], rest["master"][gi][sl]
            check(torch.isfinite(r).all().item() and rel_l2(o, r) <= REL_PG,
                  f"master_weights {n:10s} rel_l2={rel_l2(o, r):.4f} (<= {REL_PG})")

            o, r = orig["exp_avg_sq"][gi][sl], rest["exp_avg_sq"][gi][sl]
            pos = o > 1e-30
            mrel = ((r[pos] / o[pos]) - 1).abs().max().item() if pos.any() else 0.0
            check(torch.isfinite(r).all().item() and mrel <= EPS + LOG_SLACK,
                  f"exp_avg_sq     {n:10s} max|v_hat/v-1|={mrel:.4f} (<= {EPS + LOG_SLACK})")

            o, r = orig["exp_avg"][gi][sl], rest["exp_avg"][gi][sl]
            if k < (1 << 15):
                check(torch.equal(o, r), f"exp_avg        {n:10s} raw path bit-exact={torch.equal(o, r)}")
            else:
                ref = checkers_py.tt_eq_selftest(o.cpu().numpy(), eps=EPS, rank_cap=16, ndim=3,
                                                 energy_keep=0.999, eq_mode=1, entropy=1)["rel_l2"]
                rr = rel_l2(o, r)
                check(torch.isfinite(r).all().item() and rr <= ref + TT_SLACK,
                      f"exp_avg        {n:10s} rel_l2={rr:.4f} (in-memory codec {ref:.4f} + {TT_SLACK})")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\n=== measure_snr={a.measure_snr}: {verdict} ({len(rows) - len(fails)}/{len(rows)} checks)")
    for r in rows:
        print("  " + r)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump({"measure_snr": a.measure_snr, "verdict": verdict, "checks": rows}, f, indent=1)
    checkers_py.shutdown()
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
