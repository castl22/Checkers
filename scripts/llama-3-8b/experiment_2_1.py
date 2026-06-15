import os
import torch
import numpy as np
import multiprocessing as mp
# Set this BEFORE creating the pool
mp.set_start_method('spawn', force=True)
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

CHECKPOINT_DIR = "/p/lustre5/kogiou1/llama-3-8b/8_GPUs/math/DS/epoch_0"
NUM_RANKS = 8
N_WORKERS = min(96, mp.cpu_count())

# ----------------------------------------------------
# HyperLogLog — pure numpy, no extra deps
#
# Why HLL instead of exact unique arrays?
#   Exact union over a cluster of tensors requires merging sorted uint32
#   arrays that together hold billions of elements — O(N) memory and time.
#   HLL replaces each tensor's unique set with a 16 KB register array.
#   Union of two sketches is just element-wise max of 16 384 uint8 values:
#   microseconds instead of seconds, kilobytes instead of gigabytes.
#   Error: ~0.8% standard deviation (b=14 registers).
# ----------------------------------------------------

HLL_B = 14                           # register index bits
HLL_M = 1 << HLL_B                  # 16 384 registers
_HLL_ALPHA = 0.7213 / (1.0 + 1.079 / HLL_M)

# ----------------------------------------------------
# Quantized probability model builder
#
# Used to compute cross-entropy (XE) stream cost:
#   bits(s) = log2(scale / norm[s])  for each symbol s
# This is the expected coded length under a finite-precision discrete model —
# NOT actual ANS/rANS/tANS (no state machine, no renormalization tables).
#
# Why adaptive precision?  Per-tensor vocab is at most 65 536 unique fp32
# bit-patterns.  A cluster model may have 4× that.  Fixing SCALE too low
# forces many symbols to their minimum norm of 1, distorting both per-tensor
# and cluster XE estimates.  Rule: SCALE >= 2 × n_vocab.
# ----------------------------------------------------

PRECISION = 14                          # minimum log2(SCALE); raised per call


def _build_model(counts: np.ndarray):
    """
    Normalize raw symbol counts into a quantized discrete probability model.

    Precision adapts so SCALE > 2 × n_vocab: every symbol gets ≥ 2 normalized
    counts, which guarantees sum(norm) ≤ SCALE and a non-negative rounding
    correction.

    Returns
    -------
    norm  : int64 array, length n_vocab, sums exactly to `scale`
    scale : int, power of two
    """
    n_vocab = len(counts)
    p       = max(PRECISION, int(np.ceil(np.log2(max(n_vocab, 2)))) + 1)
    scale   = 1 << p                    # smallest power-of-2 s.t. scale > 2*n_vocab

    total   = int(counts.sum())
    norm    = np.floor(counts * (scale / total)).astype(np.int64)
    norm    = np.maximum(norm, 1)
    # Absorb rounding drift onto the most-frequent symbol (always non-negative)
    norm[np.argmax(counts)] += scale - int(norm.sum())
    return np.maximum(norm, 1), scale


def hll_create() -> np.ndarray:
    return np.zeros(HLL_M, dtype=np.uint8)


def hll_add_uint32(regs: np.ndarray, values: np.ndarray) -> None:
    """
    Insert all uint32 values into HLL registers IN-PLACE, fully vectorized.

    Hash: MurmurHash3 fmix32 finalizer — excellent avalanche, no Python loop.
    Leading-zero count: log2 trick, works in O(N) numpy ops.
    Scatter-max: np.maximum.at (unbuffered ufunc, correct for duplicates).
    """
    h = values.astype(np.uint32)
    h ^= h >> np.uint32(16)
    h  = (h * np.uint32(0x85ebca6b)) & np.uint32(0xFFFFFFFF)
    h ^= h >> np.uint32(13)
    h  = (h * np.uint32(0xc2b2ae35)) & np.uint32(0xFFFFFFFF)
    h ^= h >> np.uint32(16)

    idx = (h >> np.uint32(32 - HLL_B)).astype(np.int32)

    # Shift remaining bits to the top so log2 gives the leading-zero position
    w = (h << np.uint32(HLL_B)) | np.uint32(1 << (HLL_B - 1))
    with np.errstate(divide="ignore"):
        lz = (31 - np.floor(np.log2(w.astype(np.float64)))).astype(np.uint8)
    rho = np.clip(lz + 1, 1, 32 - HLL_B).astype(np.uint8)

    np.maximum.at(regs, idx, rho)


def hll_merge(sketches: list) -> np.ndarray:
    """Element-wise max over a list of register arrays — O(M * n_sketches)."""
    return np.stack(sketches).max(axis=0)      # vectorized; M=16384


def hll_count(regs: np.ndarray) -> float:
    """Estimate cardinality from register array (with small/large corrections)."""
    Z   = 1.0 / np.sum(np.exp2(-regs.astype(np.float64)))
    raw = _HLL_ALPHA * HLL_M * HLL_M * Z
    if raw <= 2.5 * HLL_M:
        V = int(np.sum(regs == 0))
        if V > 0:
            raw = HLL_M * np.log(HLL_M / V)
    elif raw > (1.0 / 30.0) * (2 ** 32):
        raw = -(2 ** 32) * np.log(1.0 - raw / 2 ** 32)
    return raw


# ----------------------------------------------------
# WORKERS
# ----------------------------------------------------

def tensor_stats_worker(symbols: np.ndarray):
    """
    Combined per-tensor statistics, XE baseline costs, and HLL sketch.

    Single pass over the symbol array — no duplicate work.
    IPC cost: O(n_unique × 12 B + 16 KB HLL).  No large symbol arrays returned.

    Keys returned
    -------------
    Clustering features : unique_ratio, entropy, hist_bytes, tensor_bytes, hll_regs
    Model summary       : vocab (uint32 sorted), counts (int64)
                          kept compact for cluster-model merging in main process
    Baseline XE costs   : xe_stream_bytes, per_tensor_model_bytes
                          cross-entropy cost using THIS tensor's own model
    """
    numel = symbols.size

    # np.unique: sorted unique values + inverse mapping + counts — one call
    vocab, compact, counts = np.unique(
        symbols, return_inverse=True, return_counts=True
    )
    n_vocab = len(vocab)

    unique_ratio = n_vocab / numel if numel > 0 else 0.0
    probs        = counts.astype(np.float64) / numel
    entropy      = float(-np.dot(probs, np.log2(probs + 1e-300)))

    hist_bytes   = (n_vocab * 8) + 64
    tensor_bytes = int(symbols.nbytes)

    # XE stream cost under own model — vectorized, no encode loop.
    # bits(s) = log2(scale / norm[s]) is the cross-entropy coded length.
    norm, scale = _build_model(counts)
    bits = float(np.sum(np.log2(scale / norm[compact].astype(np.float64))))

    # HLL sketch (built from raw stream — hll_add_uint32 handles duplicates)
    regs = hll_create()
    hll_add_uint32(regs, symbols)

    return {
        # Clustering features
        "unique_ratio": unique_ratio,
        "entropy":      entropy,
        "hist_bytes":   hist_bytes,
        "tensor_bytes": tensor_bytes,
        "hll_regs":     regs,
        # Compact model summary (vocab + counts; norm recomputed during merge)
        "vocab":        vocab,          # uint32 sorted, n_vocab entries
        "counts":       counts,         # int64, n_vocab entries
        # Baseline XE costs (own model — NOT self-modeled; norm is built from
        # the same data but the cost formula is Shannon XE, not adaptive coding)
        "xe_stream_bytes":       bits / 8.0,
        "per_tensor_model_bytes": n_vocab * 8,  # vocab(4 B) + norm_entry(4 B)
    }


def cluster_union_worker(task):
    """
    Estimate union cardinality for one cluster via HLL sketch merge.

    Replaces the old sorted np.union1d chain:
      OLD: O(N_total * log N_total) time, O(N_total) memory (N_total = billions)
      NEW: O(M * n_tensors) time, O(M) memory  (M = 16 384)

    Union of HLL sketches = element-wise max → microseconds per cluster.
    """
    cluster_id, idxs, all_regs = task

    print(f"[CLUSTER {cluster_id}] {len(idxs)} tensors", flush=True)

    sketches  = [all_regs[i] for i in idxs]
    merged    = hll_merge(sketches)
    est_union = hll_count(merged)

    print(f"[CLUSTER {cluster_id}] done | est_uniques={est_union:,.0f}", flush=True)
    return int(est_union * 8) + 64


# ----------------------------------------------------
# CROSS-ENTROPY CLUSTER COST HELPERS
#
# These work on compact (vocab, counts) summaries — never raw symbol arrays.
# IPC proportional to n_unique (kilobytes) not n_symbols (gigabytes).
#
# Naming:  XE = cross-entropy.  This is expected coded length under a
# quantized discrete model, NOT actual ANS/rANS/tANS bitstream output.
# ----------------------------------------------------

def merge_vocab_counts(vocab_list, counts_list):
    """
    K-way sorted merge of (vocab, counts) summaries into cluster-wide totals.

    Uses np.add.reduceat to accumulate runs of equal values after a single
    sort — avoids holding a duplicate sorted array during the reduction.

    Time:   O(Σ n_vocab_i · log Σ n_vocab_i)
    Memory: O(Σ n_vocab_i)  — no extra copy beyond the sort permutation
    """
    all_v = np.concatenate([v.astype(np.int64) for v in vocab_list])
    all_c = np.concatenate(counts_list)
    order = np.argsort(all_v, kind="stable")
    sv, sc = all_v[order], all_c[order]
    del order                               # free sort permutation immediately
    change   = np.concatenate([[True], sv[1:] != sv[:-1]])
    unique_v = sv[change].astype(np.uint32)
    merged   = np.add.reduceat(sc, np.where(change)[0]).astype(np.int64)
    return unique_v, merged


def _xe_chunk(args):
    """
    Vectorized XE cost for a batch of tensors against a pre-built cluster model.
    Designed to run in a thread inside cluster_cost_worker; numpy releases the
    GIL for all array ops so threads execute in true parallel.

    Lookup: np.searchsorted against sorted cluster_vocab (cast to int64).
    Safety: both cluster_vocab and every tensor vocab are sorted uint32; every
    tensor vocab is a guaranteed subset of cluster_vocab (we built it by merging
    all tensor vocabs in the cluster).  searchsorted is therefore exact.

    Algorithm:
      1. Concatenate all tensor vocabs / counts into two flat arrays
      2. One searchsorted call  →  positions in cluster model
      3. One log2 call          →  bits per unique symbol
      4. np.add.reduceat        →  per-tensor bit sums, then grand total

    Time:   O(Σ n_vocab_i · log n_cluster_vocab)  — pure C/numpy, no Python loop
    Memory: O(Σ n_vocab_i)  within this batch
    """
    vocab_slice, counts_slice, cv_i64, cluster_norm_f64, scale = args

    sizes   = np.array([len(v) for v in vocab_slice], dtype=np.intp)
    offsets = np.zeros(len(sizes), dtype=np.intp)
    if len(sizes) > 1:
        offsets[1:] = np.cumsum(sizes[:-1])

    all_v = np.concatenate([v.astype(np.int64) for v in vocab_slice])
    all_c = np.concatenate(counts_slice).astype(np.float64)

    idx  = np.searchsorted(cv_i64, all_v)
    bits = all_c * np.log2(scale / cluster_norm_f64[idx])
    return float(np.add.reduceat(bits, offsets).sum())


def cluster_cost_worker(task):
    """
    Build one cluster XE model then compute cross-entropy costs for all tensors.

    Steps
    -----
    1. Merge per-tensor (vocab, counts) → cluster (vocab, counts)       [numpy]
    2. Build quantized model: cluster_norm, scale                       [numpy]
    3. Fan out per-tensor XE to N threads (numpy GIL-free)              [parallel]
    4. Reduce: sum bits, return (stream_bytes, cluster_vocab_size)      [main thread]

    Thread count: min(n_tensors, 16).  More threads add overhead for tiny
    clusters; 16 caps parallelism at the point of diminishing returns.

    Input  : (cluster_id, list of dicts with keys 'vocab' and 'counts')
    Output : (total_stream_bytes: float, cluster_vocab_size: int)

    IPC in : O(Σ n_unique_i × 12 B) — compact summaries, NOT raw symbols
    IPC out: two Python scalars
    """
    cluster_id, summaries = task

    vocab_list  = [s["vocab"]  for s in summaries]
    counts_list = [s["counts"] for s in summaries]

    cluster_vocab, cluster_counts = merge_vocab_counts(vocab_list, counts_list)
    cluster_norm, scale           = _build_model(cluster_counts)

    cv_i64           = cluster_vocab.astype(np.int64)
    cluster_norm_f64 = cluster_norm.astype(np.float64)   # cast once, reused by all threads

    # Fan out to threads: split tensors into equal-sized chunks
    n_tensors  = len(summaries)
    n_threads  = min(n_tensors, 16)
    chunk_size = max(1, (n_tensors + n_threads - 1) // n_threads)

    chunks = [
        (vocab_list [i : i + chunk_size],
         counts_list[i : i + chunk_size],
         cv_i64, cluster_norm_f64, scale)
        for i in range(0, n_tensors, chunk_size)
    ]

    if len(chunks) == 1:
        total_bits = _xe_chunk(chunks[0])
    else:
        with ThreadPoolExecutor(max_workers=len(chunks)) as tpool:
            total_bits = sum(tpool.map(_xe_chunk, chunks))

    total_stream_bytes = total_bits / 8.0

    print(
        f"[CLUSTER-COST {cluster_id}] {n_tensors} tensors | "
        f"vocab={len(cluster_vocab):,} | threads={len(chunks)}",
        flush=True,
    )
    return total_stream_bytes, len(cluster_vocab)


# ----------------------------------------------------
# PARALLEL SHARD LOADER
# ----------------------------------------------------

def _load_optim_shard(args):
    rank, checkpoint_dir = args
    path = os.path.join(
        checkpoint_dir,
        f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
    )
    print(f"Loading {path}", flush=True)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    opt  = ckpt["optimizer_state_dict"]

    weight_shard = opt["fp32_flat_groups"][0]

    state = opt["optimizer_state_dict"]["state"]
    exp_avg_list, exp_avg_sq_list = [], []
    for v in state.values():
        if "exp_avg"    in v: exp_avg_list.append(v["exp_avg"].flatten())
        if "exp_avg_sq" in v: exp_avg_sq_list.append(v["exp_avg_sq"].flatten())

    return rank, weight_shard, torch.cat(exp_avg_list), torch.cat(exp_avg_sq_list)


def load_all_shards_parallel(checkpoint_dir, num_ranks):
    args    = [(r, checkpoint_dir) for r in range(num_ranks)]
    results = [None] * num_ranks
    with ThreadPoolExecutor(max_workers=min(num_ranks, 8)) as pool:
        futures = {pool.submit(_load_optim_shard, a): a[0] for a in args}
        for fut in as_completed(futures):
            rank, w, ea, easq = fut.result()
            results[rank] = (w, ea, easq)
    return (
        [r[0] for r in results],
        [r[1] for r in results],
        [r[2] for r in results],
    )


# ----------------------------------------------------
# TENSOR EXTRACTION
# ----------------------------------------------------

def extract_tensors(flat_tensor: torch.Tensor, param_shapes):
    numpy_flat  = flat_tensor.detach().cpu().numpy()
    uint32_flat = numpy_flat.view(np.uint32)            # zero-copy reinterpret

    tensors, all_symbols = [], []
    offset = 0
    for shape in param_shapes.values():
        numel = shape.numel()
        tensors.append(numpy_flat[offset:offset + numel])
        all_symbols.append(uint32_flat[offset:offset + numel])
        offset += numel
    return tensors, all_symbols


# ----------------------------------------------------
# REPORTING HELPERS
# ----------------------------------------------------

def _group_stats(idxs, all_stats):
    if not idxs:
        return None
    stats        = [all_stats[i] for i in idxs]
    total_tensor = sum(s["tensor_bytes"] for s in stats)
    total_hist   = sum(s["hist_bytes"]   for s in stats)
    return {
        "avg_unique_ratio":   np.mean([s["unique_ratio"] for s in stats]),
        "avg_entropy":        np.mean([s["entropy"]      for s in stats]),
        "total_tensor_bytes": total_tensor,
        "total_hist_bytes":   total_hist,
        "hist_ratio": total_hist / total_tensor if total_tensor > 0 else 0.0,
    }

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():

    model_ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "zero_pp_rank_0_mp_rank_00_model_states.pt"),
        map_location="cpu",
        weights_only=False,
    )
    param_shapes = model_ckpt["param_shapes"][0]

    weights_shards, exp_avg_shards, exp_avg_sq_shards = load_all_shards_parallel(
        CHECKPOINT_DIR, NUM_RANKS
    )

    streams = {
        "weights":    torch.cat(weights_shards),
        "exp_avg":    torch.cat(exp_avg_shards),
        "exp_avg_sq": torch.cat(exp_avg_sq_shards),
    }

    for stream_name, flat_tensor in streams.items():

        print(f"\n{'#'*20} STREAM: {stream_name} {'#'*20}")

        tensors, all_symbols = extract_tensors(flat_tensor, param_shapes)
        print(f"Loaded {len(tensors)} tensors")

        # ----------------------------
        # Per-tensor stats + HLL sketch (parallel)
        # Each worker returns ≤ 16 KB over IPC — fast regardless of tensor size
        # ----------------------------
        print("Computing stats + HLL sketches...")

        all_stats  = [None] * len(all_symbols)
        chunksize  = max(1, len(all_symbols) // (N_WORKERS * 4))

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for i, res in enumerate(pool.map(tensor_stats_worker, all_symbols,
                                             chunksize=chunksize)):
                all_stats[i] = res
                if (i + 1) % 10 == 0 or (i + 1) == len(all_symbols):
                    print(f"[STATS] {i+1}/{len(all_symbols)}")

        # ----------------------------
        # Clustering
        # ----------------------------
        features = np.stack([
            [s["unique_ratio"], s["entropy"], s["hist_bytes"]]
            for s in all_stats
        ])

        n_clusters = max(8, len(tensors) // 40)
        labels = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=0, n_init="auto",
            batch_size=min(1024, len(tensors)),
        ).fit_predict(features)

        clusters = defaultdict(list)
        for i, c in enumerate(labels):
            clusters[c].append(i)

        cluster_groups  = [v for v in clusters.values() if len(v) > 1]
        singleton_idxs  = [v[0] for v in clusters.values() if len(v) == 1]

        largest_cluster_id = max(
            range(len(cluster_groups)),
            key=lambda i: len(cluster_groups[i])
        )

        # ==================================================
        # VALIDATION CLUSTER
        # ==================================================

        for cid, idxs in enumerate(cluster_groups):
            print(f"Cluster {cid}: {len(idxs)} tensors")

        # Use a tiny cluster for exact validation
        validation_cluster_id = 0

        validation_cluster = cluster_groups[validation_cluster_id]

        validation_elements = sum(
            all_symbols[i].size
            for i in validation_cluster
        )

        print(
            f"\nValidation cluster: "
            f"{validation_cluster_id}"
        )

        print(
            f"Tensors in cluster: "
            f"{len(validation_cluster)}"
        )

        print(
            f"Elements in cluster: "
            f"{validation_elements:,}"
        )

        print(
            f"Approx raw size: "
            f"{validation_elements * 4 / 1024**3:.2f} GB"
        )

        # ==================================================
        # END VALIDATION PREP
        # ==================================================

        # ==================================================
        # SANITY CHECK PREP
        # ==================================================

        target_cluster_id = largest_cluster_id

        cluster_unique_sum = sum(
            all_stats[i]["hist_bytes"]
            for i in cluster_groups[target_cluster_id]
        )

        cluster_unique_count_sum = (
            cluster_unique_sum -
            64 * len(cluster_groups[target_cluster_id])
        ) // 8

        print(
            f"\nCluster {target_cluster_id} "
            f"sum individual uniques: "
            f"{cluster_unique_count_sum:,}"
        )


        # ----------------------------
        # Cluster union via HLL merge
        #
        # All HLL register arrays together = n_tensors * 16 KB.
        # For ~300 tensors that's ~5 MB — trivial to pass to workers.
        # Each worker does: np.stack(sketches).max(axis=0) + hll_count()
        # Runtime per cluster: microseconds regardless of how many elements
        # the tensors contain.
        # ----------------------------
        all_regs     = [s["hll_regs"] for s in all_stats]
        cluster_tasks = [
            (cid, idxs, all_regs)
            for cid, idxs in enumerate(cluster_groups)
        ]

        print("Computing cluster unions (HLL merge)...")

        cluster_union_bytes = 0
        n_union_workers = min(N_WORKERS, max(1, len(cluster_tasks)))

        with ProcessPoolExecutor(max_workers=n_union_workers) as pool:
            futures = {
                pool.submit(cluster_union_worker, task): cid
                for cid, task in enumerate(cluster_tasks)
            }
            done = 0
            for fut in as_completed(futures):
                cluster_union_bytes += fut.result()
                done += 1
                print(f"[UNION] {done}/{len(futures)} clusters done",
                      end="\r", flush=True)

        print("\nCluster unions complete.")

        # ----------------------------------------------------
        # CROSS-ENTROPY (XE) MODEL SHARING EXPERIMENT
        #
        # Measures whether sharing one probability model across a cluster of
        # tensors reduces TOTAL cost compared to per-tensor models.
        #
        # Cost metric: cross-entropy expected coded length (NOT actual ANS).
        #   Stream cost : (1/8) Σ_s  log2(scale / norm[s])  bytes
        #   Model cost  : vocab_size × 8 bytes  (4 B value + 4 B norm entry)
        #
        # (A) BASELINE  : each tensor uses its OWN model (built from own data)
        #       cost = Σ_i  per_tensor_model_bytes_i  +  XE(tensor_i | model_i)
        #
        # (B) CLUSTERED : tensors in a cluster SHARE ONE model
        #       cost = Σ_c  cluster_model_bytes_c
        #              + Σ_{i in c}  XE(tensor_i | cluster_model_c)
        #
        # The cluster model is built from ALL tensors in the cluster —
        # never just from the tensor being measured (no self-modeling).
        #
        # Model bytes are computed in main (once per cluster), NOT inside the
        # worker — avoids any risk of double-counting across task boundaries.
        # ----------------------------------------------------

        print("\n[EXPERIMENT] Cross-entropy cost estimation (XE model sharing)...")

        # (A) Baseline — already computed inside tensor_stats_worker; free here
        per_tensor_xe_bytes    = sum(s["xe_stream_bytes"]        for s in all_stats)
        per_tensor_model_bytes = sum(s["per_tensor_model_bytes"] for s in all_stats)
        raw_bytes              = sum(s["tensor_bytes"]           for s in all_stats)

        print(f"  Baseline: XE stream={per_tensor_xe_bytes / 1e9:.3f} GB  "
              f"model={per_tensor_model_bytes / 1e9:.3f} GB")

        # (B) Clustered — workers receive only compact (vocab, counts) summaries,
        #     NOT raw symbol arrays, keeping IPC at O(Σ n_unique × 12 B).
        #     Each worker returns (stream_bytes, cluster_vocab_size).
        #     Model bytes = cluster_vocab_size * 8  computed here in main.
        cluster_tasks_xe = [
            (cid, [{"vocab":  all_stats[i]["vocab"],
                    "counts": all_stats[i]["counts"]}
                   for i in idxs])
            for cid, idxs in enumerate(cluster_groups)
        ]

        print(f"  Building {len(cluster_tasks_xe)} cluster XE models in parallel...")

        cluster_xe_bytes    = 0.0
        cluster_model_bytes = 0.0

        n_xe_workers = min(N_WORKERS, max(1, len(cluster_tasks_xe)))
        with ProcessPoolExecutor(max_workers=n_xe_workers) as pool:
            futures = {
                pool.submit(cluster_cost_worker, task): task[0]
                for task in cluster_tasks_xe
            }
            for fut in as_completed(futures):
                s_b, vocab_size = fut.result()
                cluster_xe_bytes    += s_b
                cluster_model_bytes += vocab_size * 8   # one model per cluster

        # Singletons have no cluster partner → keep per-tensor baseline
        for i in singleton_idxs:
            cluster_xe_bytes    += all_stats[i]["xe_stream_bytes"]
            cluster_model_bytes += all_stats[i]["per_tensor_model_bytes"]

        # Reporting
        def report_xe(label, stream_b, model_b, raw_b):
            total = stream_b + model_b
            print(f"\n--- {label} ---")
            print(f"  XE stream : {stream_b / 1e9:8.3f} GB  ({stream_b / raw_b:.4f}x raw)")
            print(f"  Model     : {model_b / 1e9:8.3f} GB  ({model_b / raw_b:.4f}x raw)")
            print(f"  Total     : {total   / 1e9:8.3f} GB  ({total   / raw_b:.4f}x raw)")

        report_xe("BASELINE  (per-tensor model, own data)",
                  per_tensor_xe_bytes, per_tensor_model_bytes, raw_bytes)
        report_xe("CLUSTERED (shared model per cluster)",
                  cluster_xe_bytes, cluster_model_bytes, raw_bytes)

        model_savings  = per_tensor_model_bytes - cluster_model_bytes
        stream_penalty = cluster_xe_bytes       - per_tensor_xe_bytes
        net_gain       = model_savings - stream_penalty

        print(f"\n--- CLUSTER EFFECT ---")
        print(f"  Model savings   : {model_savings  / 1e6:+8.1f} MB  ({model_savings  / raw_bytes:+.4f}x raw)")
        print(f"  XE penalty      : {stream_penalty / 1e6:+8.1f} MB  ({stream_penalty / raw_bytes:+.4f}x raw)")
        print(f"  Net gain        : {net_gain       / 1e6:+8.1f} MB  ({net_gain       / raw_bytes:+.4f}x raw)")


if __name__ == "__main__":
    main()