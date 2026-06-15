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
    Per-tensor statistics + HLL sketch.

    Returns a small dict with scalars + a 16 KB uint8 HLL register array.
    No large arrays are returned — IPC cost is O(16 KB) regardless of tensor size.
    """
    numel = symbols.size

    # np.unique on uint32 is faster than float (no NaN special-casing)
    unique_vals, counts = np.unique(symbols, return_counts=True)
    unique_count = len(unique_vals)

    unique_ratio = unique_count / numel if numel > 0 else 0.0
    probs = counts.astype(np.float64) / numel
    entropy = float(-np.dot(probs, np.log2(probs + 1e-300)))

    hist_bytes   = (unique_count * 8) + 64
    tensor_bytes = symbols.nbytes

    # Build HLL sketch from the raw symbol stream (not from unique_vals —
    # hll_add_uint32 handles duplicates correctly and reflects true distribution)
    regs = hll_create()
    hll_add_uint32(regs, symbols)

    return {
        "unique_ratio": unique_ratio,
        "entropy":      entropy,
        "hist_bytes":   hist_bytes,
        "tensor_bytes": tensor_bytes,
        "hll_regs":     regs,          # 16 384 bytes — cheap to pickle
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

        # # ==================================================
        # # EXACT VS HLL VALIDATION
        # # ==================================================

        # print(
        #     f"\nRunning exact validation on cluster "
        #     f"{validation_cluster_id}..."
        # )

        # merged = np.concatenate(
        #     [all_symbols[i] for i in validation_cluster]
        # )

        # exact_unique = np.unique(merged).size

        # print(
        #     f"Exact unique count: "
        #     f"{exact_unique:,}"
        # )

        # validation_task = (
        #     validation_cluster_id,
        #     validation_cluster,
        #     all_regs
        # )

        # hll_bytes = cluster_union_worker(validation_task)

        # hll_estimate = (hll_bytes - 64) // 8

        # error_pct = (
        #     abs(exact_unique - hll_estimate)
        #     / exact_unique
        # ) * 100

        # print("\n--- HLL VALIDATION ---")
        # print(f"Cluster ID    : {validation_cluster_id}")
        # print(f"Exact uniques : {exact_unique:,}")
        # print(f"HLL uniques   : {hll_estimate:,}")
        # print(f"Error (%)     : {error_pct:.2f}")


        # ----------------------------
        # Singletons & baseline
        # ----------------------------
        singleton_hist_bytes  = sum(all_stats[i]["hist_bytes"] for i in singleton_idxs)
        baseline_hist_bytes   = sum(s["hist_bytes"] for s in all_stats)
        compressed_hist_bytes = cluster_union_bytes + singleton_hist_bytes

        clustered_idxs  = [i for grp in cluster_groups for i in grp]
        cluster_stats   = _group_stats(clustered_idxs, all_stats)
        singleton_stats = _group_stats(singleton_idxs,  all_stats)

        # ----------------------------
        # REPORTING (identical structure to original)
        # ----------------------------
        print(f"\n{'='*40}\nSTREAM: {stream_name}\n{'='*40}")
        print(f"Tensors: {len(tensors)}")
        print(f"Clusters: {len(cluster_groups)}")
        print(f"Singletons: {len(singleton_idxs)}")

        if cluster_stats:
            print("\n--- CLUSTERS ---")
            print(f"Avg unique ratio: {cluster_stats['avg_unique_ratio']:.6f}")
            print(f"Avg entropy: {cluster_stats['avg_entropy']:.4f}")
            print(f"Tensor bytes: {cluster_stats['total_tensor_bytes']:,}")
            print(f"Histogram bytes: {cluster_stats['total_hist_bytes']:,}")
            print(f"Hist/Tensor ratio: {cluster_stats['hist_ratio']:.4f}")

        if singleton_stats:
            print("\n--- SINGLETONS ---")
            print(f"Avg unique ratio: {singleton_stats['avg_unique_ratio']:.6f}")
            print(f"Avg entropy: {singleton_stats['avg_entropy']:.4f}")
            print(f"Tensor bytes: {singleton_stats['total_tensor_bytes']:,}")
            print(f"Histogram bytes: {singleton_stats['total_hist_bytes']:,}")
            print(f"Hist/Tensor ratio: {singleton_stats['hist_ratio']:.4f}")

        print("\n--- SAVINGS ---")
        print(f"Baseline histogram bytes: {baseline_hist_bytes:,}")
        print(f"Cluster-only bytes: {cluster_union_bytes:,}")
        print(f"Singleton bytes: {singleton_hist_bytes:,}")
        print(f"TOTAL compressed: {compressed_hist_bytes:,}")
        print(f"Savings: {baseline_hist_bytes - compressed_hist_bytes:,}")
        print(f"Ratio: {compressed_hist_bytes / baseline_hist_bytes:.4f}")


if __name__ == "__main__":
    main()