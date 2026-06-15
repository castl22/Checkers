import os
import math
import torch
import numpy as np

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

CHECKPOINT_DIR = "/p/lustre5/kogiou1/llama-3-8b/8_GPUs/math/DS/epoch_0"

NUM_RANKS = 8

OUTPUT_DIR = "./experiment1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(OUTPUT_DIR, "report.txt")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary.txt")

# ----------------------------------------------------
# LOAD PARAM SHAPES
# ----------------------------------------------------

model_ckpt = torch.load(
    os.path.join(
        CHECKPOINT_DIR,
        "zero_pp_rank_0_mp_rank_00_model_states.pt"
    ),
    map_location="cpu",
    weights_only=False,
)

param_shapes = model_ckpt["param_shapes"][0]

# ----------------------------------------------------
# LOAD ALL SHARDS
# ----------------------------------------------------

print("Loading optimizer shards...")

all_shards = []

weights_shards = []
exp_avg_shards = []
exp_avg_sq_shards = []

for rank in range(NUM_RANKS):

    fname = f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)

    print(f"Loading {fname}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    opt = ckpt["optimizer_state_dict"]

    # 1. weights (flat buffer)
    weights_shards.append(opt["fp32_flat_groups"][0])

    # 2. optimizer states (structured)
    state = opt["optimizer_state_dict"]["state"]

    exp_avg_list = []
    exp_avg_sq_list = []

    for _, v in state.items():
        if "exp_avg" in v:
            exp_avg_list.append(v["exp_avg"].flatten())
        if "exp_avg_sq" in v:
            exp_avg_sq_list.append(v["exp_avg_sq"].flatten())

    exp_avg_shards.append(torch.cat(exp_avg_list))
    exp_avg_sq_shards.append(torch.cat(exp_avg_sq_list))

full_weights = torch.cat(weights_shards)
full_exp_avg = torch.cat(exp_avg_shards)
full_exp_avg_sq = torch.cat(exp_avg_sq_shards)

streams = {
    "weights": full_weights,
    "exp_avg": full_exp_avg,
    "exp_avg_sq": full_exp_avg_sq,
}

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------

def shannon_entropy(counts):

    probs = counts.astype(np.float64)
    probs /= probs.sum()

    probs = probs[probs > 0]

    return float(-(probs * np.log2(probs)).sum())


def analyze_tensor(name, tensor):

    arr = tensor.detach().cpu().numpy()

    numel = arr.size

    tensor_bytes = arr.nbytes

    #
    # IMPORTANT:
    # Treat FP32 bit patterns as symbols.
    #

    symbols = arr.view(np.uint32)

    unique_vals, counts = np.unique(
        symbols,
        return_counts=True
    )

    unique_count = len(unique_vals)
    numel = symbols.size

    unique_ratio = unique_count / numel  # ✅ FIXED

    # --------------------------------------------
    # EXACT entropy of observed distribution
    # --------------------------------------------
    probs = counts.astype(np.float64)
    probs /= probs.sum()
    entropy = float(-(probs * np.log2(probs)).sum())

    # --------------------------------------------
    # LOSSLESS ANS MODEL SIZE (NO SAMPLING)
    # --------------------------------------------

    if numel < 2**16:
        freq_bytes_per_symbol = 2
    else:
        freq_bytes_per_symbol = 4

    freq_table_bytes = unique_count * freq_bytes_per_symbol

    # NOTE: CDF is NOT required for ANS model storage (we discussed this)
    cdf_bytes = 0

    metadata_bytes = 64

    histogram_bytes = freq_table_bytes + metadata_bytes

    # --------------------------------------------
    # encoded stream size estimate (lossless lower bound)
    # --------------------------------------------
    estimated_bits = entropy * numel
    estimated_compressed_bytes = estimated_bits / 8.0

    # --------------------------------------------
    # ratios
    # --------------------------------------------
    histogram_ratio = histogram_bytes / (numel * 4)

    return {
        "name": name,
        "numel": numel,
        "tensor_bytes": tensor_bytes,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "entropy": entropy,
        "estimated_compressed_bytes": estimated_compressed_bytes,
        "histogram_bytes": histogram_bytes,
        "histogram_ratio": histogram_ratio,
    }

# ----------------------------------------------------
# RECONSTRUCT TENSORS
# ----------------------------------------------------

results = {
    "weights": [],
    "exp_avg": [],
    "exp_avg_sq": [],
}

print("Reconstructing tensors...")

for stream_name, full_tensor in streams.items():
    print(f"\n=== Processing stream: {stream_name} ===")
    offset = 0
    for name, shape in param_shapes.items():
        numel = shape.numel()
        tensor = full_tensor[offset:offset + numel].reshape(shape)
        stats = analyze_tensor(name, tensor)
        stats["stream"] = stream_name
        results[stream_name].append(stats)
        offset += numel
        print(f"[{stream_name}] {len(results[stream_name]):4d} {name}")

# ----------------------------------------------------
# REPORT
# ----------------------------------------------------

print("\nWriting reports...")

for stream_name in streams.keys():
    report_path = os.path.join(OUTPUT_DIR, f"report_{stream_name}.txt")
    with open(report_path, "w") as f:
        for r in results[stream_name]:
            f.write("=" * 80 + "\n")
            f.write(f"STREAM: {stream_name}\n")
            f.write(f"Tensor: {r['name']}\n")
            f.write(f"Elements: {r['numel']:,}\n")
            f.write(f"Tensor Bytes: {r['tensor_bytes']:,}\n")
            f.write(f"Unique Values: {r['unique_count']:,}\n")
            f.write(f"Unique Ratio: {r['unique_ratio']:.6f}\n")
            f.write(f"Entropy: {r['entropy']:.4f} bits/symbol\n")
            f.write(f"Estimated Compressed Bytes: {r['estimated_compressed_bytes']:,.0f}\n")
            f.write(f"Histogram Bytes: {r['histogram_bytes']:,}\n")
            f.write(f"Histogram/Tensor Ratio: {r['histogram_ratio']:.4f}\n\n")

# ----------------------------------------------------
# SUMMARY
# ----------------------------------------------------

print("Writing summaries...")

for stream_name in streams.keys():
    r = results[stream_name]
    total_tensor_bytes = sum(x["tensor_bytes"] for x in r)
    total_hist_bytes   = sum(x["histogram_bytes"] for x in r)
    avg_unique_ratio = np.mean([x["unique_ratio"] for x in r])
    avg_entropy      = np.mean([x["entropy"] for x in r])
    largest_hist = max(r, key=lambda x: x["histogram_ratio"])
    smallest_hist = min(r, key=lambda x: x["histogram_ratio"])

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{stream_name}.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"GLOBAL SUMMARY: {stream_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tensors: {len(r)}\n")
        f.write(f"Average unique ratio: {avg_unique_ratio:.6f}\n")
        f.write(f"Average entropy: {avg_entropy:.4f}\n")
        f.write(f"Total tensor bytes: {total_tensor_bytes:,}\n")
        f.write(f"Total histogram bytes: {total_hist_bytes:,}\n")
        f.write(f"Histogram/Tensor ratio: {total_hist_bytes / total_tensor_bytes:.4f}\n\n")
        f.write(f"Largest histogram ratio:\n  {largest_hist['name']}\n  ratio={largest_hist['histogram_ratio']:.4f}\n\n")
        f.write(f"Smallest histogram ratio:\n  {smallest_hist['name']}\n  ratio={smallest_hist['histogram_ratio']:.4f}\n")

print("\nDone. Reports and summaries generated in:", OUTPUT_DIR)