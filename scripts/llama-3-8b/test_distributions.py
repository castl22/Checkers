import os, torch, numpy as np, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis
from multiprocessing import Pool, cpu_count
import functools

# ----------------------------
# CONFIG
# ----------------------------
checkpoint_dir = "/p/lustre5/kogiou1/llama-3-8b/8_GPUs/math/DS/epoch_0"
model_path     = os.path.join(checkpoint_dir, "zero_pp_rank_0_mp_rank_00_model_states.pt")
optim_path     = os.path.join(checkpoint_dir, "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")
output_root    = "./results"
RANK           = 0
NUM_RANKS      = 8
NUM_WORKERS    = cpu_count()  # or set manually, e.g. 32

# ----------------------------
# HELPERS (must be top-level for pickling)
# ----------------------------
def categorize_tensor(name):
    n = name.lower()
    if any(x in n for x in ["q_proj", "query"]):   return "query"
    if any(x in n for x in ["k_proj", "key"]):     return "key"
    if any(x in n for x in ["v_proj", "value"]):   return "value"
    if any(x in n for x in ["o_proj", "output"]):  return "attn_output"
    if any(x in n for x in ["layernorm", "norm"]): return "layernorm"
    if "gate_proj" in n: return "mlp_gate"
    if "up_proj"   in n: return "mlp_up"
    if "down_proj" in n: return "mlp_down"
    return "other"

def compute_stats(x_np):
    if x_np.size == 0:
        return {k: float('nan') for k in ["mean","std","median","iqr","skew","kurtosis"]}
    return {
        "mean":     float(np.mean(x_np)),
        "std":      float(np.std(x_np)),
        "median":   float(np.median(x_np)),
        "iqr":      float(np.percentile(x_np,75) - np.percentile(x_np,25)),
        "skew":     float(skew(x_np)),
        "kurtosis": float(kurtosis(x_np)),
    }

def plot_kde_ecdf(x_np, title, save_path):
    if x_np.size < 2 or np.all(x_np == x_np[0]): return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    try:
        kde = gaussian_kde(x_np)
        xs  = np.linspace(x_np.min(), x_np.max(), 300)
        axes[0].plot(xs, kde(xs))
        axes[0].set_title(f"KDE - {title}")
    except:
        axes[0].set_title("KDE Failed")
    xs_sorted = np.sort(x_np)
    axes[1].plot(xs_sorted, np.arange(len(xs_sorted)) / len(xs_sorted))
    axes[1].set_title(f"ECDF - {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=80)  # lower dpi = faster save
    plt.close()

def process_one_param(task):
    """Worker function — receives pre-sliced numpy arrays to avoid pickling tensors."""
    param_name, cat, safe_name, step_dir, slices_np = task
    results = []
    for key, x_np in slices_np.items():
        out_dir   = os.path.join(step_dir, key, cat)
        os.makedirs(out_dir, exist_ok=True)
        stats     = compute_stats(x_np)
        plot_path = os.path.join(out_dir, f"{safe_name}.png")
        plot_kde_ecdf(x_np, param_name, plot_path)
        stats.update({"name": param_name, "key": key, "category": cat})
        results.append(stats)
    return param_name, results

# ----------------------------
# LOAD
# ----------------------------
print("[INFO] Loading checkpoints...")
model_state = torch.load(model_path,  map_location="cpu", weights_only=False)
optim_state = torch.load(optim_path,  map_location="cpu", weights_only=False)

opt_sd      = optim_state['optimizer_state_dict']
inner       = opt_sd['optimizer_state_dict']
exp_avg     = inner['state'][0]['exp_avg']
exp_avg_sq  = inner['state'][0]['exp_avg_sq']
fp32_flat   = opt_sd['fp32_flat_groups'][0]
flat_numel  = fp32_flat.numel()
param_shapes = model_state['param_shapes'][0]

# ----------------------------
# SANITY CHECK
# ----------------------------
total_numel = sum(s.numel() for s in param_shapes.values())
print(f"Total model numel:  {total_numel:,}")
print(f"Flat buffer numel:  {flat_numel:,}")
print(f"Ratio (expect {NUM_RANKS}):  {total_numel / flat_numel:.2f}")

rank_start = RANK * flat_numel
rank_end   = rank_start + flat_numel

# ----------------------------
# BUILD TASK LIST (main process — fast tensor slicing before forking)
# ----------------------------
print("[INFO] Building task list...")
step_dir = os.path.join(output_root, "step_0")
os.makedirs(step_dir, exist_ok=True)

tasks        = []
global_offset = 0

for param_name, shape in param_shapes.items():
    numel   = shape.numel()
    p_start = global_offset
    p_end   = global_offset + numel

    overlap_start = max(p_start, rank_start)
    overlap_end   = min(p_end,   rank_end)

    if overlap_start < overlap_end:
        local_offset = overlap_start - rank_start
        local_numel  = overlap_end - overlap_start
        cat       = categorize_tensor(param_name)
        safe_name = param_name.replace("/", "_").replace(".", "_")

        # Convert to numpy HERE in the main process — avoids pickling torch tensors
        slices_np = {
            "exp_avg":        exp_avg.narrow(0, local_offset, local_numel).float().numpy().copy(),
            "exp_avg_sq":     exp_avg_sq.narrow(0, local_offset, local_numel).float().numpy().copy(),
            "master_weights": fp32_flat.narrow(0, local_offset, local_numel).detach().cpu().float().numpy().copy(),
        }
        tasks.append((param_name, cat, safe_name, step_dir, slices_np))

    global_offset += numel

print(f"[INFO] {len(tasks)} params to process across {NUM_WORKERS} workers...")

# ----------------------------
# PARALLEL EXECUTION
# ----------------------------
all_stats = []
with Pool(processes=NUM_WORKERS) as pool:
    for i, (param_name, stats) in enumerate(pool.imap_unordered(process_one_param, tasks)):
        all_stats.extend(stats)
        if i % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] done: {param_name}")

print(f"\n[INFO] Done. Processed {len(tasks)} parameters.")