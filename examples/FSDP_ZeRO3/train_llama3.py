import os
import sys
import argparse
import torch
import deepspeed
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import shutil
import time

# Manually add the build directory to the path at runtime
build_dir = "/usr/WS1/kogiou1/LLM_work/Checkers/build"
if build_dir not in sys.path:
    sys.path.append(build_dir)

import checkers_py
print("Successfully imported checkers_py!")


def rank_log(model_engine, message):
    print(
        f"[Global Rank {model_engine.global_rank} | Local Rank {model_engine.local_rank}] {message}",
        flush=True,
    )


def sync_gpu_and_report(model_engine, stage):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=model_engine.device)
    rank_log(model_engine, f"SYNC_OK stage={stage}")


def dist_barrier_and_report(model_engine, stage):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        rank_log(model_engine, f"BARRIER_OK stage={stage}")


# 1. THE MOST IMPORTANT PART: Set PATH before importing DeepSpeed/Torch
venv_bin = "/usr/WS1/kogiou1/venvs/deepspeed_env/bin"
os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]

# Set compilers for the AMD nodes
os.environ["CC"] = "/usr/tce/bin/gcc"
os.environ["CXX"] = "/usr/tce/bin/g++"

# Set cache directories
os.environ["HF_HOME"] = "/tmp/kogiou1/hf_cache" 
os.environ["HF_DATASETS_CACHE"] = "/tmp/kogiou1/hf_cache"

# Ensure the directory exists
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

import os
import sys
import argparse
import torch
import deepspeed
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch.distributed as dist


# ----------------------------
# Runtime build import
# ----------------------------
build_dir = "/usr/WS1/kogiou1/LLM_work/Checkers/build"
if build_dir not in sys.path:
    sys.path.append(build_dir)

import checkers_py
print("Successfully imported checkers_py!")


# ----------------------------
# Logging helpers
# ----------------------------
def rank_log(model_engine, message):
    print(
        f"[Global Rank {model_engine.global_rank} | Local Rank {model_engine.local_rank}] {message}",
        flush=True,
    )


# ----------------------------
# Environment
# ----------------------------
venv_bin = "/usr/WS1/kogiou1/venvs/deepspeed_env/bin"
os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]

os.environ["CC"] = "/usr/tce/bin/gcc"
os.environ["CXX"] = "/usr/tce/bin/g++"


os.environ["HF_HOME"] = "/p/vast1/kogiou1/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/p/vast1/kogiou1/hf_cache"
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)


class InstructionDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_length=1024, max_samples=12800):

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load ONLY on rank 0
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            raw_data = load_dataset(dataset_name, split="train")

            if max_samples and max_samples < len(raw_data):
                raw_data = raw_data.select(range(max_samples))
        else:
            raw_data = None

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # SAFE: broadcast only metadata, not full HF object (better pattern)
        if dist.is_initialized():
            obj_list = [raw_data]
            dist.broadcast_object_list(obj_list, src=0)
            self.raw_data = obj_list[0]
        else:
            self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def _format_prompt(self, instruction, input_text, output_text):
        user_query = instruction
        if input_text and len(input_text.strip()) > 0:
            user_query = f"{instruction}\n\nContext: {input_text}"

        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
        )

    def __getitem__(self, idx):
        example = self.raw_data[idx]

        text = self._format_prompt(
            example.get("instruction", ""),
            example.get("input", ""),
            example.get("output", "")
        )

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    # Updated output path
    output_dir = "/p/lustre5/kogiou1/llama-3-8b/8_GPUs/instruction/DS/step_1_ANS"
    os.makedirs(output_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="DeepSpeed Llama 3 Instruction Fine-tuning")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        use_cache=False
    )

    # 3. Dataset (Alpaca Cleaned - 12.8k subset)
    dataset = InstructionDataset("yahma/alpaca-cleaned", tokenizer, max_length=1024, max_samples=12800)
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 4. Initialize DeepSpeed
    model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        collate_fn=collate_fn
    )


    model_engine.train()

    # Run until at least one real optimizer step has completed.
    # With gradient_accumulation_steps > 1, model_engine.step() is a no-op for
    # gradient accumulation micro-steps; the inner FusedAdam only runs (and
    # populates its state dict) on the boundary step.  We therefore loop until
    # model_engine.global_steps >= 1 before calling initialize_context.
    context_initialized = False
    for epoch in range(1):
        for step, batch in enumerate(training_dataloader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            outputs = model_engine(**batch)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            # global_steps is incremented only after a real optimizer update.
            if model_engine.global_steps < 1:
                continue

            if not context_initialized:
                context_initialized = True

                for name, param in model_engine.module.named_parameters():
                    if hasattr(param, "ds_tensor") and param.ds_tensor is not None:
                        rank_log(model_engine, f"DEBUG: Param {name} device: {param.ds_tensor.device}")
                        break

                start_time = time.perf_counter()

                checkers_py.initialize_context(
                    model_engine,
                    optimizer,
                    histogram_bins=4096,
                    background_threads=4,
                )

                end_time = time.perf_counter()
                rank_log(model_engine, f"Context initialization time: {end_time - start_time:.4f} seconds")

            sync_gpu_and_report(model_engine, "pre_analyze")
            dist_barrier_and_report(model_engine, "pre_analyze")

            start_time = time.perf_counter()
            rank_log(model_engine, "ENTER analyze_tensors")
            checkers_py.analyze_tensors()
            end_time = time.perf_counter()
            rank_log(model_engine, f"Tensor analysis time: {end_time - start_time:.4f} seconds")

            sync_gpu_and_report(model_engine, "post_analyze")
            dist_barrier_and_report(model_engine, "post_analyze")

            # --- SAVING LOGIC ---
            if model_engine.local_rank == 0:
                print(f"Step {model_engine.global_steps} complete. Loss: {loss.item():.4f}")
                print(f"--- Saving checkpoint after step {model_engine.global_steps}... ---")

            checkpoint_state = {"epoch": epoch, "global_step": int(model_engine.global_steps)}
            rank_log(model_engine, "ENTER save_checkpoint")
            checkers_py.compress_and_save(output_dir)
            # model_engine.save_checkpoint(output_dir, tag="step_1_checkpoint", client_state=checkpoint_state)
            rank_log(model_engine, "EXIT save_checkpoint")

            sync_gpu_and_report(model_engine, "post_save_checkpoint")
            dist_barrier_and_report(model_engine, "post_save_checkpoint")

            break

    print("--- Training finished after 1 step. Checkpoint saved. ---")


if __name__ == "__main__":
    main()