import os
import sys
import argparse
import torch
import deepspeed
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

import time

# Manually add the build directory to the path at runtime
build_dir = "/usr/WS1/kogiou1/LLM_work/Checkers/build"
if build_dir not in sys.path:
    sys.path.append(build_dir)

import checkers_py
print("Successfully imported checkers_py!")


# 1. THE MOST IMPORTANT PART: Set PATH before importing DeepSpeed/Torch
venv_bin = "/usr/WS1/kogiou1/venvs/deepspeed_env/bin"
os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]

# Set compilers for the AMD nodes
os.environ["CC"] = "/usr/tce/bin/gcc"
os.environ["CXX"] = "/usr/tce/bin/g++"

# Set cache directories
os.environ["HF_HOME"] = "/p/vast1/kogiou1/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/p/vast1/kogiou1/hf_datasets/"
import shutil

class InstructionDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_length=1024, max_samples=12800):
        # yahma/alpaca-cleaned is a standard benchmark dataset from the original Alpaca paper
        self.raw_data = load_dataset(dataset_name, split="train")
        
        # Limit dataset size to ensure 3 epochs fit in ~1 hour
        if max_samples and max_samples < len(self.raw_data):
            self.raw_data = self.raw_data.select(range(max_samples))
            
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    def _format_prompt(self, instruction, input_text, output_text):
        """Formats into Llama 3 Chat template."""
        # Handle cases where 'input' might be empty (common in Alpaca)
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
        
        # REMOVE the "labels" key here. 
        # The DataCollatorForLanguageModeling will create it automatically
        # and pad it correctly to match the input_ids.
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    # Updated output path
    output_dir = "/p/lustre5/kogiou1/llama-3-8b/8_GPUs/instruction/DS"
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
                        # Added both global and local rank here
                        print(f"[Global Rank {model_engine.global_rank} | Local Rank {model_engine.local_rank}] "
                            f"DEBUG: Param {name} device: {param.ds_tensor.device}")
                        break

                start_time = time.perf_counter()

                checkers_py.initialize_context(
                    model_engine,
                    optimizer,
                    histogram_bins=4096,
                    background_threads=4,
                )

                end_time = time.perf_counter()
                # Added both global and local rank here
                print(f"[Global Rank {model_engine.global_rank} | Local Rank {model_engine.local_rank}] "
                    f"Context initialization time: {end_time - start_time:.4f} seconds")

            # --- SAVING LOGIC ---
            if model_engine.local_rank == 0:
                print(f"Step {model_engine.global_steps} complete. Loss: {loss.item():.4f}")
                print(f"--- Saving checkpoint after step {model_engine.global_steps}... ---")

            checkpoint_state = {"epoch": epoch, "global_step": int(model_engine.global_steps)}
            model_engine.save_checkpoint(output_dir, tag="step_1_checkpoint", client_state=checkpoint_state)

            break

    print("--- Training finished after 1 step. Checkpoint saved. ---")


if __name__ == "__main__":
    main()