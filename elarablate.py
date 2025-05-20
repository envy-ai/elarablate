#!/usr/bin/env python3
"""
Adversarial QLoRA Finetuning Script (with per-step prefix recomputation)
...
Usage:
    python adversarial_qlora_finetune.py \
      --model_name_or_path llama-3.3 \
      --contexts_folder ./contexts \
      --output_dir ./lora_adv \
      --temperature 0.7 \
      --threshold 0.1 \
      --top_k 10 \
      --epochs 3 \
      --lr 1e-4 \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05
"""
import argparse
import os
import torch
import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--contexts_folder", required=True,
                        help="Folder containing contexts (one .txt per file).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--low_threshold", type=float, default=0.015)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--model_type", type=str, default="llama",
                        choices=["llama", "qwen"])
    args = parser.parse_args()

    # --- Prepare templates & tokenizer/model ---
    instruct_template = ""
    if args.model_type == "llama":
        instruct_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "$system$<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            "$user$<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "$response$"
        )
    system_msg = "You are a story writing and roleplaying AI."
    user_msg   = "Write a fantasy or sci-fi story."

    # QLoRA bits-and-bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load base model in 4-bit
    base_model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    # Wrap with LoRA adapters
    
    target_modules=[f"model.layers.{i}.self_attn.{proj}" 
                for i in range(60, 79)  # Target layers 16-25 for conceptual changes
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]]
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Read and format all contexts once
    contexts = []
    for fname in sorted(os.listdir(args.contexts_folder)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(args.contexts_folder, fname)
        text = open(path, encoding="utf-8").read().strip()
        full = instruct_template\
            .replace("$system$", system_msg)\
            .replace("$user$", user_msg)\
            .replace("$response$", text)
        contexts.append(full)
        
    # TODO: Contexts should be yaml files, specifying system, user, response, top_k, 
    # temperature, threshold, etc, as well as regexps for designating good and bad tokens

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training loop, recomputing prefix per step ---
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        for ctx in contexts:
            # 1) Recompute prefix hidden states for this exact context
            model.eval()
            with torch.no_grad():
                inputs = tokenizer(ctx, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(model.device) for k,v in inputs.items()}
                outputs = model(**inputs, use_cache=True)
                pkv = outputs.past_key_values
                last_id = inputs["input_ids"][0, -1].item()
            model.train()

            # 2) Get next-token distribution for that last_id
            with torch.no_grad():
                id_tensor = torch.tensor([[last_id]], device=model.device)
                out2 = model(input_ids=id_tensor, past_key_values=pkv)
                logits = out2.logits[:, -1, :]
                probs = F.softmax(logits / args.temperature, dim=-1)[0]

            # 3) Identify high-prob tokens
            top_probs, top_ids = torch.topk(probs, args.top_k)
            has_mask = top_probs > args.threshold
            has_mask_good = top_probs < args.low_threshold
            
            # Iterate through top-k tokens ane add any token that starts with a non-space character
            # to the list of tokens to be masked
            for i, (p, idx) in enumerate(zip(top_probs.tolist(), top_ids.tolist())):
                token = tokenizer.decode([idx])
                if token[0] != " ":
                    has_mask[i] = True
                    has_mask_good[i] = False
                       
            print(f"\nContext snippet: {ctx[:75]!r}...") 
            for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
                mark = "*" if p > args.threshold else " "
                print(f" {mark} [{idx}] '{tokenizer.decode([idx])}': {p:.4f}")
                
            if not has_mask.any() and not has_mask_good.any():
                print("  -- no tokens above threshold, skipping adversarial update")
                continue

            # 4) Adversarial (negative-loss) update for each offending token
            #for bad_idx in top_ids[has_mask]:

            optimizer.zero_grad()
            id_tensor = torch.tensor([[last_id]], device=model.device)
            out3 = model(input_ids=id_tensor, past_key_values=pkv)
            log_probs = F.log_softmax(out3.logits[:, -1, :], dim=-1)
            losses = []
      
            for bad_idx in top_ids[has_mask]:
                losses.append(log_probs[0, bad_idx])
            for good_idx in top_ids[has_mask_good]:
                losses.append(-log_probs[0, good_idx])
            loss = torch.stack(losses).sum()    

            # Iterate over losses and do .backward
            loss.backward()               # negative to lower its probability
            optimizer.step()
            
            
            
    # Save LoRA adapters
    model.save_pretrained(args.output_dir)
    print(f"→ Adapters saved to {args.output_dir}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
