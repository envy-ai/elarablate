#!/usr/bin/env python3
"""
Adversarial QLoRA Finetuning Script (with per-step prefix recomputation)
...
Usage:
    python adversarial_qlora_finetune.py \
      --model_name_or_path llama-3.3 \
      --contexts_folder ./contexts_yaml \
      --output_dir ./lora_adv \
      --temperature 0.7 \
      --max_threshold 0.1 \
      --min_threshold 0.015 \
      --top_k 10 \
      --epochs 3 \
      --lr 1e-4 \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05
"""
import argparse
import os
import random
import torch
import torch.nn.functional as F
from colorist import Color, BrightColor
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import yaml # For YAML parsing
import re   # For regular expressions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--context_folders", required=True, nargs='+',
                        help="Folder containing context YAML files (one .yaml/.yml per file).  Will be searched recursively.")
    parser.add_argument("--output_dir", required=True)
    # Default hyperparameters (can be overridden by YAML)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--layer_range", type=str, default="59-79")
    parser.add_argument("--model_type", type=str, default="llama",
                        choices=["llama", "qwen"]) # Add more if needed
    parser.add_argument("--cache_quantized_model_dir", type=str, default=None,
                        help="Path to cache quantized model.  If not provided, model will not be cached after quantization.")
    parser.add_argument("--quant_type", type=str, default="nf4",
                        choices=["int8", "nf4", "fp8"],)
    

    parser.add_argument("--max_threshold_factor", type=float, default=1, help="Max threshold is 1 / top_k * max_threshold_factor.")
    parser.add_argument("--min_threshold_factor", type=float, default=.75, help="Min threshold is 1 / top_k * max_threshold_factor * min_threshold_factor.  This value should be less than 1.")
    
    args = parser.parse_args()

    # --- Prepare templates (could also be moved to YAML or a config file) ---
    instruct_templates = {
        "llama": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "$system$<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            "$user$<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "$response$"
        ),
        "qwen": ( # Example for Qwen, adjust as per actual Qwen format
            "<|im_start|>system\n$system$<|im_end|>\n"
            "<|im_start|>user\n$user$<|im_end|>\n"
            "<|im_start|>assistant\n$response$"
        )
        # Add other model type templates here
    }
    
    if args.model_type not in instruct_templates:
        raise ValueError(f"Unsupported model_type: {args.model_type}. Supported: {list(instruct_templates.keys())}")
    instruct_template = instruct_templates[args.model_type]

    # Default system/user messages (can be overridden by YAML)
    default_system_msg = '''You are an AI creative writing and roleplaying assistant.'''
    default_user_msg   = "Write a fantasy or sci-fi story."
    
    start_layer, end_layer = map(int, args.layer_range.split('-'))
    if start_layer > end_layer:
        raise ValueError(f"Invalid layer range: {args.layer_range}. Start layer must be less than or equal to end layer.")
    if start_layer < 0 or end_layer < 0:
        raise ValueError(f"Invalid layer range: {args.layer_range}. Start and end layers must be non-negative.")
    print(f"Using layer range: {start_layer}-{end_layer} for LoRA application.")

    # QLoRA bits-and-bytes config
    bnb_config = None
    if args.quant_type == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.quant_type == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif args.quant_type == "fp8":
        bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
        # FP8 specific configurations - these may vary based on implementation
        bnb_8bit_quant_type="fp8_e4m3fn",  # if supported directly
        bnb_8bit_compute_dtype=torch.float8_e4m3fn,  # PyTorch 2.1+ dtype
    )

    if args.model_type == "llama":
        model_class = LlamaForCausalLM
    # elif args.model_type == "qwen":
    #     from transformers import Qwen2ForCausalLM # Example
    #     model_class = Qwen2ForCausalLM
    else:
        # Fallback or error for unhandled model types for class loading
        print(f"Warning: Using LlamaForCausalLM for model_type '{args.model_type}'. Ensure this is correct.")
        model_class = LlamaForCausalLM


    base_model = model_class.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True
    )
    
    if args.cache_quantized_model_dir:
        # Save the quantized model to the specified directory
        os.makedirs(args.cache_quantized_model_dir, exist_ok=True)
        base_model.save_pretrained(args.cache_quantized_model_dir)
        # Save the tokenizer to the same directory
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        tokenizer.save_pretrained(args.cache_quantized_model_dir)
        
        print(f"Quantized model cached at: {args.cache_quantized_model_dir}")

    # Wrap with LoRA adapters
    # TODO: Make target_modules configurable, potentially based on model_type
    # This example is Llama specific. For other models, you'll need to find appropriate module names.
    if args.model_type == "llama": 
        num_layers = getattr(base_model.config, "num_hidden_layers", 80)
        print(f"Targeting LoRA for layers {start_layer} to {end_layer}")
        target_modules=[f"model.layers.{i}.self_attn.{proj}"
                    for i in range(start_layer, end_layer)
                    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]]
        # Add MLP layers if desired:
        # target_modules.extend([f"model.layers.{i}.mlp.{proj}"
        #                        for i in range(start_layer, num_layers)
        #                        for proj in ["gate_proj", "up_proj", "down_proj"]])
    elif args.model_type == "qwen": 
         num_layers = getattr(base_model.config, "num_hidden_layers", 32)
         target_modules=[f"transformer.h.{i}.attn.c_attn" for i in range(start_layer, end_layer)]
         target_modules.extend([f"transformer.h.{i}.attn.c_proj" for i in range(start_layer, end_layer)])
         # Qwen MLP:
         # target_modules.extend([f"transformer.h.{i}.mlp.w1" for i in range(start_layer, num_layers)])
         # target_modules.extend([f"transformer.h.{i}.mlp.w2" for i in range(start_layer, num_layers)])
         # target_modules.extend([f"transformer.h.{i}.mlp.c_proj" for i in range(start_layer, num_layers)])
    else:
        # Generic fallback or raise error
        print("Warning: No specific target_modules defined for this model type. LoRA might not be effective.")
        print("Consider adding specific target_modules for your model architecture.")
        # A common, but not universal, set of names. Might work for some models.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(f"Target modules for LoRA: {target_modules}")

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
    model.print_trainable_parameters()


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        # Check for common alternatives before setting to eos_token_id
        if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else: # Fallback, though unlikely for most models
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(f"Added new pad token: {tokenizer.pad_token_id}")
            # Important: resize model embeddings if a new token is added
            model.resize_token_embeddings(len(tokenizer))


    # Read and parse all context configurations from YAML files recursively
    context_configs = []
    print(f"Loading contexts recursively from: {args.context_folders}")
    
    # Walk through directory tree recursively
    for context_folder in args.context_folders:
        if not os.path.exists(context_folder):
            print(f"Warning: Context folder '{context_folder}' does not exist. Skipping.")
            continue
        if not os.path.isdir(context_folder):
            print(f"Warning: Context folder '{context_folder}' is not a directory. Skipping.")
            continue
        for root, dirs, files in os.walk(context_folder):
            # Sort files for consistent ordering
            for fname in sorted(files):
                if not (fname.lower().endswith(".yaml") or fname.lower().endswith(".yml")):
                    continue
                path = os.path.join(root, fname)
                
                # Create relative path for cleaner display
                rel_path = os.path.relpath(path, context_folder)
                
                try:
                    with open(path, 'r', encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                        # Compile regexes for efficiency
                        if "token_rules" in config and "bad_token_regexes" in config["token_rules"]:
                            config["token_rules"]["bad_token_regexes_compiled"] = [
                                re.compile(r) for r in config["token_rules"]["bad_token_regexes"]
                            ]
                        if "token_rules" in config and "good_token_regexes" in config["token_rules"]:
                            config["token_rules"]["good_token_regexes_compiled"] = [
                                re.compile(r) for r in config["token_rules"]["good_token_regexes"]
                            ]
                        context_configs.append(config)
                        print(f"  Loaded context: {rel_path}")
                except Exception as e:
                    print(f"Error loading or parsing YAML file {path}: {e}")
    
    if not context_configs:
        print(f"No YAML context files found in {args.context_folders} (searched recursively). Exiting.")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        epoch_losses = [] # Store losses for this epoch

        for i, ctx_config in enumerate(context_configs):
            # Get values from YAML, falling back to script defaults or arg defaults
            system_msg = ctx_config.get("system_prompt", default_system_msg).strip()
            user_msg = ctx_config.get("user_prompt", default_user_msg).strip()
            response_text = ctx_config.get("response_text", "").strip()
            hparams = ctx_config.get("hyperparameters", {})
            repeat = ctx_config.get("repeat", 1)
            current_temp = hparams.get("temperature", args.temperature)
            current_top_k = hparams.get("top_k", args.top_k)        
            current_max_threshold = hparams.get("max_threshold", 1 / current_top_k * args.max_threshold_factor)
            current_min_threshold = hparams.get("min_threshold", current_max_threshold * args.min_threshold_factor)

            token_rules = ctx_config.get("token_rules", {})
            bad_regexes = token_rules.get("bad_token_regexes_compiled", [])
            good_regexes = token_rules.get("good_token_regexes_compiled", [])
            
            force_good_tokens = token_rules.get("force_good_tokens", [])
            # Get the token IDs for force_good_tokens
            force_good_token_ids = []
            
            original_response_text = response_text  # Keep original for debugging
            
            # repeat x times:
            for _ in range(repeat):
                response_text = original_response_text  # Reset user_msg for each repeat
                # In user_msg, there may be random words specified like this: {red|blue|green|...}
                # We need to resolve those to actual tokens

                # Find all {word1|word2|...} patterns in user_msg (multiline support)
                pattern = r"\{([^}]+)\}"
                
                def replace_random_choice(match):
                    # Extract the content inside braces
                    content = match.group(1)
                    # Split by | and strip whitespace (including newlines)
                    options = [opt for opt in content.split('|')]
                    # Return a random choice, or empty string if no valid options
                    return random.choice(options) if options else ""
                
                response_text = re.sub(pattern, replace_random_choice, response_text, flags=re.DOTALL | re.MULTILINE)
                
                for token_str in force_good_tokens: 
                    token_id = tokenizer.encode(token_str, add_special_tokens=False)
                    if token_id:
                        force_good_token_ids.append(token_id[0])
                    else:
                        print(f"Warning: Token '{token_str}' not found in tokenizer vocabulary.")
                
                exclude_tokens = token_rules.get("exclude_tokens", [])
                # Get the token IDs for exclude_tokens
                exclude_token_ids = []
                for token_str in exclude_tokens:
                    token_id = tokenizer.encode(token_str, add_special_tokens=False)
                    if token_id:
                        exclude_token_ids.append(token_id[0])
                    else:
                        print(f"Warning: Token '{token_str}' not found in tokenizer vocabulary.")
                                    
                full_prompt = instruct_template \
                    .replace("$system$", system_msg) \
                    .replace("$user$", user_msg) \
                    .replace("$response$", response_text)

                # 1) Recompute prefix hidden states for this exact context
                model.eval() # Set to eval mode for inference part
                with torch.no_grad():
                    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False) # LLaMA adds bos by default
                    inputs = {k: v.to(model.device) for k,v in inputs.items()}
                    if inputs["input_ids"].shape[1] == 0:
                        print(f"Warning: Empty input for context {i} after tokenization. Skipping.")
                        continue
                    outputs = model(**inputs, use_cache=True)
                    pkv = outputs.past_key_values
                    last_id = inputs["input_ids"][0, -1].item()
                model.train() # Set back to train mode for gradient computation

                # 2) Get next-token distribution for that last_id
                with torch.no_grad(): # Still no grad for this part
                    id_tensor = torch.tensor([[last_id]], device=model.device)
                    out2 = model(input_ids=id_tensor, past_key_values=pkv)
                    logits = out2.logits[:, -1, :]
                    probs = F.softmax(logits / current_temp, dim=-1)[0]

                # 3) Identify high-prob tokens and apply regex rules
                top_probs, top_ids = torch.topk(probs, current_top_k)
                
                # Add forced tokens to top_probs and top_ids (prob doesn't matter, so set to zero)
                for token_id in force_good_token_ids:
                    if token_id not in top_ids:
                        top_probs = torch.cat((top_probs, torch.tensor([0.0], device=model.device)))
                        top_ids = torch.cat((top_ids, torch.tensor([token_id], device=model.device)))
                        
                # Remove excluded tokens from top_ids and top_probs
                if exclude_token_ids:
                    mask = torch.tensor([t not in exclude_token_ids for t in top_ids], device=model.device)
                    top_probs = top_probs[mask]
                    top_ids = top_ids[mask]
                if top_probs.numel() == 0:
                    print(f"Warning: No valid tokens left after exclusion for context {i}. Skipping.")
                    continue
                
                bad_token_indices_in_topk = []
                good_token_indices_in_topk = []

                print(f"\nContext {i+1}/{len(context_configs)} (System: '{system_msg[:30]}...', User: '{user_msg[:30]}...'):")
                print(f"Prompt ends with: ...'{response_text[-50:]}'")
                print(f"Using temp: {current_temp}, top_k: {current_top_k}, bad_thresh: {current_max_threshold}, good_thresh: {current_min_threshold}")
                # Print force good tokens, if any
                if force_good_token_ids:
                    print(f"  Force good tokens: {', '.join([tokenizer.decode([t]) for t in force_good_token_ids])}")

                for k_idx, (p, token_id_val) in enumerate(zip(top_probs.tolist(), top_ids.tolist())):
                    token_str = tokenizer.decode([token_id_val])
                    is_bad = False
                    is_good = False
                    is_force_good = False

                    # Check bad regexes first
                    for r in bad_regexes:
                        if r.search(token_str): # Using search to allow matches not at the beginning
                            is_bad = True
                            break
                    
                    # If not bad, check good regexes
                    if not is_bad:
                        for r in good_regexes:
                            if r.search(token_str):
                                is_good = True
                                break
                        
                    # Add force_good_tokens check
                    if token_id_val in force_good_token_ids:
                        is_good = True
                        is_force_good = True
                        is_bad = False # Force good takes precedence over bad
                    
                    marker = " "
                    if is_force_good and p < current_min_threshold:
                        good_token_indices_in_topk.append(token_id_val)
                        marker = f"{BrightColor.CYAN}↑{Color.OFF}" 
                    elif is_bad:
                        bad_token_indices_in_topk.append(token_id_val)
                        marker = f"{Color.RED}↓{Color.OFF}" 
                    elif is_good and p < current_min_threshold: 
                        good_token_indices_in_topk.append(token_id_val)
                        marker = f"{Color.GREEN}↑{Color.OFF}" 
                    elif is_good and p > current_max_threshold: 
                        bad_token_indices_in_topk.append(token_id_val)
                        marker = f"{Color.GREEN}↓{Color.OFF}"
                    else:
                        marker = " "

                    print(f" {marker} [{token_id_val}] '{token_str}': {p:.4f}")

                if not bad_token_indices_in_topk and not good_token_indices_in_topk:
                    print("  -- no tokens for adversarial update based on rules and thresholds, skipping backprop for this step.")
                    continue

                # 4) Adversarial update (negative-loss for bad, positive-loss for good)
                optimizer.zero_grad()
                
                # Re-run the forward pass for the last token to get logits with gradients
                id_tensor_grad = torch.tensor([[last_id]], device=model.device)
                out3 = model(input_ids=id_tensor_grad, past_key_values=pkv)
                log_probs = F.log_softmax(out3.logits[:, -1, :], dim=-1)

                step_loss_terms = []
                for bad_idx_val in bad_token_indices_in_topk:
                    # Penalize: NLL loss, so log_prob is positive for minimization (make prob smaller)
                    loss_term = log_probs[0, bad_idx_val]
                    step_loss_terms.append(loss_term)
                    print(f"  Penalizing token: {tokenizer.decode([bad_idx_val])} (raw log_prob: {loss_term.item():.4f})")

                for good_idx_val in good_token_indices_in_topk:
                    # Reward: Negative NLL loss, so -log_prob is positive for minimization (make prob_larger)
                    loss_term = -log_probs[0, good_idx_val]
                    step_loss_terms.append(loss_term)
                    print(f"  Rewarding token: {tokenizer.decode([good_idx_val])} (raw -log_prob: {loss_term.item():.4f})")
                
                if not step_loss_terms:
                    print("  -- no loss terms generated for this step after all. Skipping backprop.")
                    continue

                current_step_loss = torch.stack(step_loss_terms).sum()
                
                # Normalize loss by number of terms to make learning rate less sensitive to 
                # of bad/good tokens
                # Note: I'm not convinced this is actually a good idea, because it amplifies
                # loss that's not the average of larger numbers of terms, which could have rough
                # edges.  Also, from the standpoint of each token being a separate loss term,
                # it doesn't make sense to average them.  Disabling for now.
                #current_step_loss = current_step_loss.mul(1.0 / len(step_loss_terms))             
                
                print(f"  Step loss: {current_step_loss.item():.4f} for {len(step_loss_terms)} tokens")
                current_step_loss.backward()
                optimizer.step()
                
                epoch_losses.append(current_step_loss.item())

        if epoch_losses:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"--- Epoch {epoch} Average Loss: {avg_epoch_loss:.4f} ---")
        else:
            print(f"--- Epoch {epoch} No losses recorded ---")
            
    # Print out names of all lora layers
    # print("\nLoRA layers:")
    # for name, param in model.named_parameters():
    #     if "lora" in name:
    #         print(f"  {name}: {param.size()}")
    # Save LoRA adapters
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"→ Adapters saved to {args.output_dir}")

    # Save tokenizer if it was modified (e.g., new pad token)
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"→ Tokenizer saved to {args.output_dir}")


    print("\nTraining complete.")

if __name__ == "__main__":
    main()