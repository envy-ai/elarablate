#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM
from peft.tuners.lora import LoraLayer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA into a base model (fp16) and save fused fp16 checkpoint"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or HF repo ID of the original base model"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="Path to the directory containing your trained LoRA adapters"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the merged fp16 model"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Global LoRA scale (default: 1.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for model loading (default: cuda)"
    )
    args = parser.parse_args()
    
    print (f"Using device: {args.device}")
    
    dtype = torch.float16
    print (f"Using dtype: {dtype}")

    # 1) Load base model in fp16
    print(f"Loading base model '{args.base_model}' in fp16...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device,
    )
    
    # Print names of all modules in the model
    print("Base model modules:")
    for name, module in base.named_modules():
        print(f"  {name}: {module.__class__.__name__}")

    # 2) Attach LoRA adapters
    print(f"Loading LoRA adapters from '{args.lora_adapter}'...")
    peft = PeftModel.from_pretrained(
        base,
        args.lora_adapter,
        torch_dtype=dtype,
        device_map=args.device,
    )
    # Print names of all modules in the LoRA model
    print("LoRA model modules:")
    for name, module in peft.named_modules():
        print(f"  {name}: {module.__class__.__name__}")

    # 3) Merge adapters into base and unload
    print("Merging LoRA into base weights...")
    #merged = peft.merge_and_unload()
    # Merge this fucker manually to make sure it doesn't just stick the lora layers into the model
    merged = base
    for name, module in peft.named_modules():
        if isinstance(module, LoraLayer):
            # Get the base weight
            base_weight = getattr(base, name)
            # Get the LoRA weight
            lora_weight = module.weight
            # Merge the weights
            merged_weight = base_weight + args.scale * lora_weight
            # Set the merged weight back to the base model
            setattr(merged, name, merged_weight)

    # Unload the LoRA model
    module.unload()

    # 4) Save fused fp16 model
    print(f"Saving merged model to '{args.output_dir}'...")
    merged.save_pretrained(args.output_dir, torch_dtype=torch.float16)
    print("Done.")

if __name__ == "__main__":
    main()
