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
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for model loading (default: cuda)"
    )
    args = parser.parse_args()

    # 1) Load base model in fp16
    print(f"Loading base model '{args.base_model}' in fp16...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )

    # 2) Attach LoRA adapters
    print(f"Loading LoRA adapters from '{args.lora_adapter}'...")
    peft = PeftModel.from_pretrained(
        base,
        args.lora_adapter,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    
    if args.scale != 1.0:
        for module in peft.modules():
            if isinstance(module, LoraLayer):
                module.scaling *= args.scale

    # 3) Merge adapters into base and unload
    print("Merging LoRA into base weights...")
    merged = peft.merge_and_unload()
        
    # 4) Save fused fp16 model
    print(f"Saving merged model to '{args.output_dir}'...")
    merged.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
