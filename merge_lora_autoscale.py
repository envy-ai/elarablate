#!/usr/bin/env python3
"""
LoRA Merge with Per-Layer NF4→FP16 Calibration
Usage: python merge_lora.py \
        --input_path INPUT_DIR \
        --lora_path LORA_DIR \
        --output_path OUTPUT_DIR \
        --nf4_model_path NF4_MODEL_DIR \
    [--no-gpu] [--scale SCALE] [--layer_range START-END]
"""
import argparse
import os
import re
import shutil
import json
from pathlib import Path
from tqdm import tqdm

import safetensors
import torch
from bitsandbytes.nn import Linear4bit
import bitsandbytes.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import peft

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str, help='Path to the FP16 model directory (HF format).')
parser.add_argument('lora_path', type=str, help='Path to the LoRA adapter directory.')
parser.add_argument('output_path', type=str, help='Path to write merged safetensors shards.')
parser.add_argument('--nf4_model_path', type=str, required=True,
                    help='Path to the NF4-quantized model directory (HF format) used during LoRA training.')
parser.add_argument('--no-gpu', action='store_true', help='Use CPU for merging.')
parser.add_argument('--scale', type=float, default=1.0,
                    help='Additional global multiplier on LoRA ΔW (default=1).')
parser.add_argument('--layer_range', type=str, default='59-79',
                    help='Range of layers to apply LoRA (e.g. 59-79).')
args = parser.parse_args()

# Parse layer range
start_layer, end_layer = map(int, args.layer_range.split('-'))
if start_layer > end_layer or start_layer < 0:
    raise ValueError(f"Invalid layer_range: {args.layer_range}")

input_path = Path(args.input_path)
lora_path = Path(args.lora_path)
output_path = Path(args.output_path)
output_path.mkdir(parents=True, exist_ok=True)

device = 'cpu' if args.no_gpu else 'cuda'

# Load LoRA config
lora_config = peft.LoraConfig.from_json_file(lora_path / 'adapter_config.json')
print(f"LoRA config: {lora_config}")
base_scale = (lora_config['lora_alpha'] / lora_config['r']) * args.scale
print(f"Base LoRA scale (α/r * user_scale): {base_scale:.6f}")

# Load models for calibration
print('Loading FP16 base model for calibration...')
fp16_base = AutoModelForCausalLM.from_pretrained(
    input_path, torch_dtype=torch.float16,
    device_map='cpu', low_cpu_mem_usage=True
)

print('Loading NF4-quantized base model for calibration...')
# Load the quantized model properly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

nf4_base = AutoModelForCausalLM.from_pretrained(
    args.nf4_model_path,
    quantization_config=bnb_config,
    device_map='cpu',
    torch_dtype=torch.float16,
)
# Compute per-module norm ratios with proper dequantization
layer_ratios = {}
print('Computing per-layer norm ratios...')

for tm in tqdm(lora_config['target_modules'], desc="Computing ratios"):
    try:
        # Get the modules
        mod_fp16 = fp16_base.get_submodule(tm)
        mod_nf4 = nf4_base.get_submodule(tm)
        
        # Get FP16 weight
        w_fp16 = mod_fp16.weight.data.float()
        
        # Dequantize NF4 weight properly
        if isinstance(mod_nf4, Linear4bit):
            # For 4-bit linear layers, we need to dequantize
            # Move everything to the same device (CPU in this case)
            weight_data = mod_nf4.weight.data
            quant_state = mod_nf4.weight.quant_state
            
            # Ensure all components are on CPU
            if hasattr(quant_state, 'to'):
                quant_state = quant_state.to('cpu')
            
            # Alternative approach: manually dequantize
            # This avoids the device mismatch issue
            if hasattr(mod_nf4.weight, 'dequantize'):
                # Use the built-in method if available
                w_nf4 = mod_nf4.weight.dequantize().to('cpu').float()
            else:
                # Manual dequantization for NF4
                # Get the scale and zero point from quant_state
                if hasattr(quant_state, 'absmax'):
                    scale = quant_state.absmax / 127.0  # NF4 uses 127 as max
                    # Simple dequantization (this is approximate)
                    w_nf4 = (weight_data.float() - 8) * scale.float()
                else:
                    # If we can't properly dequantize, just use a default ratio
                    print(f"  Warning: Cannot properly dequantize {tm}, using default ratio")
                    layer_ratios[tm] = 1.0
                    continue
        else:
            # If not quantized, use as-is
            w_nf4 = mod_nf4.weight.data.float()
        
        # Compute ratio
        norm_fp16 = w_fp16.norm().item()
        norm_nf4 = w_nf4.norm().item()
        ratio = norm_fp16 / (norm_nf4 + 1e-12)
        
        # Sanity check
        if 0.1 < ratio < 10.0:  # Reasonable range
            layer_ratios[tm] = ratio
            print(f"  {tm}: ratio {ratio:.4f} (FP16: {norm_fp16:.2f}, NF4: {norm_nf4:.2f})")
        else:
            print(f"  WARNING: {tm}: extreme ratio {ratio:.4f}, defaulting to 1.0")
            layer_ratios[tm] = 1.0
            
    except Exception as e:
        print(f"  Error computing ratio for {tm}: {e}")
        layer_ratios[tm] = 1.0

# --- Prepare LoRA state ---
print('Loading LoRA weights...')
if (lora_path / 'adapter_model.safetensors').exists():
    lora_state = safetensors.torch.load_file(lora_path / 'adapter_model.safetensors')
    if not args.no_gpu:
        for k, v in tqdm(lora_state.items()):
            lora_state[k] = v.to(device)
else:
    lora_state = torch.load(lora_path / 'adapter_model.bin', map_location=device)

# Helper to extract A and B

def find_lora_weights(key):
    a = b = None
    prefix = key.strip('.weight')
    for lk, lw in lora_state.items():
        if prefix in lk:
            if 'lora_A' in lk:
                a = lw
            elif 'lora_B' in lk:
                b = lw
    assert (a is None) == (b is None), f"Missing A/B for {prefix}"
    return a, b

# Identify shards
shards = sorted(input_path.glob('*.safetensors'))
index_file = input_path / 'model.safetensors.index.json'
lora_shards = []
if index_file.exists():
    idx = json.load(open(index_file, 'r'))['weight_map']
    for layer, shard in idx.items():
        if any(f'layers.{i}' in layer for i in range(start_layer, end_layer+1)):
            lora_shards.append(shard)

# Copy non-mergeable files
print('Copying non-mergeable files...')
for fp in input_path.iterdir():
    if fp.name not in [s.name for s in shards] and not fp.is_dir():
        shutil.copy(fp, output_path / fp.name)

# --- Merging ---
print('Merging LoRA into FP16 shards with per-layer calibration...')
found = 0
for shard in tqdm(shards, desc='Shards'):
    tensors = {}
    if shard.name in lora_shards:
        with safetensors.safe_open(shard, framework='pt', device=device) as f:
            meta = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)
                lkey = re.sub(r'^language_model\.', '', key)
                match = re.search(r'layers\.(\d+)', lkey)
                if match:
                    layer = int(match.group(1))
                    if start_layer <= layer <= end_layer:
                        A, B = find_lora_weights(lkey)
                        if A is not None:
                            # determine matching target module
                            mod_key = next((tm for tm in layer_ratios if tm in lkey), None)
                            lr = layer_ratios.get(mod_key, 1.0)
                            actual_scale = base_scale * lr
                            wt = tensor.to(torch.float32)
                            delta = (B.to(torch.float32) @ A.to(torch.float32)) * actual_scale
                            tensor = (wt + delta).to(tensor.dtype)
                            found += 1
                tensors[key] = tensor
        safetensors.torch.save_file(tensors, output_path / shard.name, metadata=meta)
    else:
        shutil.copy(shard, output_path / shard.name)
print(f"Applied LoRA to {found} tensors.")
print('Done.')
