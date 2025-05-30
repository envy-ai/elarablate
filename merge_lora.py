# Usage: python merge_lora.py input_path lora_path output_path
# Output path is created if it doesn't exist

'''
MIT License

Copyright (c) 2023 tdrussell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import os
import re
import shutil
from pathlib import Path
import numpy as np

import safetensors
import torch
import json
from tqdm import tqdm

import peft


parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str, help='The path to the input directory.')
parser.add_argument('lora_path', type=str, help='The path to the LoRA directory.')
parser.add_argument('output_path', type=str, help='The path to the output directory.')
parser.add_argument('--no-gpu', action='store_true', help='Use CPU for merging. Probably unnecessary unless each individual shard is too large to fit in GPU memory.')
parser.add_argument('--scale', type=float, default=2.0, help='LoRA scale to apply (2 seems to work well).')
parser.add_argument('--layer_range', type=str, default='59-79', help='Range of layers to apply LoRA to. Default is 59-79, for 70B llama models.')
args = parser.parse_args()

start_layer, end_layer = map(int, args.layer_range.split('-'))
if start_layer > end_layer:
    raise ValueError(f'Invalid layer range: {args.layer_range}. Start layer must be less than or equal to end layer.')
if start_layer < 0 or end_layer < 0:
    raise ValueError(f'Invalid layer range: {args.layer_range}. Start and end layers must be non-negative.')

input_path, lora_path, output_path = Path(args.input_path), Path(args.lora_path), Path(args.output_path)
os.makedirs(output_path, exist_ok=True)

lora_config = peft.LoraConfig.from_json_file(lora_path / 'adapter_config.json')
scale = lora_config['lora_alpha'] / lora_config['r'] * args.scale

print(f'Using LoRA scale: {scale}')

device = 'cpu' if args.no_gpu else 'cuda'

print('Loading LoRA model...')

# Check if we have adapter_model.bin or adapter_model.safetensors
if (lora_path / 'adapter_model.safetensors').exists():
    lora_state = safetensors.torch.load_file(lora_path / 'adapter_model.safetensors')
    if not args.no_gpu:
        # Move mapped entries to cuda
        for key, value in tqdm(lora_state.items()):
            lora_state[key] = value.to('cuda')
else:
    lora_state = torch.load(lora_path / 'adapter_model.bin', map_location=device)


def find_lora_weights(key):
    lora_A = None
    lora_B = None
    for lora_key, lora_weight in lora_state.items():
        if key.strip('.weight') in lora_key:
            if 'lora_A' in lora_key:
                lora_A = lora_weight
            elif 'lora_B' in lora_key:
                lora_B = lora_weight
            else:
                raise RuntimeError()
    assert not ((lora_A is None) ^ (lora_B is None))
    return lora_A, lora_B

def find_lora_weights(key):
    a = b = None
    prefix = key.replace('.weight', '')
    for lk, lw in lora_state.items():
        if prefix in lk:
            if 'lora_A' in lk:
                a = lw
            elif 'lora_B' in lk:
                b = lw
    assert (a is None) == (b is None), f"Missing LoRA A or B for {prefix}"
    return a, b

shards = []
for shard in input_path.glob('model*.safetensors'):
    shards.append(shard)

lora_shards = []
    
# Read in model.safetensors.index.json
index_path = input_path / 'model.safetensors.index.json'
if index_path.exists():
    with open(index_path, 'r') as f:
        indexdata = json.load(f)
        
    # dump indexdata to see what it looks like
    print(json.dumps(indexdata['weight_map'], indent=4))
        
    # Only add shards to the list that contain layers within the range
    for layer, shard in indexdata['weight_map'].items():
        if shard.endswith('.safetensors'):
            # Check if the shard contains layers within the range
            if any(f'layers.{i}' in layer for i in range(start_layer, end_layer + 1)):
                lora_shards.append(shard)

print('Copying unmergable files to output')
for filepath in input_path.glob('*'):
    if filepath in shards:
        continue
    filepath = Path(filepath)
    if filepath.is_dir():
        continue
    if filepath.suffix == '.gguf':
        # Skip unrelated stray quantizations
        continue
    if filepath.suffix == '.safetensors':
        # Consolidated, possibly
        continue
    print(f'copying {filepath.name} to output')
    shutil.copy(filepath, output_path)

print('Merging and copying state_dict to output')
found = 0
strengths_l2 = []
strengths_l1 = []
for shard in (pbar := tqdm(shards)):
    tensors = {}
    if shard.name in lora_shards:
        print(f'Found {shard.name} in lora_shards')
        with safetensors.safe_open(shard, framework='pt', device=device) as f:
            metadata = f.metadata()
            for key in f.keys():
                lora_key = re.sub(r'^language_model\.', '', key)
                # Check if the key is in the range of layers to apply LoRA
                # Find 'layers.##' in the key
                
                tensor = f.get_tensor(key)
                print(f'Processing key: {key}')
                
                match = re.search(r'layers\.(\d+)', lora_key)
                if match:
                    layer_num = int(match.group(1))
                    if layer_num < start_layer or layer_num > end_layer:
                        print(f'Skipping because {layer_num} is outside of range') # Skip this key if 
                    else:
                        lora_A, lora_B = find_lora_weights(lora_key)
                        if lora_A is not None:
                            print(f'Found LoRA weights for {key}: {lora_A.size()}, {lora_B.size()}')
                            # Print sum of weights
                            print(f'LoRA weights sum: {lora_A.sum()}, {lora_B.sum()}')
                            found += 1
                            #pbar.set_description(f'found lora weights for {key}: {lora_A.size()}, {lora_B.size()}')
                            old_type = tensor.dtype
                            tensor = tensor.to(torch.float32)
                            tensor += scale * lora_B.to(torch.float32) @ lora_A.to(torch.float32)
                            tensor = tensor.to(old_type)
                            
                            delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32))
                            strengths_l2.append(torch.norm(delta).item())
                            strengths_l1.append(torch.mean(delta.abs()).item())
                tensors[key] = tensor
            safetensors.torch.save_file(tensors, output_path / shard.name, metadata=metadata)
    else:
        # Copy the shard as is
        print(f'Copying {shard.name} to output')
        shutil.copy(shard, output_path / shard.name)
print(f"Applied LoRA to {found} tensors.")

if strengths_l2:
    mean_l2 = float(np.mean(strengths_l2))
    median_l2 = float(np.median(strengths_l2))
    mean_l1 = float(np.mean(strengths_l1))
    median_l1 = float(np.median(strengths_l1))
    print(f"LoRA strength (L2 norm) - mean: {mean_l2:.6f}, median: {median_l2:.6f}")
    print(f"LoRA strength (mean abs) - mean: {mean_l1:.6f}, median: {median_l1:.6f}")
else:
    print('No LoRA strength metrics computed; check target_modules vs state keys.')
