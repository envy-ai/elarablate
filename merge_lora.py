"""
File: merge_lora.py
Author: tdrussell (Original MIT License)
Date: [Date of script creation/last modification - placeholder, as it's not in the original script]

Purpose:
This script is designed to merge one or more LoRA (Low-Rank Adaptation) adapters
into the weights of a base Hugging Face transformer model. It takes an input model,
one or more LoRA adapters (each with a specified scaling factor), and an output path.
The script then creates a new model in the output path with the LoRA modifications
directly applied (baked into) the original model weights. This is useful for
deploying fine-tuned models without needing the PEFT library at inference time or
managing separate adapter files. The script supports applying LoRAs to a specific
range of layers within the model.

Methods:
- find_lora_weights(key, lora_path): Given a base model tensor key and a path to a
  loaded LoRA, this function searches for the corresponding 'lora_A' and 'lora_B'
  weight matrices within that LoRA's state dictionary.
- LoraAction(argparse.Action): A custom action for argparse that allows users to
  specify LoRA adapters and their scales as pairs of arguments on the command line
  (e.g., --lora /path/to/lora1 0.5 /path/to/lora2 0.8). It validates that an even
  number of arguments are provided for paths and scales.
- Main script execution (unnamed, top-level):
    - Parses command-line arguments including input model path, LoRA paths and scales,
      output path, layer range for merging, and device preference.
    - Validates the specified layer range.
    - Creates the output directory if it doesn't exist.
    - Loads each specified LoRA:
        - Reads `adapter_config.json` to get `lora_alpha` and `r`.
        - Loads LoRA weights from `adapter_model.safetensors`.
        - Calculates an effective scale for the LoRA based on its alpha, r,
          the user-provided scale, and a global `scale_all` factor.
        - Moves LoRA weights to GPU if `--no-gpu` is not set.
    - Identifies model shards (e.g., `model-00001-of-0000X.safetensors`) in the input path.
    - If `model.safetensors.index.json` exists, it's used to determine which shards
      contain weights for layers within the specified `layer_range`. These are
      designated as `lora_shards`.
    - Copies non-model files and model shards that fall outside the `layer_range`
      directly to the output directory.
    - Iterates through each model shard of the base model:
        - If the shard is within the `layer_range` (is a `lora_shard`):
            - Opens the shard and iterates through its tensor keys.
            - For each tensor, checks if its layer number is within the target range.
            - If yes, it iterates through all provided LoRAs:
                - Retrieves the `lora_A` and `lora_B` matrices for the current tensor key.
                - Calculates the weight delta: `effective_scale * (lora_B @ lora_A)`.
                - Adds this delta to the base model's tensor (weights are cast to
                  float32 for the operation and then back to original dtype).
                - Optionally tracks L2 norm and mean absolute value of the delta for analysis.
            - Saves the modified tensors to a new shard in the output directory.
        - If the shard is outside the `layer_range`, it's copied as-is.
    - Prints statistics, including the number of tensors LoRA was applied to and
      LoRA strength metrics (if any weights were merged).

Objects:
- parser (argparse.ArgumentParser): Manages command-line argument definitions and parsing.
- args (argparse.Namespace): Stores the parsed command-line arguments.
- input_path, output_path (pathlib.Path): Objects representing file system paths.
- device (str): Indicates the computation device ('cpu' or 'cuda').
- lora_state (dict): A nested dictionary: {lora_adapter_path: {tensor_name: tensor_weight}}.
  Stores the weights of all loaded LoRA adapters.
- scale (dict): Maps LoRA adapter paths to their calculated effective scaling factors.
- shards (list): A list of `Path` objects for each model shard file in the input directory.
- lora_shards (list): A list of shard filenames (str) that are targeted for LoRA merging.
- indexdata (dict): Parsed content of `model.safetensors.index.json`, primarily the `weight_map`.
- tensors (dict): Temporarily holds tensors of a single shard before saving.
- lora_config (dict): Parsed content of a LoRA's `adapter_config.json`.
- strengths_l2, strengths_l1 (list): Store L2 norm and mean absolute L1 norm of the
  applied LoRA deltas for diagnostic purposes.

Parameters (Command-Line Arguments):
- input_path (str, positional): Path to the directory containing the base model (sharded).
- output_path (str, positional): Path to the directory where the merged model will be saved.
- --no-gpu (bool, action='store_true'): If specified, forces the merge operation to run on CPU.
- --layer_range (str, default='49-79'): Specifies the range of model layers to apply LoRA to
  (e.g., "min-max"). Default is suitable for some 70B Llama models.
- --lora (str, nargs='*', action=LoraAction, required=True): Defines LoRA adapters and their
  scales. Expects pairs: `[path_to_lora1] [scale1] [path_to_lora2] [scale2] ...`.
- --scale_all (float, default=1.0): A global scaling factor applied to all LoRAs,
  multiplied with their individual scales.

Configurations (Derived from parameters or files):
- `start_layer`, `end_layer` (int): Parsed from `args.layer_range`.
- Effective LoRA scale (float): Calculated per LoRA as
  `lora_config['lora_alpha'] / lora_config['r'] * individual_scale_arg * args.scale_all`.
- `lora_shards` (list of str): Determined by comparing layer names in `model.safetensors.index.json`
  against the `start_layer` and `end_layer`.

Hardcoded Variables:
- Default LoRA filenames: `adapter_config.json`, `adapter_model.safetensors`.
- Model index filename: `model.safetensors.index.json`.
- Regex `r'^language_model\.'`: Used to normalize tensor keys from the base model
  when looking up corresponding LoRA weights (removes "language_model." prefix).
- Regex `r'layers\.(\d+)'`: Used to extract the layer number from a tensor key to
  check if it's within the specified `layer_range`.
"""
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

class LoraAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) % 2 != 0:
            parser.error(f"{option_string} requires pairs of arguments: [path] [scale]")
        
        lora_configs = []
        for i in range(0, len(values), 2):
            path = values[i]
            try:
                scale = float(values[i + 1])
                lora_configs.append((path, scale))
            except ValueError:
                parser.error(f"Scale value '{values[i + 1]}' is not a valid float")
        
        setattr(namespace, self.dest, lora_configs)

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str, help='The path to the input directory.')
#parser.add_argument('lora_path', type=str, help='The path to the LoRA directory.')
parser.add_argument('output_path', type=str, help='The path to the output directory.')
parser.add_argument('--no-gpu', action='store_true', help='Use CPU for merging. Probably unnecessary unless each individual shard is too large to fit in GPU memory.')
parser.add_argument('--layer_range', type=str, default='49-79', help='Range of layers to apply LoRA to. Default is 59-79, for 70B llama models.')
parser.add_argument('--lora', nargs='*', action=LoraAction, default=[], required=True,
                    help='LoRA configurations as pairs: [path] [scale] [path] [scale] ...')
parser.add_argument('--scale_all', type=float, default=1.0, help='Scale all lora weights by this factor. Default is 1.0.')
args = parser.parse_args()

start_layer, end_layer = map(int, args.layer_range.split('-'))
if start_layer > end_layer:
    raise ValueError(f'Invalid layer range: {args.layer_range}. Start layer must be less than or equal to end layer.')
if start_layer < 0 or end_layer < 0:
    raise ValueError(f'Invalid layer range: {args.layer_range}. Start and end layers must be non-negative.')

input_path, output_path = Path(args.input_path), Path(args.output_path)
os.makedirs(output_path, exist_ok=True)

device = 'cpu' if args.no_gpu else 'cuda'

def find_lora_weights(key, lora_path):
    a = b = None
    prefix = key.replace('.weight', '')
    for lk, lw in lora_state[lora_path].items():
        if prefix in lk:
            if 'lora_A' in lk:
                a = lw
            elif 'lora_B' in lk:
                b = lw
    assert (a is None) == (b is None), f"Missing LoRA A or B for {prefix}"
    return a, b



print('Loading LoRA models...')

lora_state = {}
scale = {}
for lora_path, lora_scale in args.lora:
    lora_config = peft.LoraConfig.from_json_file(lora_path + '/adapter_config.json')
    scale[lora_path] = lora_config['lora_alpha'] / lora_config['r'] * lora_scale * args.scale_all
    
    lora_path_path = Path(lora_path)
    if not lora_path_path.exists():
        raise FileNotFoundError(f'LoRA path {lora_path} does not exist.')
    if not lora_path_path.is_dir():
        raise NotADirectoryError(f'LoRA path {lora_path} is not a directory.')
    print(f'Loading LoRA from {lora_path}')
    print(f'Using LoRA scale: {scale[lora_path]}')
    # Check if we have adapter_model.bin or adapter_model.safetensors
    

    lora_state[lora_path] = safetensors.torch.load_file(lora_path + '/adapter_model.safetensors')
    if not args.no_gpu:
        # Move mapped entries to cuda
        for key, value in tqdm(lora_state[lora_path].items()):
            lora_state[lora_path][key] = value.to('cuda')


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
                        for lora_path, junk in args.lora:
                            lora_A, lora_B = find_lora_weights(lora_key, lora_path)
                            lora_scale = scale[lora_path]
                            if lora_A is not None:
                                print(f'Found LoRA weights for {key}: {lora_A.size()}, {lora_B.size()}')
                                # Print sum of weights
                                print(f'LoRA weights sum: {lora_A.sum()}, {lora_B.sum()}')
                                found += 1
                                #pbar.set_description(f'found lora weights for {key}: {lora_A.size()}, {lora_B.size()}')
                                old_type = tensor.dtype
                                tensor = tensor.to(torch.float32)
                                tensor += lora_scale * lora_B.to(torch.float32) @ lora_A.to(torch.float32)
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
