"""
File: scale_lora.py
Author: [Author Name - placeholder, as script is new]
Date: [Date of script creation/last modification - placeholder]

Purpose:
This script modifies the strength or intensity of a LoRA (Low-Rank Adaptation)
adapter by scaling its constituent weight matrices. Specifically, it targets the
`lora_B` weight matrices (though `lora_A` could also be targeted with minor
modification) and multiplies them by a user-provided scaling factor. The result
is saved as a new LoRA adapter in a specified output directory, leaving the
original LoRA unchanged. This allows for quick adjustments to a LoRA's impact
without needing to retrain it.

Methods:
- scale_lora(lora_path_str: str, output_path_str: str, scale_factor: float, device_str: str = 'cuda'):
    The main function performing the scaling operation.
    - Takes paths for the input LoRA directory, output directory, the scaling
      factor, and the computation device ('cuda' or 'cpu') as input.
    - Validates the existence of the input LoRA directory.
    - Creates the output directory if it doesn't already exist.
    - Sets up the PyTorch device, with a fallback to CPU if CUDA is requested but unavailable.
    - Loads the LoRA model's state dictionary from either `adapter_model.safetensors`
      (preferred) or `adapter_model.bin` located within the input LoRA directory.
    - Iterates through each tensor in the loaded state dictionary:
        - If the tensor's key includes "lora_B" (indicating it's a `lora_B` matrix),
          the tensor is scaled by the `scale_factor`. The scaling operation is
          performed in float32 precision to maintain accuracy, and the tensor is
          then cast back to its original data type.
        - Tensors not matching the "lora_B" criteria are copied to the new state
          dictionary without modification.
    - Reports the number of tensors scaled or warns if no relevant tensors were found.
    - Saves the modified (scaled) state dictionary to `adapter_model.safetensors`
      in the specified output directory.
    - Copies essential configuration and tokenizer-related files (e.g.,
      `adapter_config.json`, `README.md`, tokenizer files) from the input LoRA
      directory to the output directory to ensure the new LoRA is complete and usable.
    - Issues a warning if `adapter_config.json` is missing, as it is crucial for
      the LoRA's functionality.
- Main script execution block (if __name__ == '__main__'):
    - Sets up an `argparse.ArgumentParser` to handle command-line arguments:
      `lora_path`, `output_path`, `scale_factor`, and an optional `--device`.
    - Parses the provided command-line arguments.
    - Calls the `scale_lora` function with these parsed arguments.
    - Includes basic try-except error handling for the `scale_lora` call, printing
      the error and a traceback if an exception occurs.

Objects:
- parser (argparse.ArgumentParser): Manages command-line argument definitions and parsing.
- args (argparse.Namespace): Stores the parsed command-line arguments.
- lora_path, output_path (pathlib.Path): Path objects for input and output directories.
- device (torch.device): PyTorch device object used for tensor operations.
- lora_state_dict (dict): A dictionary where keys are tensor names (str) and values
  are the corresponding PyTorch tensors, loaded from the input LoRA model file.
- scaled_lora_state_dict (dict): A new dictionary created to hold the tensors of
  the scaled LoRA, which will then be saved.
- tensor, scaled_tensor (torch.Tensor): PyTorch tensor objects representing individual
  weights from the LoRA model.

Parameters (Command-Line Arguments):
- lora_path (str, positional): The file system path to the input LoRA directory.
- output_path (str, positional): The file system path to the directory where the
  newly scaled LoRA will be saved.
- scale_factor (float, positional): The numerical factor by which the `lora_B`
  weights will be multiplied.
- --device (str, optional, default='cuda', choices=['cuda', 'cpu']): Specifies
  whether to use 'cuda' or 'cpu' for tensor operations.

Configurations (Derived from parameters or environment):
- Computation device: Selected based on the `--device` argument and CUDA availability.
- File paths: Input and output paths for LoRA data and configuration files are
  constructed based on the `lora_path` and `output_path` arguments.

Hardcoded Variables:
- Standard LoRA model filenames: "adapter_model.safetensors", "adapter_model.bin".
- Target substring for weight scaling: "lora_B" (identifies the LoRA B matrices).
  (A commented-out section indicates "lora_A" was also considered).
- `config_files_to_copy` (list of str): A predefined list of common configuration and
  tokenizer file names that are typically associated with a LoRA adapter and should
  be copied to the new scaled LoRA directory (e.g., "adapter_config.json",
  "README.md", "tokenizer.json"). "peft_config.json" is also conditionally copied.
"""
import argparse
import os
import shutil
from pathlib import Path
import json

import safetensors.torch
import torch
from tqdm import tqdm

# MIT License parts from the original script are good practice but omitted here for brevity
# as this is a new script. If distributing, add appropriate licensing.

def scale_lora(lora_path_str: str, output_path_str: str, scale_factor: float, device_str: str = 'cuda'):
    """
    Loads a LoRA, scales its lora_B weights, and saves the modified LoRA.

    Args:
        lora_path_str (str): Path to the input LoRA directory.
        output_path_str (str): Path to the output directory for the scaled LoRA.
        scale_factor (float): The factor by which to scale the lora_B weights.
        device_str (str): Device to use for tensor operations ('cuda' or 'cpu').
    """
    lora_path = Path(lora_path_str)
    output_path = Path(output_path_str)

    if not lora_path.exists() or not lora_path.is_dir():
        raise FileNotFoundError(f"Input LoRA path not found or not a directory: {lora_path}")

    os.makedirs(output_path, exist_ok=True)

    # Determine device
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # --- Load LoRA model ---
    lora_model_path_safetensors = lora_path / 'adapter_model.safetensors'
    lora_model_path_bin = lora_path / 'adapter_model.bin'
    lora_state_dict = None

    print('Loading LoRA model...')
    if lora_model_path_safetensors.exists():
        lora_state_dict = safetensors.torch.load_file(lora_model_path_safetensors, device=device_str) # Load directly to device
        print(f"Loaded LoRA from {lora_model_path_safetensors}")
    elif lora_model_path_bin.exists():
        lora_state_dict = torch.load(lora_model_path_bin, map_location=device)
        print(f"Loaded LoRA from {lora_model_path_bin}")
    else:
        raise FileNotFoundError(
            f"Neither 'adapter_model.safetensors' nor 'adapter_model.bin' found in {lora_path}"
        )

    # --- Scale LoRA weights ---
    scaled_lora_state_dict = {}
    scaled_tensors_count = 0
    print(f"Scaling LoRA weights by a factor of {scale_factor}...")

    for key, tensor in tqdm(lora_state_dict.items(), desc="Processing tensors"):
        # Typically, LoRA weights are identified by 'lora_A' or 'lora_B' in their names.
        # We'll scale lora_B. This is a common way to adjust LoRA strength.
        # The actual names can vary slightly based on the PEFT library version or model structure,
        # e.g., "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        if "lora_B" in key: # Targeting lora_B weights
            original_dtype = tensor.dtype
            # Perform scaling in float32 for precision, then cast back
            scaled_tensor = (tensor.to(torch.float32) * scale_factor).to(original_dtype)
            scaled_lora_state_dict[key] = scaled_tensor
            scaled_tensors_count += 1
        # elif "lora_A" in key: # If you wanted to scale lora_A instead or additionally
        #     original_dtype = tensor.dtype
        #     scaled_tensor = (tensor.to(torch.float32) * scale_factor).to(original_dtype) # or math.sqrt(scale_factor)
        #     scaled_lora_state_dict[key] = scaled_tensor
        #     scaled_tensors_count += 1
        else:
            scaled_lora_state_dict[key] = tensor # Copy other tensors as is

    if scaled_tensors_count == 0:
        print("Warning: No LoRA tensors (containing 'lora_B') found to scale. Output will be a copy.")
    else:
        print(f"Scaled {scaled_tensors_count} 'lora_B' tensors.")


    # --- Save the scaled LoRA ---
    output_model_file = output_path / 'adapter_model.safetensors' # Prefer saving as safetensors
    print(f"Saving scaled LoRA model to {output_model_file}...")
    safetensors.torch.save_file(scaled_lora_state_dict, output_model_file)

    # --- Copy configuration files ---
    config_files_to_copy = ['adapter_config.json', 'README.md', 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer.model'] # Common files
    
    # Also copy peft_config.json if present (older PEFT versions)
    if (lora_path / 'peft_config.json').exists():
        config_files_to_copy.append('peft_config.json')

    print("Copying configuration files...")
    for config_file_name in config_files_to_copy:
        source_file = lora_path / config_file_name
        dest_file = output_path / config_file_name
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"  Copied {config_file_name}")
        # else: # Be less verbose about missing non-critical files
        #     if config_file_name == 'adapter_config.json':
        #         print(f"  Warning: Crucial '{config_file_name}' not found in source.")


    # Check for adapter_config.json specifically
    if not (output_path / 'adapter_config.json').exists():
         print(f"Warning: 'adapter_config.json' was not found in '{lora_path}' and could not be copied. "
               "The scaled LoRA might not be usable without it.")


    print(f"Scaled LoRA saved to {output_path}")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scales LoRA A and B tensors and saves the modified LoRA.")
    parser.add_argument('lora_path', type=str, help='Path to the input LoRA directory.')
    parser.add_argument('output_path', type=str, help='Path to the output directory for the scaled LoRA.')
    parser.add_argument('scale_factor', type=float, help='The factor by which to scale the LoRA B weights (e.g., 0.5 to halve, 2.0 to double).')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for tensor operations (cuda or cpu). Default: cuda.')

    args = parser.parse_args()

    try:
        scale_lora(args.lora_path, args.output_path, args.scale_factor, args.device)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()