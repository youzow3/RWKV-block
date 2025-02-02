# CLI tooling for converting RWKV models into HF compatible format with the following params
#
# - Model Source, either as
#     - RWKV model path file
#     - RWKV model safetensor file
#     - HF model directory
# - Directory to output the converted safetensor huggingface model
# - With the "model_class" defaults to "v7_goose"
#
# As part of the build process, you would need to do the following relative to the hf_builder.py script (this file).
#
# - Copy all files recursively from "../rwkv_block/{model_class}" into "./hf_code/{model_class}/rwkv_block/{model_class}/"
# - Import 'RWKV7PreTrainedModel' from '.hf_code.{model_class}.modeling_rwkv7'
# - Import 'RWKV7Config' from '.hf_code.{model_class}.configuration_rwkv7'
# - These are HF compatible classes that are used to load the model and save them
#
# - Load the model weights dictionary to CPU
# - Using the 'RWKV7Config.from_model_state_dict' class, get the config from the model weights
# - Using the config, initialize the model
# - Load the model weights into the configured model
# - Save the model to the output directory

import argparse, shutil, json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

####
# System path configuration
####
# Get the current script directory
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the system path, if not already present
if current_dir not in sys.path:
    sys.path.append(current_dir)

####
# Conversion and sync code
####
def sync_hf_code_rwkv_block(model_class):
    source_dir = Path(f"{current_dir}/../rwkv_block/{model_class}")
    target_dir = Path(f"{current_dir}/hf_code/{model_class}/rwkv_block/{model_class}")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    
def load_model_from_filepath(model_path):
    if model_path.endswith('.safetensors'):
        return load_file(model_path, device='cpu')
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        return torch.load(model_path, map_location='cpu', weights_only=True, mmap=True)

def save_tokenizer_to_output_dir(output_dir, tokenizer_type):
    # Check if the source directory exists
    source_dir = Path(f"{current_dir}/hf_code/tokenizer/{tokenizer_type}/")
    if not source_dir.exists():
        raise ValueError(f"Tokenizer type '{tokenizer_type}' not found in hf_code/tokenizer directory.")
    
    # Copy the tokenizer files to the output directory
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

def save_model_code_to_output_dir(output_dir, model_class):
    # Check if the source directory exists
    source_dir = Path(f"{current_dir}/hf_code/{model_class}")
    if not source_dir.exists():
        raise ValueError(f"Model class '{model_class}' not found in hf_code directory.")
    
    # Copy the model files to the output directory
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

####
# Builder scripts
####
def hf_builder(args):
    # Print the args
    print("-----------------------------")
    print("Converting RWKV model to HuggingFace format...")
    print(f"Model Class     : {args.model_class}")
    print(f"Model Source    : {args.model_source}")
    print(f"Tokenizer Type  : {args.tokenizer_type}")
    print(f"Output Directory: {args.output_dir}")
    print("-----------------------------")

    # Copy rwkv_block code files
    print("Syncing rwkv_block with HF code ...")
    sync_hf_code_rwkv_block(args.model_class)

    # Load model weights
    print("Loading model weights raw state ...")
    state_dict = load_model_from_filepath(args.model_source)

    # Load for the respective class
    print("Loading model class instance ...")
    model_class = args.model_class
    if model_class == "v7_goose":
        from hf_code.v7_goose.modeling_rwkv7 import RWKV7PreTrainedModel
        from hf_code.v7_goose.configuration_rwkv7 import RWKV7Config

        model_config = RWKV7Config.from_model_state_dict(state_dict)
        model_config = RWKV7Config(**model_config.__dict__)
        model_instance = RWKV7PreTrainedModel(model_config)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    # Deduce the tokenizer
    tokenizer_type = args.tokenizer_type
    if tokenizer_type == "auto":
        # Check for world tokenizer
        if model_config.vocab_size >= 65529 and model_config.vocab_size <= 65536:
            tokenizer_type = "world"
        elif model_config.vocab_size >= 50304 and model_config.vocab_size <= 50432:
            tokenizer_type = "neox"
        else:
            raise ValueError(f"Unable to detect tokenizer type for: {tokenizer_type} (vocab_size: {model_config.vocab_size})")
        
        # Print the detected tokenizer type
        print(f"Detected Tokenizer Type: {tokenizer_type}")

    # Load the model files
    print("Loading model state into class ...")
    model_instance.load_state_dict(state_dict)

    # print("-----------------------------")
    # print("Model Configuration:")
    # print(model_instance.config)
    print("-----------------------------")

    print("Saving tokenizer files ...")
    os.makedirs(args.output_dir, exist_ok=True)
    save_tokenizer_to_output_dir(args.output_dir, tokenizer_type)

    print("Saving model code files ...")
    save_model_code_to_output_dir(args.output_dir, model_class)
    
    print("Saving model weight files ...")
    model_instance.save_pretrained(args.output_dir)
    model_config.save_pretrained(args.output_dir)

    print("Patching configuration ...")
    config_json_path = Path(f"{args.output_dir}/config.json")
    config_json = json.load(config_json_path.open())

    # Fill in the auto_map, and architecture
    if model_class == "v7_goose":
        config_json["auto_map"] = {
            "AutoConfig": "modeling_rwkv7.RWKV7Config",
            "AutoModel": "modeling_rwkv7.RWKV7Model",
            "AutoModelForCasualLM": "modeling_rwkv7.RWKV7ForCasualLM"
        }
        config_json["architectures"] = ["RWKV7ForCasualLM", "RWKV7Model", "RWKV7PreTrainedModel"]
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    # Save the patched config
    with config_json_path.open("w") as f:
        json.dump(config_json, f, indent=2)
    
    # Print the success message
    print("-----------------------------")
    print("Successfully converted RWKV model to HuggingFace format")
    print("-----------------------------")

def main():
    parser = argparse.ArgumentParser(description="Convert RWKV models to HuggingFace format")
    parser.add_argument("model_source", help="Path to RWKV model file in .pth or .safetensors format")
    parser.add_argument("output_dir", help="Directory to output the converted HuggingFace model")
    parser.add_argument("--model_class", default="v7_goose", help="Model class (default: v7_goose)")
    parser.add_argument("--tokenizer_type", default="auto", help="Tokenizer to use, either 'auto','world' or 'neox' (default: auto)")
    args = parser.parse_args()
    hf_builder(args)

if __name__ == "__main__":
    main()