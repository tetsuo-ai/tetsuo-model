"""
Tetsuo AI Training Configuration

Generates ComfyUI API workflow JSONs for Flux LoRA training.
Can also submit workflows to a running ComfyUI server.

Usage:
    # Generate workflow JSON:
    python scripts/tetsuo_train_config.py generate

    # Submit to running ComfyUI server:
    python scripts/tetsuo_train_config.py submit

    # Submit with custom settings:
    python scripts/tetsuo_train_config.py submit --steps 2000 --rank 64 --lr 5e-5
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

# Default training hyperparameters optimized for Flux character training
DEFAULTS = {
    # Model paths (user must set these to match their downloaded files)
    "unet_name": "flux1-dev.safetensors",
    "clip_name1": "clip_l.safetensors",
    "clip_name2": "t5xxl_fp16.safetensors",
    "vae_name": "ae.safetensors",
    "clip_type": "flux",

    # Dataset
    "dataset_folder": "tetsuo_dataset",

    # Training params
    "batch_size": 1,
    "grad_accumulation_steps": 4,
    "steps": 1500,
    "learning_rate": 0.0001,
    "rank": 32,
    "optimizer": "AdamW",
    "loss_function": "MSE",
    "seed": 42,
    "training_dtype": "bf16",
    "lora_dtype": "bf16",
    "algorithm": "lora",
    "gradient_checkpointing": True,
    "checkpoint_depth": 1,
    "offloading": False,
    "existing_lora": "[None]",
    "bucket_mode": True,
    "bypass_mode": False,

    # Output
    "lora_prefix": "loras/tetsuo_v1",
    "loss_graph_prefix": "tetsuo_loss",

    # Server
    "server_address": "127.0.0.1:8188",
}


def build_training_workflow(config: dict) -> dict:
    """Build a ComfyUI API-format workflow for Flux LoRA training."""
    workflow = {
        # Node 1: Load Flux UNET
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": config["unet_name"],
                "weight_dtype": "default",
            },
        },
        # Node 2: Load Dual CLIP (CLIP-L + T5-XXL for Flux)
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": config["clip_name1"],
                "clip_name2": config["clip_name2"],
                "type": config["clip_type"],
            },
        },
        # Node 3: Load VAE
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": config["vae_name"],
            },
        },
        # Node 4: Load Image+Text Dataset
        "4": {
            "class_type": "LoadImageTextDataSetFromFolder",
            "inputs": {
                "folder": config["dataset_folder"],
            },
        },
        # Node 5: VAE Encode (images -> latents)
        # Receives list of images from node 4, runs per-image
        "5": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["4", 0],  # images from dataset loader
                "vae": ["3", 0],     # VAE model
            },
        },
        # Node 6: CLIP Text Encode (captions -> conditioning)
        # Receives list of texts from node 4, runs per-text
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": ["4", 1],    # texts from dataset loader
                "clip": ["2", 0],    # CLIP model
            },
        },
        # Node 7: Train LoRA (core training node)
        "7": {
            "class_type": "TrainLoraNode",
            "inputs": {
                "model": ["1", 0],               # Flux UNET
                "latents": ["5", 0],             # encoded latents
                "positive": ["6", 0],            # encoded conditioning
                "batch_size": config["batch_size"],
                "grad_accumulation_steps": config["grad_accumulation_steps"],
                "steps": config["steps"],
                "learning_rate": config["learning_rate"],
                "rank": config["rank"],
                "optimizer": config["optimizer"],
                "loss_function": config["loss_function"],
                "seed": config["seed"],
                "training_dtype": config["training_dtype"],
                "lora_dtype": config["lora_dtype"],
                "algorithm": config["algorithm"],
                "gradient_checkpointing": config["gradient_checkpointing"],
                "checkpoint_depth": config["checkpoint_depth"],
                "offloading": config["offloading"],
                "existing_lora": config["existing_lora"],
                "bucket_mode": config["bucket_mode"],
                "bypass_mode": config["bypass_mode"],
            },
        },
        # Node 8: Save LoRA weights
        "8": {
            "class_type": "SaveLoRA",
            "inputs": {
                "lora": ["7", 0],    # LORA_MODEL from training
                "prefix": config["lora_prefix"],
                "steps": ["7", 2],   # steps count from training
            },
        },
        # Node 9: Plot Loss Graph
        "9": {
            "class_type": "LossGraphNode",
            "inputs": {
                "loss": ["7", 1],    # LOSS_MAP from training
                "filename_prefix": config["loss_graph_prefix"],
            },
        },
    }

    # If bucket_mode is enabled, insert ResolutionBucket node between
    # VAEEncode/CLIPTextEncode and TrainLoraNode
    if config["bucket_mode"]:
        workflow["10"] = {
            "class_type": "ResolutionBucket",
            "inputs": {
                "latents": ["5", 0],    # latents from VAEEncode
                "conditioning": ["6", 0],  # conditioning from CLIPTextEncode
            },
        }
        # Rewire training node to use bucketed outputs
        workflow["7"]["inputs"]["latents"] = ["10", 0]
        workflow["7"]["inputs"]["positive"] = ["10", 1]

    return workflow


def build_inference_workflow(config: dict, prompt: str, negative: str = "",
                             width: int = 1024, height: int = 1024,
                             steps: int = 20, cfg: float = 3.5,
                             lora_name: str = "tetsuo_v1.safetensors",
                             lora_strength: float = 1.0) -> dict:
    """Build a ComfyUI API-format workflow for Flux inference with Tetsuo LoRA."""
    workflow = {
        # Load Flux UNET
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": config["unet_name"],
                "weight_dtype": "default",
            },
        },
        # Load CLIP
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": config["clip_name1"],
                "clip_name2": config["clip_name2"],
                "type": config["clip_type"],
            },
        },
        # Load VAE
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": config["vae_name"],
            },
        },
        # Apply Tetsuo LoRA
        "4": {
            "class_type": "LoRALoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["2", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        },
        # Positive prompt
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1],  # CLIP from LoRALoader
            },
        },
        # Empty latent
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        # KSampler
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],  # model from LoRALoader
                "positive": ["5", 0],
                "negative": ["5", 0],  # Flux doesn't really use negative
                "latent_image": ["6", 0],
                "seed": config["seed"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        # VAE Decode
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0],
            },
        },
        # Save Image
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "tetsuo_output",
            },
        },
    }
    return workflow


def submit_workflow(workflow: dict, server_address: str):
    """Submit a workflow to a running ComfyUI server."""
    import urllib.request

    prompt_id = str(uuid.uuid4())
    payload = {
        "prompt": workflow,
        "client_id": prompt_id,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{server_address}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            print(f"Workflow submitted successfully!")
            print(f"  Prompt ID: {result.get('prompt_id', 'unknown')}")
            print(f"  Server: {server_address}")
            return result
    except urllib.error.URLError as e:
        print(f"Error: Could not connect to ComfyUI server at {server_address}")
        print(f"  Make sure ComfyUI is running (python main.py)")
        print(f"  Error: {e}")
        sys.exit(1)


def save_workflow(workflow: dict, output_path: str):
    """Save workflow to JSON file."""
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)
    print(f"Workflow saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tetsuo AI Training Config")
    subparsers = parser.add_subparsers(dest="command")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate workflow JSON files")
    gen_parser.add_argument("--output", default="workflows/tetsuo_flux_train.json")

    # Submit command
    sub_parser = subparsers.add_parser("submit", help="Submit training workflow to ComfyUI")
    sub_parser.add_argument("--server", default=DEFAULTS["server_address"])

    # Inference command
    inf_parser = subparsers.add_parser("inference", help="Generate inference workflow")
    inf_parser.add_argument("--prompt", required=True, help="Generation prompt")
    inf_parser.add_argument("--output", default="workflows/tetsuo_flux_inference.json")
    inf_parser.add_argument("--lora_name", default="tetsuo_v1.safetensors")
    inf_parser.add_argument("--width", type=int, default=1024)
    inf_parser.add_argument("--height", type=int, default=1024)

    # Shared training params for generate and submit
    for p in [gen_parser, sub_parser]:
        p.add_argument("--unet_name", default=DEFAULTS["unet_name"])
        p.add_argument("--clip_name1", default=DEFAULTS["clip_name1"])
        p.add_argument("--clip_name2", default=DEFAULTS["clip_name2"])
        p.add_argument("--vae_name", default=DEFAULTS["vae_name"])
        p.add_argument("--dataset_folder", default=DEFAULTS["dataset_folder"])
        p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
        p.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
        p.add_argument("--rank", type=int, default=DEFAULTS["rank"])
        p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
        p.add_argument("--grad_acc", type=int, default=DEFAULTS["grad_accumulation_steps"])
        p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
        p.add_argument("--no_bucket", action="store_true", help="Disable bucket mode")
        p.add_argument("--bypass", action="store_true", help="Enable bypass mode")
        p.add_argument("--offload", action="store_true", help="Enable RAM offloading")

    args = parser.parse_args()

    if args.command in ("generate", "submit"):
        config = dict(DEFAULTS)
        config.update({
            "unet_name": args.unet_name,
            "clip_name1": args.clip_name1,
            "clip_name2": args.clip_name2,
            "vae_name": args.vae_name,
            "dataset_folder": args.dataset_folder,
            "steps": args.steps,
            "learning_rate": args.lr,
            "rank": args.rank,
            "batch_size": args.batch_size,
            "grad_accumulation_steps": args.grad_acc,
            "seed": args.seed,
            "bucket_mode": not args.no_bucket,
            "bypass_mode": args.bypass,
            "offloading": args.offload,
        })

        workflow = build_training_workflow(config)

        if args.command == "generate":
            save_workflow(workflow, args.output)
            print("\nTraining config:")
            print(f"  Steps: {config['steps']}")
            print(f"  Learning rate: {config['learning_rate']}")
            print(f"  Rank: {config['rank']}")
            print(f"  Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['grad_accumulation_steps']})")
            print(f"  Bucket mode: {config['bucket_mode']}")
            print(f"  Algorithm: {config['algorithm']}")
            print(f"\nTo use: Load this JSON in ComfyUI or run 'python scripts/tetsuo_train_config.py submit'")
        else:
            print("Submitting training workflow to ComfyUI...")
            print(f"  Steps: {config['steps']}, LR: {config['learning_rate']}, Rank: {config['rank']}")
            submit_workflow(workflow, args.server)

    elif args.command == "inference":
        config = dict(DEFAULTS)
        workflow = build_inference_workflow(
            config, args.prompt,
            width=args.width, height=args.height,
            lora_name=args.lora_name,
        )
        save_workflow(workflow, args.output)

    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. Generate training workflow:  python scripts/tetsuo_train_config.py generate")
        print("  2. Submit to ComfyUI server:    python scripts/tetsuo_train_config.py submit")
        print("  3. Generate inference workflow:  python scripts/tetsuo_train_config.py inference --prompt 'tetsuo_character, ...'")


if __name__ == "__main__":
    main()
