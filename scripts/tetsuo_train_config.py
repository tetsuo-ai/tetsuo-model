"""
Tetsuo AI Training Configuration

Generates ComfyUI API workflow JSONs for Flux (image) and WAN 2.2 (video) LoRA training.
Can also submit workflows to a running ComfyUI server.

Usage:
    # Generate Flux image training workflow:
    python scripts/tetsuo_train_config.py generate --model flux

    # Generate WAN 2.2 video training workflow:
    python scripts/tetsuo_train_config.py generate --model wan

    # Submit to running ComfyUI server:
    python scripts/tetsuo_train_config.py submit --model flux
    python scripts/tetsuo_train_config.py submit --model wan

    # Custom settings:
    python scripts/tetsuo_train_config.py submit --model flux --steps 2000 --rank 64 --lr 5e-5
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

# Model presets
MODEL_PRESETS = {
    "flux": {
        "unet_name": "flux_dev_fp8_scaled_diffusion_model.safetensors",
        "clip_name1": "clip_l.safetensors",
        "clip_name2": "t5xxl_fp16.safetensors",
        "vae_name": "flux-vae-bf16.safetensors",
        "clip_type": "flux",
        "clip_loader": "DualCLIPLoader",
        "lora_prefix": "loras/tetsuo_flux_v1",
        "loss_graph_prefix": "tetsuo_flux_loss",
        "steps": 1500,
        "learning_rate": 0.0001,
    },
    "wan": {
        "unet_name": "wan2.2_t2v_high_noise_14B_Q8_0.gguf",
        "clip_name1": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "clip_name2": None,  # WAN uses single text encoder
        "vae_name": "wan2.2_vae.safetensors",
        "clip_type": None,  # Uses CLIPLoader, not DualCLIPLoader
        "clip_loader": "CLIPLoader",
        "lora_prefix": "loras/tetsuo_wan_v1",
        "loss_graph_prefix": "tetsuo_wan_loss",
        "steps": 1000,
        "learning_rate": 0.0001,
    },
}

# Shared training defaults
DEFAULTS = {
    "dataset_folder": "tetsuo_dataset",
    "batch_size": 1,
    "grad_accumulation_steps": 4,
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
    "server_address": "127.0.0.1:8188",
}


def build_training_workflow(config: dict) -> dict:
    """Build a ComfyUI API-format workflow for Flux or WAN 2.2 LoRA training."""
    # Node 1: Load UNET (works for both Flux and WAN 2.2)
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": config["unet_name"],
                "weight_dtype": "default",
            },
        },
    }

    # Node 2: Load text encoder(s)
    if config["clip_loader"] == "DualCLIPLoader":
        workflow["2"] = {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": config["clip_name1"],
                "clip_name2": config["clip_name2"],
                "type": config["clip_type"],
            },
        }
    else:
        # WAN 2.2 uses single CLIPLoader (UMT5-XXL)
        workflow["2"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": config["clip_name1"],
                "type": "wan",
            },
        }

    # Node 3: Load VAE
    workflow["3"] = {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": config["vae_name"],
        },
    }

    # Node 4: Load Image+Text Dataset
    workflow["4"] = {
        "class_type": "LoadImageTextDataSetFromFolder",
        "inputs": {
            "folder": config["dataset_folder"],
        },
    }

    # Node 5: VAE Encode (images -> latents)
    workflow["5"] = {
        "class_type": "VAEEncode",
        "inputs": {
            "pixels": ["4", 0],
            "vae": ["3", 0],
        },
    }

    # Node 6: CLIP Text Encode (captions -> conditioning)
    workflow["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": ["4", 1],
            "clip": ["2", 0],
        },
    }

    # Node 7: Train LoRA (core training node - works for both Flux and WAN 2.2)
    workflow["7"] = {
        "class_type": "TrainLoraNode",
        "inputs": {
            "model": ["1", 0],
            "latents": ["5", 0],
            "positive": ["6", 0],
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
    }

    # Node 8: Save LoRA weights
    workflow["8"] = {
        "class_type": "SaveLoRA",
        "inputs": {
            "lora": ["7", 0],
            "prefix": config["lora_prefix"],
            "steps": ["7", 2],
        },
    }

    # Node 9: Plot Loss Graph
    workflow["9"] = {
        "class_type": "LossGraphNode",
        "inputs": {
            "loss": ["7", 1],
            "filename_prefix": config["loss_graph_prefix"],
        },
    }

    # If bucket_mode is enabled, insert ResolutionBucket node
    if config["bucket_mode"]:
        workflow["10"] = {
            "class_type": "ResolutionBucket",
            "inputs": {
                "latents": ["5", 0],
                "conditioning": ["6", 0],
            },
        }
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


def build_config(model_type: str, args=None) -> dict:
    """Merge model preset + shared defaults + CLI overrides into a single config."""
    preset = MODEL_PRESETS[model_type]
    config = dict(DEFAULTS)
    config.update(preset)

    if args:
        overrides = {}
        if hasattr(args, "steps") and args.steps is not None:
            overrides["steps"] = args.steps
        if hasattr(args, "lr") and args.lr is not None:
            overrides["learning_rate"] = args.lr
        if hasattr(args, "rank") and args.rank is not None:
            overrides["rank"] = args.rank
        if hasattr(args, "batch_size") and args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        if hasattr(args, "grad_acc") and args.grad_acc is not None:
            overrides["grad_accumulation_steps"] = args.grad_acc
        if hasattr(args, "seed") and args.seed is not None:
            overrides["seed"] = args.seed
        if hasattr(args, "dataset_folder") and args.dataset_folder is not None:
            overrides["dataset_folder"] = args.dataset_folder
        if hasattr(args, "no_bucket") and args.no_bucket:
            overrides["bucket_mode"] = False
        if hasattr(args, "bypass") and args.bypass:
            overrides["bypass_mode"] = True
        if hasattr(args, "offload") and args.offload:
            overrides["offloading"] = True
        config.update(overrides)

    return config


def main():
    parser = argparse.ArgumentParser(description="Tetsuo AI Training Config")
    subparsers = parser.add_subparsers(dest="command")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate workflow JSON files")
    gen_parser.add_argument("--output", default=None, help="Output path (auto-named if omitted)")

    # Submit command
    sub_parser = subparsers.add_parser("submit", help="Submit training workflow to ComfyUI")
    sub_parser.add_argument("--server", default=DEFAULTS["server_address"])

    # Inference command
    inf_parser = subparsers.add_parser("inference", help="Generate inference workflow")
    inf_parser.add_argument("--prompt", required=True, help="Generation prompt")
    inf_parser.add_argument("--output", default="workflows/tetsuo_flux_inference.json")
    inf_parser.add_argument("--lora_name", default="tetsuo_flux_v1.safetensors")
    inf_parser.add_argument("--width", type=int, default=1024)
    inf_parser.add_argument("--height", type=int, default=1024)

    # Shared training params for generate and submit
    for p in [gen_parser, sub_parser]:
        p.add_argument("--model", choices=["flux", "wan"], default="flux",
                        help="Base model: flux (images) or wan (WAN 2.2 video)")
        p.add_argument("--dataset_folder", default=None)
        p.add_argument("--steps", type=int, default=None)
        p.add_argument("--lr", type=float, default=None)
        p.add_argument("--rank", type=int, default=None)
        p.add_argument("--batch_size", type=int, default=None)
        p.add_argument("--grad_acc", type=int, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--no_bucket", action="store_true", help="Disable bucket mode")
        p.add_argument("--bypass", action="store_true", help="Enable bypass mode")
        p.add_argument("--offload", action="store_true", help="Enable RAM offloading")

    args = parser.parse_args()

    if args.command in ("generate", "submit"):
        model_type = args.model
        config = build_config(model_type, args)
        workflow = build_training_workflow(config)

        model_label = "Flux (image)" if model_type == "flux" else "WAN 2.2 (video)"

        if args.command == "generate":
            output = args.output or f"workflows/tetsuo_{model_type}_train.json"
            save_workflow(workflow, output)
            print(f"\n{model_label} training config:")
            print(f"  Model: {config['unet_name']}")
            print(f"  Steps: {config['steps']}")
            print(f"  Learning rate: {config['learning_rate']}")
            print(f"  Rank: {config['rank']}")
            print(f"  Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['grad_accumulation_steps']})")
            print(f"  Bucket mode: {config['bucket_mode']}")
            print(f"  Algorithm: {config['algorithm']}")
            print(f"\nTo use: Load this JSON in ComfyUI or run:")
            print(f"  python scripts/tetsuo_train_config.py submit --model {model_type}")
        else:
            print(f"Submitting {model_label} training workflow to ComfyUI...")
            print(f"  Model: {config['unet_name']}")
            print(f"  Steps: {config['steps']}, LR: {config['learning_rate']}, Rank: {config['rank']}")
            submit_workflow(workflow, args.server)

    elif args.command == "inference":
        config = build_config("flux")
        workflow = build_inference_workflow(
            config, args.prompt,
            width=args.width, height=args.height,
            lora_name=args.lora_name,
        )
        save_workflow(workflow, args.output)

    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. Train Flux (images):   python scripts/tetsuo_train_config.py generate --model flux")
        print("  2. Train WAN 2.2 (video): python scripts/tetsuo_train_config.py generate --model wan")
        print("  3. Submit to ComfyUI:     python scripts/tetsuo_train_config.py submit --model flux")
        print("  4. Inference workflow:     python scripts/tetsuo_train_config.py inference --prompt 'tetsuo_character, ...'")


if __name__ == "__main__":
    main()
