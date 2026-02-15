"""
Tetsuo AI Hunyuan Video LoRA Training Script

Sets up and launches Musubi Tuner for HunyuanVideo LoRA fine-tuning.

Prerequisites:
    1. Clone Musubi Tuner: git clone https://github.com/kohya-tech/musubi-tuner.git
    2. Install deps: cd musubi-tuner && pip install -r requirements.txt
    3. Download HunyuanVideo model files (see setup command)
    4. Prepare video/image dataset

Usage:
    python scripts/tetsuo_video_train.py setup          # Clone and install Musubi Tuner
    python scripts/tetsuo_video_train.py prepare         # Prepare dataset for video training
    python scripts/tetsuo_video_train.py train           # Launch training
    python scripts/tetsuo_video_train.py train --steps 1500 --rank 32
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MUSUBI_DIR = PROJECT_ROOT / "musubi-tuner"
DATASET_DIR = PROJECT_ROOT / "dataset" / "video"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRIGGER_WORD = "tetsuo_character"

# Default training config
DEFAULTS = {
    "steps": 1000,
    "rank": 32,
    "learning_rate": 1e-4,
    "optimizer": "adamw",
    "batch_size": 1,
    "grad_acc": 1,
    "resolution": "512x512",
    "video_length": 25,
    "seed": 42,
    "mixed_precision": "bf16",
    "save_every_n_steps": 250,
    "output_name": "tetsuo_hunyuan_v1",
}


def run_cmd(cmd: list[str], cwd: str | None = None, check: bool = True):
    """Run a subprocess command."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        print(f"  Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def setup_musubi():
    """Clone and install Musubi Tuner."""
    print("Setting up Musubi Tuner for HunyuanVideo training...")

    if MUSUBI_DIR.exists():
        print(f"  Musubi Tuner already exists at {MUSUBI_DIR}")
        print("  Updating...")
        run_cmd(["git", "pull"], cwd=str(MUSUBI_DIR))
    else:
        print(f"  Cloning to {MUSUBI_DIR}...")
        run_cmd(["git", "clone", "https://github.com/kohya-tech/musubi-tuner.git", str(MUSUBI_DIR)])

    print("  Installing dependencies...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=str(MUSUBI_DIR))

    print("\nMusubi Tuner setup complete!")
    print("\nNext steps:")
    print("  1. Download HunyuanVideo model:")
    print("     huggingface-cli download tencent/HunyuanVideo --local-dir models/hunyuan_video")
    print("  2. Prepare your video dataset:")
    print("     python scripts/tetsuo_video_train.py prepare")
    print("  3. Start training:")
    print("     python scripts/tetsuo_video_train.py train")


def prepare_dataset():
    """Prepare video/image dataset for Hunyuan Video training."""
    print("Preparing video dataset...")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing images in the image dataset
    image_dataset = COMFYUI_ROOT / "input" / "tetsuo_dataset"
    image_dirs = list(image_dataset.glob("*_tetsuo_character"))

    if image_dirs:
        print(f"  Found image dataset at {image_dirs[0]}")
        print("  You can use these images for video training too.")
        print("  Musubi Tuner supports both images and videos in the dataset.")
    else:
        print("  No existing image dataset found.")

    # Create dataset structure
    train_dir = DATASET_DIR / f"10_{TRIGGER_WORD}"
    train_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset directory created: {train_dir}")
    print("\nAdd your training data:")
    print("  - Videos: .mp4, .avi, .mov files with matching .txt caption files")
    print("  - Images: .png, .jpg files with matching .txt caption files")
    print(f"  - Caption format: '{TRIGGER_WORD}, description of the scene...'")
    print("\nExample structure:")
    print(f"  {train_dir}/")
    print(f"    video_001.mp4")
    print(f"    video_001.txt  -> '{TRIGGER_WORD}, cyberpunk woman walking through neon city'")
    print(f"    image_001.png")
    print(f"    image_001.txt  -> '{TRIGGER_WORD}, platinum hair woman in tactical gear'")

    # Generate the TOML config
    generate_config(DEFAULTS)


def generate_config(config: dict):
    """Generate Musubi Tuner TOML config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / "tetsuo_hunyuan_config.toml"

    toml_content = f"""# Tetsuo AI - HunyuanVideo LoRA Training Config
# For use with Musubi Tuner (https://github.com/kohya-tech/musubi-tuner)

[general]
mixed_precision = "{config['mixed_precision']}"
seed = {config['seed']}

[dataset]
batch_size = {config['batch_size']}
enable_bucket = true
resolution = "{config['resolution']}"

  [[dataset.subsets]]
  image_directory = "{DATASET_DIR / f'10_{TRIGGER_WORD}'}"
  caption_extension = ".txt"
  num_repeats = 10

[training]
output_dir = "{OUTPUT_DIR}"
output_name = "{config['output_name']}"
save_every_n_steps = {config['save_every_n_steps']}
max_train_steps = {config['steps']}
gradient_accumulation_steps = {config['grad_acc']}
gradient_checkpointing = true
learning_rate = {config['learning_rate']}
optimizer_type = "{config['optimizer']}"
lr_scheduler = "cosine"
lr_warmup_steps = 100

[network]
network_module = "networks.lora"
network_dim = {config['rank']}
network_alpha = {config['rank'] // 2}

[video]
video_length = {config['video_length']}
"""

    config_path.write_text(toml_content, encoding="utf-8")
    print(f"\nConfig saved to: {config_path}")


def train(config: dict):
    """Launch Musubi Tuner training."""
    if not MUSUBI_DIR.exists():
        print("Error: Musubi Tuner not found. Run 'setup' first.")
        sys.exit(1)

    config_path = CONFIG_DIR / "tetsuo_hunyuan_config.toml"
    if not config_path.exists():
        print("Generating training config...")
        generate_config(config)

    # Check for dataset
    train_dir = DATASET_DIR / f"10_{TRIGGER_WORD}"
    if not train_dir.exists() or not any(train_dir.iterdir()):
        print(f"Error: No training data found in {train_dir}")
        print("Run 'prepare' first and add your training data.")
        sys.exit(1)

    print("Launching HunyuanVideo LoRA training...")
    print(f"  Config: {config_path}")
    print(f"  Steps: {config['steps']}")
    print(f"  Rank: {config['rank']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Output: {OUTPUT_DIR / config['output_name']}.safetensors")

    # Build training command
    train_script = MUSUBI_DIR / "train.py"
    cmd = [
        sys.executable, str(train_script),
        "--config", str(config_path),
    ]

    # Add overrides from CLI
    if config.get("steps") != DEFAULTS["steps"]:
        cmd.extend(["--max_train_steps", str(config["steps"])])
    if config.get("rank") != DEFAULTS["rank"]:
        cmd.extend(["--network_dim", str(config["rank"])])
    if config.get("learning_rate") != DEFAULTS["learning_rate"]:
        cmd.extend(["--learning_rate", str(config["learning_rate"])])

    print(f"\nCommand: {' '.join(cmd)}")
    print("=" * 60)

    run_cmd(cmd, cwd=str(MUSUBI_DIR), check=False)

    # Check output
    output_file = OUTPUT_DIR / f"{config['output_name']}.safetensors"
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\nTraining complete!")
        print(f"  Output: {output_file} ({size_mb:.1f} MB)")
        print(f"\nNext: Push to HuggingFace:")
        print(f"  python scripts/push_to_hf.py --lora_path {output_file} --model_type hunyuan_video")
    else:
        print(f"\nTraining may still be running or failed.")
        print(f"  Check {OUTPUT_DIR} for output files.")


def main():
    parser = argparse.ArgumentParser(description="Tetsuo AI HunyuanVideo Training")
    subparsers = parser.add_subparsers(dest="command")

    # Setup
    subparsers.add_parser("setup", help="Clone and install Musubi Tuner")

    # Prepare
    subparsers.add_parser("prepare", help="Prepare video dataset")

    # Train
    train_parser = subparsers.add_parser("train", help="Launch training")
    train_parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    train_parser.add_argument("--rank", type=int, default=DEFAULTS["rank"])
    train_parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    train_parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    train_parser.add_argument("--video_length", type=int, default=DEFAULTS["video_length"])
    train_parser.add_argument("--output_name", default=DEFAULTS["output_name"])

    args = parser.parse_args()

    if args.command == "setup":
        setup_musubi()
    elif args.command == "prepare":
        prepare_dataset()
    elif args.command == "train":
        config = dict(DEFAULTS)
        config.update({
            "steps": args.steps,
            "rank": args.rank,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "video_length": args.video_length,
            "output_name": args.output_name,
        })
        train(config)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. python scripts/tetsuo_video_train.py setup")
        print("  2. python scripts/tetsuo_video_train.py prepare")
        print("  3. Add videos/images to input/tetsuo_video_dataset/10_tetsuo_character/")
        print("  4. python scripts/tetsuo_video_train.py train")


if __name__ == "__main__":
    main()
