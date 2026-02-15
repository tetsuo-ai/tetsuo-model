"""
Tetsuo AI - Push Trained Models to HuggingFace

Uploads trained LoRA safetensors to HuggingFace with auto-generated model cards.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login

Usage:
    # Push Flux LoRA:
    python scripts/push_to_hf.py --lora_path output/loras/tetsuo_v1.safetensors --model_type flux

    # Push Hunyuan Video LoRA:
    python scripts/push_to_hf.py --lora_path output/loras/tetsuo_hunyuan_v1.safetensors --model_type hunyuan_video

    # Custom repo name:
    python scripts/push_to_hf.py --lora_path output/loras/tetsuo_v1.safetensors --repo_name my-org/tetsuo-flux-lora
"""

import argparse
import sys
from pathlib import Path


TRIGGER_WORD = "tetsuo_character"

MODEL_CARDS = {
    "flux": {
        "base_model": "black-forest-labs/FLUX.1-dev",
        "pipeline_tag": "text-to-image",
        "tags": ["flux", "lora", "text-to-image", "cyberpunk", "anime", "manga", "character"],
        "default_repo": "tetsuo-ai-flux-lora",
    },
    "hunyuan_video": {
        "base_model": "tencent/HunyuanVideo",
        "pipeline_tag": "text-to-video",
        "tags": ["hunyuan-video", "lora", "text-to-video", "cyberpunk", "anime", "character"],
        "default_repo": "tetsuo-ai-hunyuan-video-lora",
    },
}


def generate_model_card(model_type: str, lora_path: str, training_info: dict | None = None) -> str:
    """Generate a HuggingFace model card README."""
    info = MODEL_CARDS[model_type]
    lora_filename = Path(lora_path).name

    tags_yaml = "\n".join(f"- {tag}" for tag in info["tags"])

    card = f"""---
base_model: {info['base_model']}
pipeline_tag: {info['pipeline_tag']}
license: apache-2.0
tags:
{tags_yaml}
- tetsuo
- uncensored
- open-source
widget:
- text: "{TRIGGER_WORD}, cyberpunk portrait, platinum silver hair, pale skin, intense eyes, neon blue lighting, cinematic, 8k"
---

# Tetsuo AI {'Image' if model_type == 'flux' else 'Video'} Generation LoRA

A fine-tuned LoRA adapter for {'FLUX.1-dev' if model_type == 'flux' else 'HunyuanVideo'} that generates the **Tetsuo AI** character - a cyberpunk anime female with platinum/silver hair, pale skin, and intense eyes.

## Character Description

The Tetsuo character exists in two rendering modes:

1. **High-contrast B&W manga**: Heavy ink work, aggressive linework, seinen manga aesthetic with mechanical details
2. **3D cyberpunk portraiture**: Moody, cinematic, volumetric lighting with neon blue highlights, rain, and dark metallic surfaces

The visual DNA draws from Akira, Ghost in the Shell, Blame!, and Blade Runner.

## Usage

**Trigger word:** `{TRIGGER_WORD}`

### ComfyUI

1. Place `{lora_filename}` in your `ComfyUI/models/loras/` directory
2. Use the **LoRA Loader** node to apply it to your {'Flux' if model_type == 'flux' else 'HunyuanVideo'} model
3. Include `{TRIGGER_WORD}` in your prompt

### Diffusers (Python)

```python
from diffusers import {'FluxPipeline' if model_type == 'flux' else 'HunyuanVideoPipeline'}

pipe = {'FluxPipeline' if model_type == 'flux' else 'HunyuanVideoPipeline'}.from_pretrained(
    "{info['base_model']}",
    torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights("YOUR_USERNAME/{info['default_repo']}")
pipe.to("cuda")

{'image' if model_type == 'flux' else 'video'} = pipe(
    "{TRIGGER_WORD}, cyberpunk portrait, platinum silver hair, neon blue lighting, cinematic, 8k",
    num_inference_steps={'20' if model_type == 'flux' else '30'},
).images[0]
```

## Example Prompts

### Cyberpunk Mode
```
{TRIGGER_WORD}, cyberpunk portrait, volumetric lighting, neon blue highlights, rain, dark metallic surfaces, black tactical suit, platinum silver hair, pale skin, cinematic, high detail, 8k
```

### Manga Mode
```
{TRIGGER_WORD}, manga illustration, black and white, heavy ink linework, seinen style, detailed hatching, sharp platinum silver hair, intense eyes, mechanical details
```

### Blade Runner Scene
```
{TRIGGER_WORD}, walking through cyberpunk cityscape, neon signs, TETSUO CORP glowing in background, rain soaked streets, cinematic composition, ambient orange and blue lighting
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | {info['base_model']} |
| Method | LoRA (Low-Rank Adaptation) |
| Trigger Word | `{TRIGGER_WORD}` |
| Training Dtype | bfloat16 |
| Framework | ComfyUI built-in training |

## About Tetsuo AI

Tetsuo AI is an open-source project building uncensored generative AI models. This model is released under the Apache 2.0 license as part of our commitment to open-source AI.

## License

Apache 2.0 - Free for personal and commercial use.

---

*Built with [ComfyUI](https://github.com/comfyanonymous/ComfyUI)*
"""
    return card


def push_to_hub(lora_path: str, model_type: str, repo_name: str | None = None,
                private: bool = False):
    """Push LoRA weights and model card to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("  pip install huggingface_hub")
        print("  huggingface-cli login")
        sys.exit(1)

    lora_file = Path(lora_path)
    if not lora_file.exists():
        print(f"Error: LoRA file not found: {lora_path}")
        sys.exit(1)

    api = HfApi()

    # Get username
    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception:
        print("Error: Not logged in to HuggingFace.")
        print("  Run: huggingface-cli login")
        sys.exit(1)

    # Determine repo name
    if repo_name is None:
        info = MODEL_CARDS[model_type]
        repo_name = f"{username}/{info['default_repo']}"
    elif "/" not in repo_name:
        repo_name = f"{username}/{repo_name}"

    print(f"Pushing to HuggingFace: {repo_name}")
    print(f"  LoRA file: {lora_file.name} ({lora_file.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  Model type: {model_type}")
    print(f"  Private: {private}")

    # Create repo
    try:
        create_repo(repo_name, repo_type="model", private=private, exist_ok=True)
        print(f"  Repository created/found: {repo_name}")
    except Exception as e:
        print(f"  Warning: Could not create repo: {e}")

    # Generate model card
    model_card = generate_model_card(model_type, lora_path)
    readme_path = lora_file.parent / "README.md"
    readme_path.write_text(model_card, encoding="utf-8")

    # Upload files
    print("  Uploading LoRA weights...")
    api.upload_file(
        path_or_fileobj=str(lora_file),
        path_in_repo=lora_file.name,
        repo_id=repo_name,
        repo_type="model",
    )

    print("  Uploading model card...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model",
    )

    # Clean up temp readme
    readme_path.unlink(missing_ok=True)

    print(f"\nDone! Model published at:")
    print(f"  https://huggingface.co/{repo_name}")
    print(f"\nUsers can load with:")
    print(f"  ComfyUI: Place in models/loras/ and use LoRA Loader node")
    print(f"  Python:  pipe.load_lora_weights(\"{repo_name}\")")


def main():
    parser = argparse.ArgumentParser(description="Push Tetsuo AI models to HuggingFace")
    parser.add_argument("--lora_path", required=True, help="Path to trained LoRA .safetensors file")
    parser.add_argument("--model_type", choices=["flux", "hunyuan_video"], required=True,
                        help="Type of base model the LoRA was trained on")
    parser.add_argument("--repo_name", default=None,
                        help="HuggingFace repo name (default: username/tetsuo-ai-{type}-lora)")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()
    push_to_hub(args.lora_path, args.model_type, args.repo_name, args.private)


if __name__ == "__main__":
    main()
