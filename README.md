# Tetsuo AI - Image & Video Generation Model

Uncensored, hyper-realistic LoRA adapters for [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (images) and [WAN 2.2](https://github.com/Wan-Video/Wan2.2) (video). Built for high-definition generative output with no content restrictions, aligned with Tetsuo AI's open-source ethos.

## Uncensored and Open Source

These models are trained without safety filters or content restrictions. We believe generative AI should be open, transparent, and free from arbitrary censorship. Released under Apache 2.0 for personal and commercial use.

## Hyper-Realistic Output

Fine-tuned with a focus on high-definition, photorealistic rendering. The models produce sharp, cinematic output with realistic lighting, textures, depth of field, and strong temporal consistency (video). Trained in bfloat16 for maximum detail and visual fidelity.

## Character Training

The initial training run focuses on the Tetsuo AI character, a cyberpunk anime female rendered in two modes:

1. **High-contrast B&W manga** - Heavy ink work, aggressive linework, seinen manga aesthetic with mechanical details
2. **3D cyberpunk portraiture** - Volumetric lighting, neon blue highlights, rain, dark metallic surfaces

Trigger word: `tetsuo_character`

## Quick Start

### Prerequisites
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed
- GPU with 24GB+ VRAM (RTX 3090/4090/A100)
- Python 3.10+
- FLUX.1-dev model files downloaded

### 1. Prepare Dataset
```bash
# Put raw training images in dataset/raw/
python scripts/tetsuo_dataset_prep.py prepare --input_dir dataset/raw --output_dir dataset/train --auto_caption

# Validate
python scripts/tetsuo_dataset_prep.py validate --dataset_dir dataset/train
```

### 2. Train Flux LoRA (Images)
```bash
# Copy dataset to ComfyUI input folder
cp -r dataset/train /path/to/ComfyUI/input/tetsuo_dataset

# Generate and submit training workflow
python scripts/tetsuo_train_config.py submit --steps 1500 --rank 32 --lr 1e-4
```

Or load `workflows/tetsuo_flux_train.json` directly in the ComfyUI web UI.

### 3. Train WAN 2.2 LoRA (Video)
```bash
# Same ComfyUI built-in training, just different model
python scripts/tetsuo_train_config.py submit --model wan --steps 1000
```

Or load `workflows/tetsuo_wan_train.json` in the ComfyUI web UI.

### 4. Push to HuggingFace
```bash
python scripts/push_to_hf.py --lora_path output/tetsuo_flux_v1.safetensors --model_type flux
python scripts/push_to_hf.py --lora_path output/tetsuo_wan_v1.safetensors --model_type wan
```

## Training Config

| Parameter | Flux (Image) | WAN 2.2 (Video) |
|-----------|-------------|-----------------|
| Base Model | FLUX.1-dev | WAN 2.2 T2V 14B |
| Method | LoRA | LoRA |
| Rank | 32 | 32 |
| Steps | 1500 | 1000 |
| Learning Rate | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Dtype | bf16 | bf16 |
| Batch Size | 1 (eff. 4) | 1 (eff. 4) |
| Bucket Mode | Yes | Yes |
| Training | ComfyUI built-in | ComfyUI built-in |
| Trigger Word | `tetsuo_character` | `tetsuo_character` |

## Project Structure

```
tetsuo-model/
├── scripts/
│   ├── tetsuo_dataset_prep.py    # Dataset validation, resizing, captioning
│   ├── tetsuo_train_config.py    # ComfyUI workflow generation & submission (Flux + WAN 2.2)
│   └── push_to_hf.py            # HuggingFace upload with model cards
├── workflows/
│   ├── tetsuo_flux_train.json    # ComfyUI Flux training workflow (API format)
│   ├── tetsuo_wan_train.json     # ComfyUI WAN 2.2 training workflow (API format)
│   └── tetsuo_flux_inference.json # ComfyUI inference workflow
├── dataset/
│   ├── raw/                       # Raw training images (add yours here)
│   └── train/                     # Processed training-ready dataset
└── output/                        # Trained LoRA weights land here
```

## Using the Trained Model

### ComfyUI
Place the `.safetensors` file in `ComfyUI/models/loras/` and use the **LoRA Loader** node. Always include `tetsuo_character` in your prompt.

### Diffusers (Python)
```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("tetsuo-ai/tetsuo-flux-lora")
pipe.to("cuda")

image = pipe("tetsuo_character, cyberpunk portrait, platinum silver hair, neon blue lighting, 8k").images[0]
```

## Example Prompts

```
tetsuo_character, cyberpunk portrait, volumetric lighting, neon blue highlights, rain, black tactical suit, cinematic, 8k
```

```
tetsuo_character, manga illustration, black and white, heavy ink linework, seinen style, mechanical details, intense eyes
```

```
tetsuo_character, blade runner cityscape, TETSUO CORP neon sign, rain soaked streets, ambient orange and blue lighting
```

```
tetsuo_character, performing on stage, electric guitar, neon-lit venue, detached intensity, shallow depth of field
```

## License

Apache 2.0

## Links

- [Tetsuo AI](https://www.tetsuocorp.com)
- [Twitter/X](https://x.com/tetsuoai)
- [GitHub](https://github.com/tetsuo-ai)
