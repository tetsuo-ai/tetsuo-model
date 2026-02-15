"""
Tetsuo AI Dataset Preparation Script

Prepares training images for Flux LoRA fine-tuning via ComfyUI.
Handles validation, resizing, caption generation, and deduplication.

Usage:
    python scripts/tetsuo_dataset_prep.py --input_dir input/tetsuo_raw --output_dir input/tetsuo_dataset
    python scripts/tetsuo_dataset_prep.py --input_dir input/tetsuo_raw --output_dir input/tetsuo_dataset --auto_caption
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from PIL import Image

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TRIGGER_WORD = "tetsuo_character"
TARGET_RESOLUTIONS = [
    (1024, 1024),
    (768, 1344),   # portrait
    (1344, 768),   # landscape
    (896, 1152),   # tall portrait
    (1152, 896),   # wide landscape
]

# Caption templates by detected style
MANGA_KEYWORDS = ["manga", "ink", "bw", "black_white", "linework", "sketch"]
CYBERPUNK_KEYWORDS = ["cyber", "neon", "3d", "render", "rain", "city", "glow"]

MANGA_CAPTIONS = [
    f"{TRIGGER_WORD}, manga illustration, black and white, heavy ink linework, seinen style, detailed hatching, sharp platinum silver hair, pale skin, intense eyes, mechanical details",
    f"{TRIGGER_WORD}, manga panel, monochrome, aggressive linework, speed lines, cyberpunk manga, biomechanical armor, female character, short silver hair",
    f"{TRIGGER_WORD}, black and white manga, high contrast, detailed ink work, shounen crossover, mecha details, exposed servos, combat ready pose",
]

CYBERPUNK_CAPTIONS = [
    f"{TRIGGER_WORD}, cyberpunk portrait, volumetric lighting, neon blue highlights, rain, dark metallic surfaces, black tactical suit, platinum silver hair, pale skin, cinematic, high detail, 8k",
    f"{TRIGGER_WORD}, 3d rendered, cyberpunk cityscape, holographic screens, blade runner aesthetic, cold blue lighting, bokeh, shallow depth of field, female character, silver bob hair",
    f"{TRIGGER_WORD}, cinematic cyberpunk, neon reflections, dark atmosphere, armored suit, teal cyan accents, high definition, photorealistic, detailed face, intense expression",
]

GENERAL_CAPTIONS = [
    f"{TRIGGER_WORD}, a woman with platinum silver hair, pale skin, intense eyes, cyberpunk aesthetic, high detail, 8k quality",
]


def get_image_hash(filepath: str) -> str:
    """Compute perceptual hash for deduplication."""
    img = Image.open(filepath).convert("RGB").resize((64, 64), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(sum(p) for p in pixels) / (len(pixels) * 3)
    bits = "".join("1" if sum(p) / 3 > avg else "0" for p in pixels)
    return hashlib.md5(bits.encode()).hexdigest()


def find_best_resolution(width: int, height: int) -> tuple[int, int]:
    """Find the closest target resolution for the given aspect ratio."""
    aspect = width / height
    best = min(TARGET_RESOLUTIONS, key=lambda r: abs(r[0] / r[1] - aspect))
    return best


def detect_style(filename: str) -> str:
    """Guess the style from filename/path."""
    name_lower = filename.lower()
    if any(kw in name_lower for kw in MANGA_KEYWORDS):
        return "manga"
    if any(kw in name_lower for kw in CYBERPUNK_KEYWORDS):
        return "cyberpunk"
    return "general"


def detect_style_from_image(img: Image.Image) -> str:
    """Detect style from image content (B&W vs color)."""
    small = img.convert("RGB").resize((64, 64), Image.LANCZOS)
    pixels = list(small.getdata())
    # Check if image is mostly grayscale
    gray_count = sum(1 for r, g, b in pixels if abs(r - g) < 15 and abs(g - b) < 15 and abs(r - b) < 15)
    gray_ratio = gray_count / len(pixels)
    if gray_ratio > 0.85:
        return "manga"
    return "cyberpunk"


def resize_and_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize image to target resolution with center crop."""
    w, h = img.size
    target_aspect = target_w / target_h
    img_aspect = w / h

    if img_aspect > target_aspect:
        # Image is wider - fit height, crop width
        new_h = target_h
        new_w = int(new_h * img_aspect)
    else:
        # Image is taller - fit width, crop height
        new_w = target_w
        new_h = int(new_w / img_aspect)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    return img


def generate_caption(style: str, index: int) -> str:
    """Generate a caption based on style."""
    if style == "manga":
        return MANGA_CAPTIONS[index % len(MANGA_CAPTIONS)]
    elif style == "cyberpunk":
        return CYBERPUNK_CAPTIONS[index % len(CYBERPUNK_CAPTIONS)]
    return GENERAL_CAPTIONS[index % len(GENERAL_CAPTIONS)]


def process_dataset(input_dir: str, output_dir: str, auto_caption: bool = False,
                    repeats: int = 10, min_size: int = 512):
    """Process raw images into training-ready dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print(f"Please create it and add your Tetsuo character images.")
        sys.exit(1)

    # Create output structure (kohya-style with repeats)
    train_dir = output_path / f"{repeats}_{TRIGGER_WORD}"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Collect all valid images
    image_files = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                image_files.append(os.path.join(root, f))

    if not image_files:
        print(f"No valid images found in '{input_dir}'.")
        print(f"Supported formats: {', '.join(VALID_EXTENSIONS)}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in '{input_dir}'")

    # Deduplication
    seen_hashes = {}
    unique_files = []
    for f in image_files:
        try:
            h = get_image_hash(f)
            if h not in seen_hashes:
                seen_hashes[h] = f
                unique_files.append(f)
            else:
                print(f"  Skipping duplicate: {os.path.basename(f)} (same as {os.path.basename(seen_hashes[h])})")
        except Exception as e:
            print(f"  Warning: Could not process {f}: {e}")

    print(f"  {len(unique_files)} unique images after deduplication")

    # Process each image
    manga_count = 0
    cyber_count = 0
    processed = 0

    for i, filepath in enumerate(unique_files):
        try:
            img = Image.open(filepath).convert("RGB")
            w, h = img.size

            # Skip images that are too small
            if w < min_size or h < min_size:
                print(f"  Skipping {os.path.basename(filepath)}: too small ({w}x{h}, min {min_size})")
                continue

            # Find best target resolution
            target_w, target_h = find_best_resolution(w, h)

            # Resize and crop
            img = resize_and_crop(img, target_w, target_h)

            # Detect style
            style = detect_style(os.path.basename(filepath))
            if style == "general":
                style = detect_style_from_image(img)

            if style == "manga":
                manga_count += 1
            else:
                cyber_count += 1

            # Save processed image
            out_name = f"tetsuo_{processed:04d}"
            img.save(train_dir / f"{out_name}.png", "PNG", quality=100)

            # Generate or copy caption
            caption_path = train_dir / f"{out_name}.txt"
            existing_caption = Path(filepath).with_suffix(".txt")

            if existing_caption.exists():
                # Use existing caption but prepend trigger word if missing
                caption = existing_caption.read_text(encoding="utf-8").strip()
                if TRIGGER_WORD not in caption:
                    caption = f"{TRIGGER_WORD}, {caption}"
                caption_path.write_text(caption, encoding="utf-8")
                print(f"  [{style}] {os.path.basename(filepath)} -> {out_name}.png ({target_w}x{target_h}) [existing caption]")
            elif auto_caption:
                caption = generate_caption(style, processed)
                caption_path.write_text(caption, encoding="utf-8")
                print(f"  [{style}] {os.path.basename(filepath)} -> {out_name}.png ({target_w}x{target_h}) [auto caption]")
            else:
                # Write placeholder caption for user to edit
                caption = generate_caption(style, processed)
                caption_path.write_text(f"# EDIT THIS CAPTION\n{caption}", encoding="utf-8")
                print(f"  [{style}] {os.path.basename(filepath)} -> {out_name}.png ({target_w}x{target_h}) [needs caption edit]")

            processed += 1

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

    print(f"\nDataset prepared: {processed} images")
    print(f"  Manga style: {manga_count}")
    print(f"  Cyberpunk style: {cyber_count}")
    print(f"  Output: {train_dir}")
    print(f"  Repeats: {repeats}x per epoch")
    print(f"\nTrigger word: '{TRIGGER_WORD}'")

    if not auto_caption:
        print(f"\nIMPORTANT: Review and edit caption files in {train_dir}")
        print("Each .txt file should describe the corresponding image.")
        print(f"Always include '{TRIGGER_WORD}' at the start of each caption.")

    return processed


def validate_dataset(dataset_dir: str):
    """Validate a prepared dataset is ready for training."""
    dataset_path = Path(dataset_dir)
    issues = []

    # Find training subdirectories
    train_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not train_dirs:
        # Check for images directly in the folder
        images = [f for f in dataset_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        if images:
            train_dirs = [dataset_path]
        else:
            print("Error: No training subdirectories or images found.")
            return False

    total_images = 0
    total_captions = 0
    missing_captions = []
    placeholder_captions = []

    for train_dir in train_dirs:
        images = [f for f in train_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        total_images += len(images)

        for img_file in images:
            caption_file = img_file.with_suffix(".txt")
            if caption_file.exists():
                total_captions += 1
                content = caption_file.read_text(encoding="utf-8")
                if content.startswith("# EDIT THIS"):
                    placeholder_captions.append(str(caption_file))
                if TRIGGER_WORD not in content:
                    issues.append(f"Missing trigger word in {caption_file.name}")
            else:
                missing_captions.append(str(img_file))

    print(f"Dataset Validation: {dataset_dir}")
    print(f"  Images: {total_images}")
    print(f"  Captions: {total_captions}")
    print(f"  Missing captions: {len(missing_captions)}")
    print(f"  Placeholder captions: {len(placeholder_captions)}")

    if missing_captions:
        print("\n  Images missing captions:")
        for f in missing_captions:
            print(f"    - {f}")
        issues.append(f"{len(missing_captions)} images missing captions")

    if placeholder_captions:
        print("\n  Captions that need editing:")
        for f in placeholder_captions:
            print(f"    - {f}")
        issues.append(f"{len(placeholder_captions)} captions still have placeholder text")

    if issues:
        print(f"\n  Issues found: {len(issues)}")
        for issue in issues:
            print(f"    - {issue}")
        return False

    print("\n  Dataset is ready for training!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetsuo AI Dataset Preparation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Prepare raw images for training")
    prep_parser.add_argument("--input_dir", type=str, default="input/tetsuo_raw",
                             help="Directory containing raw training images")
    prep_parser.add_argument("--output_dir", type=str, default="input/tetsuo_dataset",
                             help="Output directory for processed dataset")
    prep_parser.add_argument("--auto_caption", action="store_true",
                             help="Auto-generate captions (otherwise creates placeholders)")
    prep_parser.add_argument("--repeats", type=int, default=10,
                             help="Number of repeats per image per epoch (kohya-style)")
    prep_parser.add_argument("--min_size", type=int, default=512,
                             help="Minimum image dimension (skip smaller)")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a prepared dataset")
    val_parser.add_argument("--dataset_dir", type=str, default="input/tetsuo_dataset",
                            help="Dataset directory to validate")

    args = parser.parse_args()

    if args.command == "prepare":
        process_dataset(args.input_dir, args.output_dir, args.auto_caption,
                        args.repeats, args.min_size)
    elif args.command == "validate":
        validate_dataset(args.dataset_dir)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. Put raw images in input/tetsuo_raw/")
        print("  2. Run: python scripts/tetsuo_dataset_prep.py prepare --auto_caption")
        print("  3. Review captions in input/tetsuo_dataset/10_tetsuo_character/")
        print("  4. Run: python scripts/tetsuo_dataset_prep.py validate")
