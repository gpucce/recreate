from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import pipeline

DEFAULT_VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MAX_SIZE = 512
DEFAULT_MAX_NEW_TOKENS = 2048
DIMENSION_MULTIPLE = 8
SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}

PROMPT_REQUEST = """
Write a prompt describing the image to enable an image generation model to accurately replicate it.
Include all the elements that are relevant to replicate the image in its content and in its visual appearance.
Relevant elements are the subjects, their pose, look, glance, the detailed composition of the image, camera angle, lighting, colors, materials, textures, background, image style, white balance, color saturation, palette, grain, focus, blur, mood, any legible text, number, or symbol, the nature and quality of the image (a photo made with an old phone, a smartphone, a reflex camera, in an open setting, in studio). Use at most 350 words.
""".strip()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a text prompt from an input image using a local vision-language model.",
    )
    parser.add_argument("input_image", type=Path, help="Path to the image to describe.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output prompt path. Defaults to <input_stem>_recreated.prompt.txt beside the input.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output prompt file.",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip processing if output prompt file already exists (ignored if --force is set).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print generated prompt(s) to stdout instead of saving prompt files.",
    )
    parser.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help=f"Name of the vision model used to describe the input image. Default: {DEFAULT_VISION_MODEL}",
    )
    parser.add_argument(
        "--max-size",
        type=positive_int,
        default=DEFAULT_MAX_SIZE,
        help=f"Resize the longest side before prompt generation. Default: {DEFAULT_MAX_SIZE}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto",
    )
    return parser


def default_prompt_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_recreated.prompt.txt")


def default_prompt_output_name(input_path: Path) -> str:
    return f"{input_path.stem}_recreated.prompt.txt"


def default_prompt_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}.prompts")


def validate_paths(input_path: Path, output_path: Path, force: bool) -> None:
    if not input_path.exists():
        raise ValueError(f"input image does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"input path is not a file: {input_path}")
    if not output_path.parent.exists():
        raise ValueError(f"output directory does not exist: {output_path.parent}")
    if output_path.exists() and not force:
        raise ValueError(f"refusing to overwrite existing file: {output_path}. Use --force to overwrite.")


def is_supported_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def list_supported_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if is_supported_image_file(path))


def resolve_device_and_dtype(requested_device: str) -> tuple[str, Any]:
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false")
    if device == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise ValueError("MPS was requested, but torch.backends.mps.is_available() is false")

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    return device, dtype


def round_to_multiple(value: float) -> int:
    rounded = int(round(value / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
    return max(DIMENSION_MULTIPLE, rounded)


def resize_for_diffusion(image: Any, max_size: int) -> Any:
    target_long_side = max_size - (max_size % DIMENSION_MULTIPLE)
    if target_long_side < DIMENSION_MULTIPLE:
        raise ValueError(f"--max-size must be at least {DIMENSION_MULTIPLE}")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("input image has invalid dimensions")

    scale = target_long_side / max(width, height)
    resized_width = round_to_multiple(width * scale)
    resized_height = round_to_multiple(height * scale)

    if width >= height:
        resized_width = target_long_side
    else:
        resized_height = target_long_side

    return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def load_image(input_path: Path, max_size: int) -> Any:
    try:
        image = Image.open(input_path).convert("RGB")
    except OSError as exc:
        raise ValueError(f"could not open input image: {input_path}") from exc
    return resize_for_diffusion(image, max_size)


def extract_generated_text(outputs: Any) -> str:
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, list) and outputs:
        return extract_generated_text(outputs[0])
    if isinstance(outputs, dict):
        if "generated_text" in outputs:
            return extract_generated_text(outputs["generated_text"])
        if "content" in outputs:
            return extract_generated_text(outputs["content"])
        if outputs.get("role") == "assistant" and "text" in outputs:
            return str(outputs["text"])
    return str(outputs)


def clean_prompt(prompt: str) -> str:
    cleaned = prompt.strip()
    for prefix in ("Prompt:", "Image generation prompt:", "Reconstruction prompt:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return " ".join(cleaned.split())


def load_vlm_pipeline(
    model_id: str,
    device: str,
    dtype: Any,
    device_map: str | None = None,
    max_memory: dict[int, str] | None = None,
) -> Any:
    device_arg: int | str
    if device == "cuda":
        device_arg = 0
    elif device == "mps":
        device_arg = device
    else:
        device_arg = -1

    pipeline_attempts = [
        {
            "task": "image-text-to-text",
            "model": model_id,
            "device": device_arg,
            "dtype": dtype,
            "trust_remote_code": True,
            "model_kwargs": {"low_cpu_mem_usage": False},
        },
        {
            "task": "image-text-to-text",
            "model": model_id,
            "device": device_arg,
            "trust_remote_code": True,
            "model_kwargs": {"low_cpu_mem_usage": False},
        },
        {
            "task": "image-to-text",
            "model": model_id,
            "device": device_arg,
            "dtype": dtype,
            "trust_remote_code": True,
            "model_kwargs": {"low_cpu_mem_usage": False},
        },
        {
            "task": "image-to-text",
            "model": model_id,
            "device": device_arg,
            "trust_remote_code": True,
            "model_kwargs": {"low_cpu_mem_usage": False},
        },
    ]
    if device_map is not None:
        # Shard the model over several GPUs (e.g. one too big for a single card) instead of
        # placing it whole on one device; max_memory caps what each GPU may take.
        for pipe_kwargs in pipeline_attempts:
            del pipe_kwargs["device"]
            pipe_kwargs["device_map"] = device_map
            pipe_kwargs["model_kwargs"] = {"max_memory": max_memory} if max_memory else {}

    last_error: Exception | None = None
    vlm = None
    for pipe_kwargs in pipeline_attempts:
        try:
            vlm = pipeline(**pipe_kwargs)
            break
        except (TypeError, ValueError, OSError) as exc:
            last_error = exc

    if vlm is None:
        raise ValueError(
            "failed to load the vision model. If you are using SmolVLM, upgrade "
            "transformers to the latest version. "
            f"Underlying error: {last_error}"
        )

    for generation_config in (
        getattr(vlm, "generation_config", None),
        getattr(vlm.model, "generation_config", None),
    ):
        if generation_config is not None:
            generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
            generation_config.max_length = None
    return vlm


def generate_reconstruction_prompt(
    image: Any, vlm: Any
) -> str:

    message_variants = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_REQUEST},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": PROMPT_REQUEST},
                ],
            }
        ],
    ]

    outputs = None
    for messages in message_variants:
        try:
            outputs = vlm(
                text=messages,
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )
            break
        except (TypeError, ValueError):
            continue

    if outputs is None:
        try:
            outputs = vlm(
                images=[image],
                prompt=PROMPT_REQUEST,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            outputs = vlm(
                image,
                prompt=PROMPT_REQUEST,
                clean_up_tokenization_spaces=False,
            )

    prompt = clean_prompt(extract_generated_text(outputs))
    if not prompt:
        raise ValueError("vision model returned an empty prompt")
    return prompt


def main(args: argparse.Namespace) -> None:
    input_path = args.input_image
    vision_model = args.vision_model
    device, dtype = resolve_device_and_dtype(args.device)
    print(f"Using device: {device}")

    if input_path.is_dir():
        if args.output is not None:
            raise ValueError("--output cannot be used when input_image is a directory")
        if args.print and (args.force or args.skip):
            raise ValueError("--force and --skip cannot be used with --print")

        output_dir = None
        if not args.print:
            output_dir = default_prompt_output_dir(input_path)
            output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_supported_images(input_path)
        if not image_paths:
            raise ValueError(f"no supported images found in directory: {input_path}")

        jobs: list[tuple[Path, Path | None]] = []
        for image_path in image_paths:
            output_path = None if args.print else output_dir / default_prompt_output_name(image_path)
            if output_path is not None and output_path.exists():
                if args.force:
                    pass  # Overwrite
                elif args.skip:
                    print(f"Skipping existing file: {output_path}")
                    continue
                else:
                    raise ValueError(f"refusing to overwrite existing file: {output_path}. Use --force to overwrite or --skip to skip.")

            jobs.append((image_path, output_path))

        if not jobs:
            print("Nothing to process after applying --skip/--force rules.")
            return

        print(f"Loading vision model: {vision_model}")
        vlm = load_vlm_pipeline(model_id=vision_model, device=device, dtype=dtype)
        print(f"Vision model loaded: {vision_model} on {device}")
        print(f"Processing {len(jobs)} image(s) from: {input_path}")

        for image_path, output_path in jobs:
            source_image = load_image(image_path, args.max_size)
            prompt = generate_reconstruction_prompt(source_image, vlm)
            if args.print:
                print(f"=== {image_path.name} ===")
                print(prompt)
            else:
                output_path.write_text(prompt + "\n", encoding="utf-8")
                print(f"Saved prompt: {output_path}")
        return

    output_path = args.output or default_prompt_output_path(input_path)
    if args.print and (args.output is not None or args.force or args.skip):
        raise ValueError("--output, --force, and --skip cannot be used with --print")
    if not args.print:
        if output_path.exists():
            if args.force:
                pass  # Overwrite
            elif args.skip:
                print(f"Skipping existing file: {output_path}")
                return
            else:
                validate_paths(input_path, output_path, args.force)
        else:
            validate_paths(input_path, output_path, args.force)
    elif not input_path.exists() or not input_path.is_file():
        raise ValueError(f"input image does not exist or is not a file: {input_path}")

    source_image = load_image(input_path, args.max_size)

    print(f"Loading vision model: {vision_model}")
    vlm = load_vlm_pipeline(model_id=vision_model, device=device, dtype=dtype)
    print(f"Vision model loaded: {vision_model} on {device}")
    prompt = generate_reconstruction_prompt(source_image, vlm)
    print(f"Generated prompt ({len(prompt)} characters).")

    if args.print:
        print(prompt)
    else:
        output_path.write_text(prompt + "\n", encoding="utf-8")
        print(f"Saved prompt: {output_path}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
