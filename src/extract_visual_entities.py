"""
Extract visual entities: given a JSON file storing a list of entities possibly
present in an image (as produced by src/structure_description.py), run a text-prompt
segmentation model (SAM3) on the image to determine which entities are actually
present, and return a new JSON with only those entities plus the extracted masks.

The input JSON is expected to have the following structure. The entities to check
are taken from the "entities" list; the "scene" block is passed through unchanged:

{
  "scene": {
    "background": "dense green trees and foliage with dappled sunlight filtering through the leaves",
    "lighting": "natural, golden, and diffused with a slightly warm white balance",
    "camera_angle": "low, rear-facing angle",
    "style": "realistic, high-resolution photograph",
    "mood": "serene, loving, and timeless",
    "color_palette": "vibrant greens, cheerful yellow, soft muted tones, and warm tones"
  },
  "entities": [
    {
      "type": "person",
      "attributes": { "age": "senior", "gender": "female", ... },
      "id": "person_1",
      "count": 1
    },
    {
      "type": "camera",
      "attributes": { "type": "reflex camera", ... },
      "id": "camera_1",
      "count": 1
    }
  ]
}

SAM3 is prompted with each entity TYPE (e.g. "person"), NOT with the entity
attributes: the goal is to extract every instance of each concept present in the
image. By default only the top-level entity "type" values are used. With
--all-entities, every string stored under ANY "type" key in the structure JSON is
also segmented (e.g. attribute subtypes like clothing.shirt.type="button-up").
All unique types are segmented in a single batched forward pass.

A type is considered PRESENT when at least one instance is detected above the score
threshold. Entities whose type is not present are dropped from the output
(recorded under "absent_types"); entities of present types are kept with all of the
detected instances (surplus instances are kept too -- an over-count is itself a
signal). Masks are saved as PNG files under a masks/ directory and referenced
by path in the output JSON, together with score, bounding box and area, so a later
step can crop the original image on each mask and re-extract fine-grained attributes
with a vision-language model. By default the PNGs are binary (black/white) masks;
with --overlay each mask is instead painted in a distinct color on the original
image.

Output JSON (default <image_stem>.visual_entities.json):

{
  "image": "image_0.png",
  "structure_source": "prompt_1.structure.json",
  "scene": { ...unchanged... },
  "entities": [ ...input entities filtered to present types... ],
  "detections": [
    {
      "type": "person",
      "present": true,
      "instances": [
        {"mask": "image_0.masks/person_0.png", "score": 0.93,
         "box": [x1, y1, x2, y2], "area": 12345}
      ]
    }
  ],
  "absent_types": ["camera"]
}
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

DEFAULT_SAM_MODEL = "facebook/sam3"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MASK_THRESHOLD = 0.5
TYPES_PER_BATCH = 16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the visual entities actually present in an image using text-prompt "
            "segmentation (SAM3), given the possibly-present entities from a structure JSON."
        ),
    )
    parser.add_argument("image", type=Path, help="Path to the image to segment.")
    parser.add_argument("structure_json", type=Path, help="Path to the structure JSON (entities + scene).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to <image_stem>.visual_entities.json beside the image.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        help="Directory for the mask PNGs. Defaults to <image_stem>.masks beside the image.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip processing if the output JSON already exists (ignored if --force is set).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the output JSON to stdout instead of saving it to a file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SAM_MODEL,
        help=f"Name of the local segmentation model. Default: {DEFAULT_SAM_MODEL}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Instance score threshold (presence-adjusted). Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=DEFAULT_MASK_THRESHOLD,
        help=f"Mask binarization threshold. Default: {DEFAULT_MASK_THRESHOLD}",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Save each mask painted in a distinct color on the original image, instead of the binary mask PNG.",
    )
    parser.add_argument(
        "--all-entities",
        action="store_true",
        help=(
            "Also segment strings stored under any nested 'type' key in the structure JSON "
            "(e.g. attribute subtypes like clothing.shirt.type='button-up'), not only the "
            "top-level entity types."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto",
    )
    parser.add_argument(
        "--drun",
        action="store_true",
        help="Enable debug mode. All outputs will be stored to a debug folder (debug/).",
    )
    return parser


def default_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}.visual_entities.json")


def default_masks_dir(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}.masks")


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

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    return device, dtype


def load_sam_model(model_id: str, device: str, dtype: Any) -> tuple[Any, Any]:
    model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device)
    processor = Sam3Processor.from_pretrained(model_id)
    model.eval()
    return model, processor


def load_structure_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read structure JSON: {path} ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
        raise ValueError(f"structure JSON has no entities list: {path}")
    return data


def load_image(image_path: Path) -> Any:
    try:
        return Image.open(image_path).convert("RGB")
    except OSError as exc:
        raise ValueError(f"could not open input image: {image_path}") from exc


def segment_types(
    image: Any,
    types: list[str],
    model: Any,
    processor: Any,
    device: str,
    threshold: float,
    mask_threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    """Segment every unique type in one (chunked) batched forward pass.

    Returns {type: [instance, ...]} where each instance is
    {"score": float, "box": [x1, y1, x2, y2], "mask": torch.LongTensor (H, W)}.
    """
    detections: dict[str, list[dict[str, Any]]] = {t: [] for t in types}
    for start in range(0, len(types), TYPES_PER_BATCH):
        chunk = types[start : start + TYPES_PER_BATCH]
        inputs = processor(
            images=[image] * len(chunk),
            text=chunk,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        for type_name, result in zip(chunk, results):
            scores = result["scores"]
            boxes = result["boxes"]
            masks = result["masks"]
            for i in range(masks.shape[0]):
                box = [float(v) for v in boxes[i].tolist()]
                detections[type_name].append(
                    {
                        "score": float(scores[i]),
                        "box": box,
                        "mask": masks[i],
                    }
                )
    return detections


def _rainbow_color(index: int) -> tuple[int, int, int]:
    """A distinct RGB color per index, cycling through the rainbow (HSV hue)."""
    hue = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (round(r * 255), round(g * 255), round(b * 255))


def _overlay_mask(image: Any, mask_np: np.ndarray, color: tuple[int, int, int]) -> Any:
    """Blend a binary mask (0/255 uint8) onto the image in a given color."""
    base = image.convert("RGBA")
    mask_img = Image.fromarray(mask_np, mode="L")
    overlay = Image.new("RGBA", base.size, color + (0,))
    alpha = mask_img.point(lambda v: int(v * 0.5))
    overlay.putalpha(alpha)
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_masks(
    masks_dir: Path,
    detections: dict[str, list[dict[str, Any]]],
    image: Any | None = None,
    overlay: bool = False,
) -> None:
    """Write each detected mask as a PNG and replace the tensor with its path.

    Mask files are named <type>_<n>.png (n = instance index within the type) and the
    path stored in the instance dict is relative to masks_dir. By default a binary
    black/white mask is saved; with overlay=True each mask is painted in a distinct
    color on the original image instead.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(instances) for instances in detections.values())
    color_index = 0
    for type_name, instances in detections.items():
        for n, instance in enumerate(instances):
            mask = instance.pop("mask")
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            file_name = f"{type_name}_{n}.png"
            if overlay:
                if image is None:
                    raise ValueError("--overlay requires the original image")
                _overlay_mask(image, mask_np, _rainbow_color(color_index)).save(masks_dir / file_name)
                color_index += 1
            else:
                Image.fromarray(mask_np, mode="L").save(masks_dir / file_name)
            instance["area"] = int((mask_np > 0).sum())
            instance["mask"] = file_name


def collect_entity_types(structure: dict[str, Any]) -> list[str]:
    """Deduplicated, insertion-ordered top-level entity types for SAM3 prompting.

    Only each entity's own "type" is a segmentation concept. Attribute sub-values
    that happen to sit under a "type" key (e.g. clothing.shirt.type="button-up")
    must NOT become SAM3 prompts, so attributes are not traversed.
    """
    types: list[str] = []
    for entity in structure.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = entity.get("type")
        if not isinstance(entity_type, str):
            continue
        entity_type = entity_type.strip()
        if entity_type and entity_type not in types:
            types.append(entity_type)
    return types


def collect_all_types(structure: dict[str, Any]) -> list[str]:
    """Deduplicated, insertion-ordered string values of every "type" key in the JSON.

    Traverses dicts and lists recursively: top-level entity "type" fields (e.g.
    "person") AND nested "type" keys inside attributes (e.g.
    clothing.shirt.type="button-up") all become SAM3 segmentation prompts.
    """
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "type" and isinstance(value, str):
                    value = value.strip()
                    if value and value not in out:
                        out.append(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(structure)
    return out


def extract_visual_entities(
    image_path: Path,
    structure_path: Path,
    model: Any,
    processor: Any,
    device: str,
    threshold: float,
    mask_threshold: float,
    output_path: Path,
    masks_dir: Path,
    overlay: bool = False,
    all_entities: bool = False,
) -> dict[str, Any]:
    image = load_image(image_path)
    structure = load_structure_json(structure_path)

    entities = []
    for entity in structure.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = entity.get("type")
        if not isinstance(entity_type, str):
            continue
        entity_type = entity_type.strip()
        if not entity_type:
            continue
        entity["type"] = entity_type
        entities.append(entity)

    if all_entities:
        unique_types = sorted(collect_all_types(structure))
    else:
        unique_types = sorted(collect_entity_types(structure))
    detections = segment_types(
        image,
        unique_types,
        model,
        processor,
        device,
        threshold,
        mask_threshold,
    )

    present_types = {t for t, instances in detections.items() if instances}
    absent_types = sorted(set(unique_types) - present_types)
    present_entities = [e for e in entities if e["type"] in present_types]

    save_masks(masks_dir, detections, image=image, overlay=overlay)

    detections_out = []
    for type_name in sorted(present_types):
        instances = []
        for instance in detections[type_name]:
            mask_rel = str(
                Path(masks_dir / instance["mask"]).resolve().relative_to(output_path.parent.resolve())
            )
            instances.append(
                {
                    "mask": mask_rel,
                    "score": instance["score"],
                    "box": instance["box"],
                    "area": instance["area"],
                }
            )
        detections_out.append(
            {
                "type": type_name,
                "present": True,
                "instances": instances,
            }
        )

    return {
        "image": image_path.name,
        "structure_source": structure_path.name,
        "scene": structure.get("scene") if isinstance(structure.get("scene"), dict) else {},
        "entities": present_entities,
        "detections": detections_out,
        "absent_types": absent_types,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.print and (args.output is not None or args.force or args.skip):
        parser.error("--output, --force, and --skip cannot be used with --print")
    if not args.image.is_file():
        parser.error(f"input image is not a file: {args.image}")
    if not args.structure_json.is_file():
        parser.error(f"structure JSON is not a file: {args.structure_json}")

    output_path = args.output or default_output_path(args.image)
    masks_dir = args.masks_dir or default_masks_dir(args.image)

    if args.drun:
        debug_dir = Path("debug")
        output_path = debug_dir / output_path.parent.name / output_path.name
        masks_dir = debug_dir / output_path.parent.name / masks_dir.name
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(output_path.parent, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

    if not args.print:
        if not output_path.parent.exists():
            parser.error(f"output directory does not exist: {output_path.parent}")
        if output_path.exists():
            if args.force:
                pass  # Overwrite
            elif args.skip:
                print(f"Skipping existing file: {output_path}", file=sys.stderr)
                return
            else:
                parser.error(f"refusing to overwrite existing file: {output_path}. Use --force to overwrite.")

    device, dtype = resolve_device_and_dtype(args.device)
    print(f"Using device: {device}", file=sys.stderr)

    print(f"Loading segmentation model: {args.model}", file=sys.stderr)
    model, processor = load_sam_model(args.model, device, dtype)
    print(f"Segmentation model loaded: {args.model} on {device}", file=sys.stderr)

    result = extract_visual_entities(
        image_path=args.image,
        structure_path=args.structure_json,
        model=model,
        processor=processor,
        device=device,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        output_path=output_path,
        masks_dir=masks_dir,
        overlay=args.overlay,
        all_entities=args.all_entities,
    )

    if args.print:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved visual entities: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()