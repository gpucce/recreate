"""
This module contains the entity extraction engine leveraging SAM3.

The model receives an image and a list of entities that have to be extracted from the image.
The model returns a list of bounding boxes for each entity, along with a confidence score.
"""

import torch
import pandas as pd
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from tqdm.auto import tqdm

TYPES_PER_BATCH = 16


def resolve_device_and_dtype(requested_device):
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


def get_model(model_name="facebook/sam3", device="cuda", dtype=torch.bfloat16):
    model = Sam3Model.from_pretrained(model_name, dtype=dtype).to(device)
    processor = Sam3Processor.from_pretrained(model_name)
    model.eval()
    return model, processor


def process_dataset(df, model_name="facebook/sam3", device="auto",
                    threshold=0.5, mask_threshold=0.5):
    """Extract bounding boxes + confidence for each entity of each image.

    df must have `image_path`, `step` and `entity` (a list of entity types) columns.
    Returns a flat DataFrame with one row per detected instance:
        image_path, step, type, instance_index, box, score
    """
    device, dtype = resolve_device_and_dtype(device)
    model, processor = get_model(model_name, device, dtype)

    rows = []
    for rec in tqdm(df.to_dict("records"), desc="Processing images"):
        types = sorted({t for t in rec["entity"] if t})
        if not types:
            continue

        image = Image.open(rec["image_path"]).convert("RGB")

        for start in range(0, len(types), TYPES_PER_BATCH):
            chunk = types[start : start + TYPES_PER_BATCH]
            inputs = processor(images=[image] * len(chunk), text=chunk, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )

            for type_name, result in zip(chunk, results):
                for i, box in enumerate(result["boxes"].tolist()):
                    rows.append(
                        {
                            "folder": rec["folder"],
                            "image_path": rec["image_path"],
                            "step": rec["step"],
                            "type": type_name,
                            "instance_index": i,
                            "box": box,
                            "score": float(result["scores"][i]),
                        }
                    )

    return pd.DataFrame(rows)