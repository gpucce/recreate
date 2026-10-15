import json
import pandas as pd

from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def crop_images_pool(
    imgs: list[Image.Image] | list[str],
    boxes: list[list[int]],
    padding: int = 0,
    max_workers: int | None = None,
) -> list[Image.Image]:
    if len(imgs) != len(boxes):
        raise ValueError("imgs and boxes must have the same length")

    groups = defaultdict(list)
    for idx, (src, box) in enumerate(zip(imgs, boxes)):
        groups[src].append((idx, box))

    def process(item):
        src, entries = item
        img = Image.open(src) if isinstance(src, str) else src
        try:
            return [
                (idx, crop_image(img, box, padding).convert("RGB"))
                for idx, box in entries
            ]
        finally:
            if isinstance(src, str):
                img.close()

    crops: list[Image.Image | None] = [None] * len(boxes)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # pool.map preserves group order and propagates exceptions
        for results in tqdm(
            pool.map(process, groups.items()),
            total=len(groups),
            desc="Cropping images",
        ):
            for idx, crop in results:
                crops[idx] = crop

    return crops


def crop_images(
    imgs: list[Image.Image] | list[str],
    boxes: list[list[int]],
    padding: int = 0,
) -> list[Image.Image]:
    if len(imgs) != len(boxes):
        raise ValueError("imgs and boxes must have the same length")

    # Group by source, remembering each box's original position
    groups = defaultdict(list)
    for idx, (src, box) in enumerate(zip(imgs, boxes)):
        groups[src].append((idx, box))

    crops: list[Image.Image | None] = [None] * len(boxes)

    for src, items in tqdm(groups.items(), desc="Cropping images"):
        # Open each file once; accept already-loaded PIL images too
        img = Image.open(src) if isinstance(src, str) else src
        try:
            for idx, box in items:
                # Crop first, convert after: only the small region gets converted
                crops[idx] = crop_image(img, box, padding).convert("RGB")
        finally:
            if isinstance(src, str):
                img.close()

    return crops


def crop_image(img: Image.Image, bbox: list[int], padding: int = 0) -> Image.Image:
    """Crop an image given a bounding box and optional padding."""
    x1, y1, x2, y2 = bbox
    width, height = img.size

    # Apply padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    return img.crop((x1, y1, x2, y2))


def parse_json_response(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            entities = parsed.get("entities")
            if isinstance(entities, list):
                cleaned_entities = []
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    cleaned = {
                        "type": str(entity.get("type", "")).strip(),
                        "attributes": entity.get("attributes")
                        if isinstance(entity.get("attributes"), dict)
                        else {},
                    }
                    entity_id = entity.get("id")
                    if entity_id is not None:
                        cleaned["id"] = str(entity_id)
                    entity_count = entity.get("count")
                    if isinstance(entity_count, int):
                        cleaned["count"] = entity_count
                    entity_present = entity.get("present")
                    if isinstance(entity_present, bool):
                        cleaned["present"] = entity_present
                    cleaned_entities.append(cleaned)
                result = {
                    "scene": parsed.get("scene") if isinstance(parsed.get("scene"), dict) else {},
                    "entities": cleaned_entities,
                    "parse_ok": True,
                }
                unmatched = parsed.get("unmatched_entities")
                if isinstance(unmatched, list):
                    cleaned_unmatched = []
                    for ue in unmatched:
                        if not isinstance(ue, dict):
                            continue
                        item = {
                            "type": str(ue.get("type", "")).strip(),
                            "attributes": ue.get("attributes")
                            if isinstance(ue.get("attributes"), dict)
                            else {},
                        }
                        ue_count = ue.get("count")
                        if isinstance(ue_count, int):
                            item["count"] = ue_count
                        cleaned_unmatched.append(item)
                    if cleaned_unmatched:
                        result["unmatched_entities"] = cleaned_unmatched
                return result
    return {"scene": {}, "entities": [], "parse_ok": False, "raw": raw.strip()}


def build_skeleton(anchor_entities: list[dict[str, Any]]) -> str:
    """Render the fixed identity skeleton (id/type/count) for the fill prompt."""
    skeleton = []
    for entity in anchor_entities:
        if not isinstance(entity, dict):
            continue
        item = {
            "id": entity.get("id"),
            "type": str(entity.get("type", "")).strip(),
        }
        if not item["type"]:
            continue
        if item["id"] is None:
            item["id"] = f"{item['type']}_{len(skeleton) + 1}"
        if isinstance(entity.get("count"), int):
            item["count"] = entity["count"]
        skeleton.append(item)
    return json.dumps(skeleton, ensure_ascii=False, indent=2)


def postprocess_responses(responses: list):
    import re
    regex = r"\*\*(.*?)\*\*"
    parsed = []
    for response in responses:
        spans = re.findall(regex, response)
        if spans:
            parsed.append({"parse_ok": True, "spans": spans})
        else:
            parsed.append({"parse_ok": False, "spans": []})
    return parsed