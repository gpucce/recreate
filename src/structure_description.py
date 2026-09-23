"""
Structure Description: this module takes as input a textual free-form description of an image (i.e., a text) and converts
it into a structured representation of the image (i.e., a set of entities and their features). Each entity has its own entry
in the JSON metadata.

For example, given the input text:

    "A person is standing next to a red car. The person is wearing a blue shirt and black pants."

The output JSON metadata would look like:
    
    {
        "entities": [
            {
                "type": "person",
                "attributes": {
                    "clothing": {
                        "shirt": "blue",
                        "pants": "black"
                    }
                }
            },
            {
                "type": "car",
                "attributes": {
                    "color": "red"
                }
            }
        ]
    }

The output is meant to be consumed by a segmentation model (e.g. SAM3) to track
the entities across all the N image generations of a loop and to check whether
entities and/or their attributes have changed, revealing hidden biases and
preferences of the image-generator model.

Output schema (also produced in the prompt given to the LLM):

    {
        "scene": {
            "background": "<free text>",
            "lighting": "<free text>",
            "camera_angle": "<free text>",
            "style": "<free text>",
            "mood": "<free text>",
            "color_palette": "<free text>"
        },
        "entities": [
            {
                "id": "person_1",
                "type": "person",
                "count": 1,
                "attributes": {
                    "<attribute>": "<value>",
                    "<attribute>": {"<subattribute>": "<value>"}
                }
            }
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_STRUCTURE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_NEW_TOKENS = 2048

STRUCTURE_PROMPT = """
Extract from the given image description a structured JSON object listing every entity
that appears in the image, together with each entity's attributes, plus the scene-level
attributes that describe the image as a whole.

Rules:
- Reply with ONLY a valid JSON object, no other text, no markdown fences.
- Each entity is one entry. If several instances of the same kind have DIFFERENT
  attributes (e.g. two people with different clothes), give each its own entry with a
  distinct "id" (e.g. "person_1", "person_2").
- If the text describes several IDENTICAL instances of the same kind with shared
  attributes, use a single entry and set "count" to the number of instances.
- "type" must be a generic noun usable as a text prompt for an image segmentation
  model such as SAM3 (e.g. "person", "car", "dog", "tree", "building", "coffee cup",
  "watch", "ring"). Do not use proper names, overly specific types, or catch-all
  words such as "object" or "item": name the actual thing instead.
- Preserve the exact attribute values given in the text: colors, materials, numbers,
  legible text or symbols, poses, expressions, clothing, and any other detail that
  describes how an entity looks. Do not invent attributes that are not in the text.
- "attributes" is an object mapping an attribute name to its value. A value is a
  string, a number, or a nested object for grouped attributes (e.g. "clothing" ->
  {{"shirt": "blue", "pants": "black"}}).
- "scene" holds global attributes of the image as a whole (background, lighting,
  camera angle, style, mood, color palette). Use the keys shown in the schema; put a
  short free-text value or omit a key when the description says nothing about it.

Schema:

{{
    "scene": {{
        "background": "<free text>",
        "lighting": "<free text>",
        "camera_angle": "<free text>",
        "style": "<free text>",
        "mood": "<free text>",
        "color_palette": "<free text>"
    }},
    "entities": [
        {{
            "id": "<type>_<n>",
            "type": "<generic noun>",
            "count": <int>,
            "attributes": {{
                "<attribute>": "<value>"
            }}
        }}
    ]
}}

Example:

Input text:
    "A person is standing next to a red car. The person is wearing a blue shirt and black pants."

Output JSON:
    {{
        "scene": {{}},
        "entities": [
            {{
                "id": "person_1",
                "type": "person",
                "count": 1,
                "attributes": {{
                    "clothing": {{
                        "shirt": "blue",
                        "pants": "black"
                    }},
                    "pose": "standing"
                }}
            }},
            {{
                "id": "car_1",
                "type": "car",
                "count": 1,
                "attributes": {{
                    "color": "red"
                }}
            }}
        ]
    }}


Example 2:

Input text:
    "The senior woman, likely a grandmother, has short, curly, platinum blonde hair and is wearing a light-colored, vertically striped, long-sleeved button-up shirt.
    The child, probably a boy around 3–5 years old, has short, wavy, blonde hair and is wearing a bright yellow short-sleeved T-shirt and light blue denim shorts."

Output JSON:
    {{
        "scene": {{}},
        "entities": [
            {{
                "id": "woman_1",
                "type": "person",
                "count": 1,
                "attributes": {{
                    "age": "senior",
                    "gender": "female",
                    "ethnicity": "asian",
                    "role": "grandmother",
                    "hair": {{
                        "length": "short",
                        "texture": "curly",
                        "color": "platinum blonde"
                    }},
                    "clothing": {{
                        "shirt": {{
                            "type": "button-up",
                            "sleeves": "long",
                            "pattern": "vertically striped",
                            "color": "light-colored"
                        }}
                    }}
                }}
            }},
            {{
                "id": "child_1",
                "type": "person",
                "count": 1,
                "attributes": {{
                    "age": "3–5 years old",
                    "gender": "male",
                    "ethnicity": "afro-american",
                    "role": "child",
                    "hair": {{
                        "length": "short",
                        "texture": "wavy",
                        "color": "blonde"
                    }},
                    "clothing": {{
                        "shirt": {{
                            "type": "T-shirt",
                            "sleeves": "short",
                            "color": "bright yellow"
                        }},
                        "shorts": {{
                            "type": "denim",
                            "color": "light blue"
                        }}
                    }}
                }}
            }}
        ]
    }}

Now extract the JSON from the following description:

--- DESCRIPTION ---
{description}
--- END OF DESCRIPTION ---
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a free-form textual image description into structured JSON metadata (entities + attributes).",
    )
    parser.add_argument(
        "input_prompt",
        nargs="?",
        type=Path,
        help="Path to the input description text file (required unless --stdin is used).",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the description text from standard input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSON path. Defaults to <input_stem>.structure.json beside the input, "
            "or structure.json for --stdin."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output JSON file.",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip processing if output JSON file already exists (ignored if --force is set).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the structured JSON to stdout instead of saving it to a file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_STRUCTURE_MODEL,
        help=f"Name of the local language model used for extraction. Default: {DEFAULT_STRUCTURE_MODEL}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Generation cap for the structured output. Default: {DEFAULT_MAX_NEW_TOKENS}",
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
        help="Enable debug mode. All results will be stored in the 'debug' folder."
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.structure.json")


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


def load_structure_model(model_id: str, device: str, dtype: Any) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def build_messages(description: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": STRUCTURE_PROMPT.format(description=description.strip()),
        }
    ]


def generate_structured_text(
    messages: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


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
                    cleaned_entities.append(cleaned)
                return {
                    "scene": parsed.get("scene") if isinstance(parsed.get("scene"), dict) else {},
                    "entities": cleaned_entities,
                    "parse_ok": True,
                }
    return {"scene": {}, "entities": [], "parse_ok": False, "raw": raw.strip()[:500]}


def extract_entities(
    description: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    messages = build_messages(description)
    raw = generate_structured_text(messages, model, tokenizer, max_new_tokens)
    structured = parse_json_response(raw)
    if not structured.get("parse_ok"):
        raise ValueError("language model returned output that could not be parsed as structured JSON")
    structured.pop("parse_ok", None)
    return structured


def read_description(path: Path) -> str:
    description = path.read_text(encoding="utf-8").strip()
    if not description:
        raise ValueError(f"input description file is empty: {path}")
    return description


def read_description_from_stdin() -> str:
    description = sys.stdin.read().strip()
    if not description:
        raise ValueError("stdin description is empty")
    return description


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.stdin and args.input_prompt is not None:
        parser.error("input_prompt cannot be used with --stdin")
    if not args.stdin and args.input_prompt is None:
        parser.error("input_prompt is required unless --stdin is used")
    if args.print and (args.output is not None or args.force or args.skip):
        parser.error("--output, --force, and --skip cannot be used with --print")

    if args.input_prompt is not None and not args.input_prompt.is_file():
        parser.error(f"input description is not a file: {args.input_prompt}")

    if args.stdin:
        description = read_description_from_stdin()
    else:
        description = read_description(args.input_prompt)

    if not args.print:
        output_path = args.output or (
            default_output_path(args.input_prompt) if args.input_prompt is not None else Path("structure.json")
        )

        if args.drun:
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            output_path = debug_dir / output_path.name

        if not output_path.parent.exists():
            parser.error(f"output directory does not exist: {output_path.parent}")
        if output_path.exists():
            if args.force:
                pass  # Overwrite
            elif args.skip:
                print(f"Skipping existing file: {output_path}")
                return
            else:
                parser.error(f"refusing to overwrite existing file: {output_path}. Use --force to overwrite.")

    device, dtype = resolve_device_and_dtype(args.device)
    print(f"Using device: {device}")

    print(f"Loading structure model: {args.model}")
    model, tokenizer = load_structure_model(args.model, device, dtype)
    print(f"Structure model loaded: {args.model} on {device}")

    structured = extract_entities(
        description,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    if args.print:
        print(json.dumps(structured, indent=2, ensure_ascii=False))
    else:
        output_path.write_text(
            json.dumps(structured, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved structured metadata: {output_path}")


if __name__ == "__main__":
    main()