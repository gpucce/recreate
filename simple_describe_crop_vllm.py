import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.crop_describe import (
    crop_entity,
    image_to_data_uri,
    load_vllm_model,
)

PROMPT = """
You are shown a crop of a {type} taken from an image. Describe this {type} in fine
detail.

Rules:
- Fill in the following template already encoding the most target features that need to be described:
{template}
- Reply with ONLY a valid JSON object, no other text, no markdown fences:
""".strip()


def build_prompt(target_entity, template_dir="data/templates"):
    template_path = os.path.join(template_dir, f"{target_entity}.json")
    with open(template_path, "r") as f:
        template = json.load(f)
    return PROMPT.format(type=target_entity, template=json.dumps(template))


def build_message(crop, prompt):
    """OpenAI-style chat message with the crop as an inline image (vLLM-compatible)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_uri(crop)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def parse_json_object(raw):
    """Generic extraction of the filled-template JSON object from the reply."""
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            parsed["parse_ok"] = True
            return parsed
    return {"parse_ok": False}


def collect_tasks(args):
    """Return a flat list of {sample, step, entity_index, box, message}."""
    dataset_base_dir = Path(args.visual_root) / args.dataset
    dataset_dirs = sorted(os.listdir(dataset_base_dir))[: args.max_data]
    tasks = []
    for sample in dataset_dirs:
        for step in args.steps:
            visual_path = dataset_base_dir / sample / f"image_{step}.visual_entities.json"
            if not visual_path.is_file():
                continue
            img_path = (
                Path(args.runs_parent) / f"{args.dataset}_runs" / sample / f"image_{step}.png"
            )
            if not img_path.is_file():
                continue

            with open(visual_path, "r") as f:
                data = json.load(f)
            whole_img = Image.open(img_path).convert("RGB")

            dets = [d for d in data.get("detections", []) if d.get("type") == args.target_entity]
            if not dets:
                continue
            instances = [
                e for e in dets[0].get("instances", [])
                if (e.get("area") or 0) >= args.min_area
            ]
            if not instances:
                continue

            prompt = build_prompt(args.target_entity, args.template_dir)
            for i, ent in enumerate(instances):
                crop = crop_entity(whole_img, ent["box"], padding=args.padding)
                tasks.append(
                    {
                        "sample": sample,
                        "step": step,
                        "entity_index": i,
                        "box": ent["box"],
                        "message": build_message(crop, prompt),
                    }
                )
    return tasks


def main(args):
    tasks = collect_tasks(args)
    print(f"Collected {len(tasks)} crop task(s)")

    if not tasks:
        print("Nothing to do.")
        return

    from vllm import SamplingParams

    llm = load_vllm_model(
        args.vlm_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)

    results = []
    pbar = tqdm(range(0, len(tasks), args.batch_size), desc="Describing crops")
    for start in pbar:
        chunk = tasks[start : start + args.batch_size]
        messages = [t["message"] for t in chunk]
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        for task, out in zip(chunk, outputs):
            raw = out.outputs[0].text
            results.append(
                {
                    "step": task["step"],
                    "entity_index": task["entity_index"],
                    "entity_type": args.target_entity,
                    "box": task["box"],
                    "response": raw,
                    "parsed": parse_json_object(raw),
                }
            )

    by_sample = defaultdict(list)
    for task, result in zip(tasks, results):
        by_sample[task["sample"]].append(result)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for sample in sorted(by_sample):
        out_path = outdir / f"{args.dataset}.{sample}.json"
        with open(out_path, "w") as f:
            json.dump(by_sample[sample], f, indent=2, ensure_ascii=False)
        print(f"Saved {out_path} ({len(by_sample[sample])} records)")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="vLLM batched fine-grained crop description (template fill).")
    parser.add_argument("--dataset", type=str, default="vlstereoset", help="Dataset name (default: vlstereoset).")
    parser.add_argument("--visual_root", type=str, default="data/anchored/visual_entities",
                        help="Root holding <dataset>/<sample>/image_N.visual_entities.json")
    parser.add_argument("--runs-parent", type=str, default="data",
                        help="Parent of the <dataset>_runs image directories.")
    parser.add_argument("--max-data", type=int, default=10,
                        help="Number of sample folders to process (default: 10).")
    parser.add_argument("--steps", type=int, nargs="+", default=list(range(1, 16)),
                        help="Steps to process (default: 1..15).")
    parser.add_argument("--target-entity", type=str, default="person",
                        help="Entity type to describe (default: person).")
    parser.add_argument("--min-area", type=int, default=64 * 64,
                        help="Skip instances with a bbox area below this (default: 4096).")
    parser.add_argument("--template-dir", type=str, default="data/templates",
                        help="Directory with <entity>.json templates (default: data/templates).")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="Crop padding as a fraction of the box size.")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3.8-27B",
                        help="Vision-language model (default: Qwen/Qwen3.8-27B).")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="vLLM tensor parallel size (the 27B model needs 2 GPUs).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM GPU memory utilization (0..1).")
    parser.add_argument("--batch-size", type=int, default=32, help="vLLM batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation cap per crop.")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="vLLM max concurrent sequences.")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM max model context length.")
    parser.add_argument("--outdir", type=str, default="debug/descs",
                        help="Output directory (default: debug/descs).")
    args = parser.parse_args()
    main(args)