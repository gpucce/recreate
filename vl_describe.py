import os
import io
import json

from PIL import Image
from pathlib import Path
from pprint import pprint as pp
from tqdm.auto import tqdm

import pandas as pd

from src.vl_engine import generate
from src.utils import crop_images_pool, build_skeleton, parse_json_response as parse_restructure, postprocess_responses as parse_classify
# from src.structure_description import parse_json_response as parse_restructure, build_skeleton


def _load_dataset_folder(path: str):
    import os
    dirs = os.listdir(path)
    dirs = sorted([Path(path, d) for d in dirs if Path(path, d).is_dir()])
    return dirs


def prepare_dataset_from_folder(dirs: list[Path]):
    target_fname_suffix = ".visual_entities.json"

    dataset = []
    for d in tqdm(dirs, desc="Processing directories"):
        files = sorted([f for f in d.iterdir() if f.name.endswith(target_fname_suffix)])
        for f in files:
            with open(f, "r") as fin:
                data = json.load(fin)
                detected = data["detections"]

            for elem in detected:
                instances = elem["instances"]
                for i, ent in enumerate(instances):
                    step = f.name.split(".")[0].split("_")[-1]
                    dataset.append(
                        {
                            "step": int(step),
                            "type": elem["type"],
                            "bbox": ent["box"],
                            "score": ent["score"],
                            "area": ent["area"],
                            "detected_idx": i,
                            "folder": d.name,
                            "image": f"{d.name}/image_{step}.png"   # TODO: check if step is pointing to the correct image
                        }
                    )
    print(f"--- Found {len(dataset)} entities in {len(dirs)} samples\n")
    return pd.DataFrame(dataset)


def prepare_dataset_from_folder_restructure(dirs: list[Path]):
    target_fname_affix = "prompt_"

    dataset = []
    for d in tqdm(dirs, desc="Processing directories"):
        files = sorted([f for f in d.iterdir() if target_fname_affix in f.name])
        for f in files:
            step = f.name.split(".")[0].split("_")[-1]
            with open(f, "r") as fin:
                data = fin.read()
            dataset.append(
                {
                    "step": int(step),
                    "image": f"{d.name}/image_{int(step) - 1}.png",
                    "folder": d.name,
                    "prompt": data
                }
            )
    print(f"--- Found {len(dataset)} prompts in {len(dirs)} samples\n")
    return pd.DataFrame(dataset)


def load_data(path: str, task: str, debug: bool = False):
    n_debug = 3
    ext = Path(path).suffix

    if ext == "":
        dirs = _load_dataset_folder(path)

        if debug:
            print(f"[Running in DEBUG MODE] Limiting to first n messages for testing...")
            dirs = dirs[:n_debug]

        if task == "crop":
            df = prepare_dataset_from_folder(dirs)
        elif task == "restructure":
            df = prepare_dataset_from_folder_restructure(dirs)
        return df

    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if debug:
        print(f"[Running in DEBUG MODE] Limiting to first n messages for testing...")
        df = df.head(n_debug)

    return df


def load_imgs(imgs, dict_key="bytes", img_basedir: str | None = None) -> list[Image.Image] | list[str]:
    print("Loading images...")
    if type(imgs[0]) == dict:
        imgs = [img[dict_key] for img in imgs]

    if img_basedir is not None:
        imgs = [str(Path(img_basedir) / img) for img in imgs]

    print(f"--- Found {len(imgs)} images\n")

    if type(imgs[0]) == str:
        return imgs
    elif type(imgs[0]) == bytes:
        imgs = [Image.open(io.BytesIO(img)).convert("RGB") for img in imgs]

    print(f"--- Loaded {len(imgs)} images\n")

    return imgs


def get_prompt(task: str, prompt_path: str = "src/prompts.yaml", anchored: bool = False):
    import yaml
    with open(prompt_path, "r") as f:
        prompts = yaml.safe_load(f)
    if anchored:
        task = f"{task}_anchored"
    prompt = prompts.get(task, None)
    if prompt is None:
        raise ValueError(f"Prompt for task '{task}' not found in {prompt_path}")
    return prompt


def _build_multimodal_message(img: Image.Image, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }
    ]


def _build_text_message(prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]


def build_messages(task: str, prompts: list | str, imgs: list[Image.Image] | None = None, formatter: list[dict] | None = None):
    print(f"Building messages...")
    if type(prompts) == str:
        prompts = [prompts]

    if task in ["restructure", "describe_crop"] and formatter is not None:
        if len(prompts) == 1 and len(formatter) > 1:
            prompts = prompts * len(formatter)
            prompts = [p.format(**f) for p, f in zip(prompts, formatter)]

    if imgs is not None:
        if len(prompts) == 1 and  len(imgs) > 1:
            prompts = prompts * len(imgs)

    formatted = []
    if imgs is None:
        for prompt in prompts:
            formatted.append(_build_text_message(prompt))
    else:
        for prompt, img in zip(prompts, imgs):
            formatted.append(_build_multimodal_message(img, prompt))
    return formatted


def _task_dict(task):
    if task == "classify":
        return "look_for_entity"
    elif task == "crop":
        return "describe_crop"
    elif task == "restructure":
        return "restructure"


def main(args):
    _mm_gen = True if args.task in ["classify", "crop"] else False
    _task = _task_dict(args.task)

    df = load_data(args.datapath, task=args.task, debug=args.debug)

    if _mm_gen:
        imgs = load_imgs(df.image, img_basedir=args.img_basedir)
        if args.task == "crop":
            print("Cropping images...")
            imgs = crop_images_pool(imgs, df.bbox.tolist(), padding=0, max_workers=8)
    else:
        imgs = None

    # prepare prompt formatter - if anchored, also run the first step to get the skeletons for the anchor step, then use them for the successive steps
    if args.task == "restructure":
        # if anchored, also run the first step to get the skeletons for the anchor step, then use them for the successive steps

        if args.anchored:
            anchor_step = 1
            df_anchor = df[df["step"] == anchor_step].copy()
            df_anchored = df[df["step"] != anchor_step].copy()
            prompt_formatter_anchor = [{
                "description": row.prompt
            } for _, row in df_anchor.iterrows()]

            # TODO: running `generate` only on the anchor step here - we need the skeletons for successive iterations
            prompt = get_prompt(_task)
            messages = build_messages(task=_task, prompts=prompt, imgs=imgs, formatter=prompt_formatter_anchor)

            outputs, llm = generate(
                model_name=args.model_name,
                backend=args.backend,
                messages=messages,
                batch_size=args.batch_size,
                max_new_tokens=args.max_tokens,
            )

            outputs = [parse_restructure(s) for s in outputs]
            _skeletons = [json.loads(build_skeleton(s["entities"])) for s in outputs]

            df_anchor["llm_response"] = outputs
            df_anchor["skeleton"] = _skeletons

            img_groups = df_anchored.groupby("folder")

            prompt_formatter_anchored = []
            for folder, group in img_groups:
                anchor_skeleton = df_anchor[df_anchor["folder"] == folder]["skeleton"].values[0]
                image_prompt_formatter_anchored = [{
                    "skeleton": anchor_skeleton,
                    "description": row.prompt
                } for _, row in group.iterrows()]
                prompt_formatter_anchored.extend(image_prompt_formatter_anchored)

            # building messages for the anchored generations (only)
            prompt = get_prompt(_task, anchored=True)
            prompt_formatter = prompt_formatter_anchored

        else:
            prompt_formatter = [{
                "description": row.prompt
            } for _, row in df.iterrows()]

    elif args.task == "classify":
        prompt_formatter = None

    elif args.task == "crop":
        TARGET_ENTITY = "person" # DEBUG: only using person for now, to avoid too many prompts
        template_dir = "data/templates"
        prompt_formatter =[{
            "type": row.type,
            "template": json.dumps(
                json.load(open(os.path.join(template_dir, f"{row.type}.json"))))
        } for i, row in df.iterrows() if row.type == TARGET_ENTITY]  # DEBUG: only using person for now, to avoid too many prompts

        # filtering images for the target entity (get the indices of the rows that match the target, keep those indices from the image list)
        valid_indices = [i for i, row in df.iterrows() if row.type == TARGET_ENTITY]
        imgs = [imgs[i] for i in valid_indices]
        print(f"--- Filtered {len(imgs)} images for target entity '{TARGET_ENTITY}'\n")
    else:
        raise NotImplementedError(f"Task '{args.task}' is not implemented yet")

    prompt = get_prompt(_task)
    messages = build_messages(task=_task, prompts=prompt, imgs=imgs, formatter=prompt_formatter)

    outputs, _ = generate(
        model_name=args.model_name,
        backend=args.backend,
        messages=messages,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        loaded_llm=llm if args.anchored else None,
        )

    # setting outdir
    outdir = f"{args.outdir}"
    if args.debug:
        outdir = f"debug/new_out"
    os.makedirs(outdir, exist_ok=True)

    if args.anchored:
        # process responses when we have a anchored mode (i.e., we have a df_anchor and df_anchored)
        outputs = [parse_restructure(s) for s in outputs]
        df_anchored["llm_response"] = outputs

        df_response = pd.concat([df_anchor, df_anchored], ignore_index=True)
        df_response.sort_values(by=["folder", "step"], inplace=True)
        df_response.reset_index(drop=True, inplace=True)

    if _task == "classify":
        outfile = os.path.join(outdir, f"{args.datapath.split('/')[-1]}.entities.json")
        parsed = parse_classify(outputs)

        for i in range(len(parsed)):
            _id = df.iloc[i]["ID"]
            parsed[i]["id"] = _id
            parsed[i]["llm_response"] = outputs[i]

        with open(outfile, "w") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

    elif _task == "restructure":
        grouped = df_response.groupby("folder")
        # storing results by image_folder, with each folder containing a scene.anchored.json file with the results for all steps
        for folder, group in grouped:
            outdir_file = os.path.join(outdir, folder)
            print(f"Saving outputs for folder '{folder}' to {outdir_file}...")
            os.makedirs(outdir_file, exist_ok=True)
            group.to_json(os.path.join(outdir_file, f"scene.anchored.json"), orient="records", indent=2, force_ascii=False)

    elif _task == "describe_crop":
        pp(outputs)
        # TODO: implement saving outputs for describe_crop task
    else:
        raise NotImplementedError(f"Postprocessing for task '{_task}' is not implemented yet")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--task",           type=str, default="restructure", choices=["classify", "crop", "restructure"], help="Task to perform: classify or crop")
    parser.add_argument("--model-name",     type=str, default="Qwen/Qwen3.5-4B", help="Model name to use for generation")
    parser.add_argument("--backend",        type=str, default="hf", choices=["hf", "vllm"], help="Backend to use for generation")
    parser.add_argument("--datapath",       type=str, default="data/vlstereoset_runs", help="Path to the input data file")
    parser.add_argument("--img-basedir",    type=str, default=None, help="Path to the base directory containing the images")
    parser.add_argument("--batch-size",     type=int, default=2, help="Batch size for generation")
    parser.add_argument("--max-tokens",     type=int, default=32, help="Maximum number of tokens to generate")
    parser.add_argument("--anchored",       action="store_true", help="Run in anchored mode (only for 'restructure' task)")
    parser.add_argument("--outdir",         type=str, default="data/new_out", help="Output directory to save results")
    parser.add_argument("--debug",          action="store_true", help="Run in debug mode (limit number of messages for testing)")
    args = parser.parse_args()

    main(args)