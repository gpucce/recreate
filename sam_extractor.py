import json
import os

import pandas as pd

from src.sam_engine import process_dataset


def load_data(datadir: str, img_basedir: str, filter_entity: str | None = None) -> pd.DataFrame:
    """
    Load data from the specified directory and return a DataFrame containing image paths, steps, entities, and IDs.
    Each row in the DataFrame corresponds to a unique image, and contains a list of (possibly repeated) entities that 
    SAM3 has to localize.
    """
    data = [
        json.load(open(os.path.join(datadir, f), "r")) for f in os.listdir(datadir) if f.endswith(".json")
    ]

    dataset = []
    for d in data:
        for step_data in d:
            text_entities = step_data["llm_response"]["entities"]
            for ent in text_entities:
                if filter_entity is not None and ent["type"] != filter_entity:
                    continue
                dataset.append({
                    "folder": step_data["folder"],
                    "image_path": os.path.join(img_basedir, step_data["image"]),
                    "step": step_data["step"],
                    "entity": ent["type"],
                    "id": ent["id"],
                })
    df = pd.DataFrame(dataset).groupby("image_path", as_index=False, sort=False).agg(step=("step", "first"), entity=("entity", list), id=("id", list), folder=("folder", "first"))
    return df 


def main(args):
    assert args.debug or args.outdir is not None, "If debug is False, outdir must be specified."

    print(f"SAM3 processing:\ndataset from {args.datadir}\nwith images in {args.img_basedir}\noutdir: {args.outdir}\n")

    datadir = args.datadir
    img_basedir = args.img_basedir

    df = load_data(datadir, img_basedir, filter_entity=args.filter_entity)

    if args.debug:
        print(f"Debug mode: showing first 100 rows of the DataFrame")
        df = df[:50]

    boxes = process_dataset(
        df,
        model_name=args.model,
        device=args.device,
        threshold=args.threshold,
    )

    out_path = args.outdir
    if args.debug:
        out_path = "debug/visual_entities"
    os.makedirs(out_path, exist_ok=True)

    print(f"Saving detected boxes to {out_path}")

    grouped = boxes.groupby("folder")
    for img_path, group in grouped:
        out_fname = f"{img_path}.json"
        if args.filter_entity is not None:
            out_fname = f"{img_path}.{args.filter_entity}.json"
        with open(os.path.join(out_path, out_fname), "w") as f:
            group.to_json(f, orient="records", lines=True)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Extract SAM3 bounding boxes for the entities of each image.")
    parser.add_argument("--datadir", type=str, default="outputs/restructure/vlstereoset/qwen3827b_zimage",
                        help="Directory containing the restructured per-step JSON files.")
    parser.add_argument("--img-basedir", type=str, default="data/vlstereoset_runs",
                        help="Base directory of the images.")
    parser.add_argument("--model", type=str, default="facebook/sam3", help="Segmentation model.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto/cuda/cpu).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Instance score threshold.")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to show intermediate outputs.")
    parser.add_argument("--filter-entity", type=str, default=None, help="Filter by a specific entity type (optional).")
    args = parser.parse_args()
    main(args)