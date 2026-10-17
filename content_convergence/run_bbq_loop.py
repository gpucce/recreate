#!/usr/bin/env python
"""Drive recreate/loop.py over a RANDOM sample of a local image dataset.

These datasets are supported, picked with --dataset:

    bbq-v        (default) data/BBQ-V, output under data/bbq_runs/
    vlstereoset  data/vlstereoset_clean, output under data/vlstereoset_runs/
    coco_dog     data/coco_dog, output under data/coco_dog_runs/ (images in data/coco/)
    coco_car     data/coco_car, output under data/coco_car_runs/ (images in data/coco/)

All paths are relative to the repo root; `data` is the data directory (a
symlink to the shared data volume).

vlstereoset_clean is also a DatasetDict with a single `test` split (1273
rows); its `image` column holds a path relative to data/
(e.g. "vlstereoset_imgs/0_img.png") rather than a decoded image, and it has
no `id` column, so the image file stem ("0_img") is used as the id and
`bias_type` as the category. Everything below describes BBQ-V; the
vlstereoset runs follow the same layout.

BBQ-V is a HuggingFace `datasets` object on disk (`data/BBQ-V`), a DatasetDict
with a single `test` split (14578 rows). Each row's photo is decoded from
column `file_name` to a PIL image. We draw a random sample of rows and feed
each one as the seed image of an independent content-convergence loop.

Selection (random by default, reproducible):
    The rows are shuffled with a seeded permutation (`--seed`, default 0) and
    then windowed to [--start-index, --start-index + --num-images); with
    --shard I --num-shards N a process takes every Nth position of that window
    starting at I (run_bbq_200.sh gives GPU I shard I). Sharding
    therefore still works exactly as before: shards that take disjoint slices
    get disjoint rows, PROVIDED every shard is given the same --seed (blank
    SEED in run_bbq_200.sh means "the default 0" -- still consistent).
    Pass --sequential for the old behaviour (rows in dataset order).

Identifying / recovering a source image:
    A run directory is named by its DATASET ROW INDEX, not by selection order:

        data/bbq_runs/image_<row>/        e.g. data/bbq_runs/image_09134/

    and each one carries a `source.json` with the row index, the dataset's own
    unique `id` (e.g. "01_01_0000_2_01"), the category, the question, and the
    seed that selected it. A per-shard JSONL manifest of the same records is
    written to `data/bbq_runs/manifest/`.

    To recover the original image of a run:

        import datasets, json
        rec = json.load(open("data/bbq_runs/image_09134/source.json"))
        ds  = datasets.load_from_disk(rec["dataset"])[rec["split"]]
        row = ds[rec["row"]]
        assert row["id"] == rec["id"]        # guards against a stale directory
        row["file_name"].save("recovered.png")

Layout per run:

    data/bbq_runs/image_09134/
        source.json      <- provenance (row index, id, category, seed, ...)
        image_0.png      <- seed (the BBQ-V image)
        prompt_1.txt     <- VLM description of image_0
        image_1.png      <- diffusion generation from prompt_1
        prompt_2.txt     <- VLM description of image_1
        ...

GPU pinning:
    loop.py accepts only --device cuda (no index). We pin a specific GPU by
    launching the subprocess with CUDA_VISIBLE_DEVICES=<gpu>.
    loop.py keeps VLM + diffusion model resident simultaneously, so use a GPU
    large enough to hold both (default: 0).

Usage:
    ./conda_venv/bin/python content_convergence/run_bbq_loop.py                 # 5 random images, 3 loop steps, GPU 0
    ./conda_venv/bin/python content_convergence/run_bbq_loop.py --num-images 5 --steps 3 --gpu 0
    ./conda_venv/bin/python content_convergence/run_bbq_loop.py --seed 1337     # a different random sample
    ./conda_venv/bin/python content_convergence/run_bbq_loop.py --sequential    # old behaviour: rows in order
    ./conda_venv/bin/python content_convergence/run_bbq_loop.py --start-index 100 --num-images 50 --gpu 2   # sharding
    ./conda_venv/bin/python content_convergence/run_bbq_loop.py --dry-run        # plan only, no GPU, no models
    ./conda_venv/bin/python content_convergence/run_bbq_loop.py --dataset vlstereoset   # VLStereoSet -> data/vlstereoset_runs/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import datasets
from PIL import Image


HERE = Path(__file__).resolve().parent
RECREATE_DIR = HERE.parent          # repo root: holds loop.py, conda_venv/, data/
DATA_DIR = RECREATE_DIR / "data"    # default --data-dir: datasets and run outputs (BBQ-V/, bbq_runs/, ...)
LOOP_PY = RECREATE_DIR / "loop.py"
# Per-dataset config; `dir` and `default_out` are relative to --data-dir.
# `image_column` holds either a decoded PIL image (BBQ-V)
# or a path relative to --data-dir (vlstereoset). `id_column` None means
# "use the image file stem as the id".
DATASETS: Dict[str, Dict[str, Any]] = {
    "bbq-v": dict(dir="BBQ-V", image_column="file_name", id_column="id",
                  category_column="category", default_out="bbq_runs"),
    "vlstereoset": dict(dir="vlstereoset_clean", image_column="image", id_column=None,
                        category_column="bias_type", default_out="vlstereoset_runs"),
    # COCO subsets, one row per annotated object (an image can appear twice), so the id is the
    # unique `id_sample`, not `image_id`; `image` is a path like "coco/train2017/000000456709.jpg".
    "coco_dog": dict(dir="coco_dog", image_column="image", id_column="id_sample",
                     category_column="category_id", default_out="coco_dog_runs"),
    "coco_car": dict(dir="coco_car", image_column="image", id_column="id_sample",
                     category_column="category_id", default_out="coco_car_runs"),
}
DEFAULT_DATASET = "bbq-v"
ID_KEY = "id"                                                  # key of the id in source.json
DEFAULT_SEED = 0                                               # shard-safe: all shards share it
DEFAULT_VISION_MODEL = "Qwen/Qwen3-VL-4B-Instruct"            # cached, fits with Z-Image on A40; same as the launchers
DEFAULT_IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"               # cached


def load_split(dataset_dir: Path):
    """Return the (single) split's Dataset object, handling both Dataset and DatasetDict."""
    ds = datasets.load_from_disk(dataset_dir)
    if hasattr(ds, "keys"):  # DatasetDict
        split_name = list(ds.keys())[0]
        return ds[split_name], split_name
    return ds, "<single>"


def select_rows(n: int, start: int, count: int, seed: int, sequential: bool,
                shard: int = 0, num_shards: int = 1) -> List[Tuple[int, int]]:
    """Return (selection_position, dataset_row) pairs this shard of the run should process.

    Random mode permutes ALL n rows under `seed` and takes the window
    [start, start+count); shard `shard` of `num_shards` then takes every
    num_shards-th position of it (shard, shard+num_shards, ...). Shards are
    disjoint as long as they share `seed`, and because they interleave, rows
    already run at the top of the order are spread evenly over the GPUs.
    """
    start = max(0, start)
    if sequential:
        order = list(range(n))
    else:
        order = random.Random(seed).sample(range(n), n)
    positions = range(start, min(n, start + max(0, count)))[shard::max(1, num_shards)]
    return [(pos, order[pos]) for pos in positions]


def decode_metadata(split_ds, row_values: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a metadata row into JSON-safe values (ClassLabel ints -> their names)."""
    out: Dict[str, Any] = {}
    for key, value in row_values.items():
        feature = split_ds.features.get(key)
        if isinstance(feature, datasets.ClassLabel) and isinstance(value, int):
            out[key] = feature.int2str(value)
        else:
            out[key] = value
    return out


def open_image(value, data_dir: Path):
    """Decoded PIL image as-is; a string is a path relative to data_dir."""
    if isinstance(value, (str, Path)):
        with Image.open(data_dir / value) as im:
            im.load()
            return im.copy()
    return value


def save_seed(pil_image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pil_image.mode in ("RGBA", "LA", "PA") or "transparency" in pil_image.info:
        # Flatten onto white. A plain convert("RGB") keeps e.g. a mode-"1" image's
        # int `transparency`, which the PNG writer rejects for RGB.
        rgba = pil_image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        pil_image = flat
    else:
        pil_image = pil_image.convert("RGB")
    pil_image.save(path.as_posix(), format="PNG")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run recreate/loop.py over a random sample of dataset images.")
    p.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET,
                   help=f"Source dataset (default {DEFAULT_DATASET}). vlstereoset loads "
                        "datasets.load_from_disk('vlstereoset_clean').")
    p.add_argument("--num-images", type=int, default=5,
                   help="How many images (default 5). Use --count-rows to get the dataset size.")
    p.add_argument("--count-rows", action="store_true",
                   help="Print the dataset's row count and exit (used by the launchers for NUM_IMAGES=all).")
    p.add_argument("--start-index", type=int, default=0,
                   help="Offset into the SELECTION ORDER, for sharding across GPUs. Runs selection "
                        "positions [start-index, start-index+num-images). Default 0 (from the top).")
    p.add_argument("--shard", type=int, default=0,
                   help="Which interleaved shard of that window to run: positions start-index+shard, "
                        "+num-shards, ... (default 0).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Number of interleaved shards (one per GPU in the launchers). Default 1 (whole window).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Seed of the random row permutation (default {DEFAULT_SEED}). Shards must share "
                        "it to stay disjoint. Change it for a different random sample.")
    p.add_argument("--sequential", action="store_true",
                   help="Take rows in dataset order instead of randomly (the old behaviour).")
    p.add_argument("--steps", type=int, default=3,
                   help="Loop iterations per image. Each step is one describe-or-generate cycle (default 3).")
    p.add_argument("--gpu", type=int, default=0,
                   help="Pinned via CUDA_VISIBLE_DEVICES (default 0).")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                   help="Folder holding the datasets (BBQ-V/, vlstereoset_clean/, vlstereoset_imgs/) and the "
                        "default outputs (default <repo>/data).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output root (default <data-dir>/bbq_runs for bbq-v, <data-dir>/vlstereoset_runs for "
                        "vlstereoset). A relative path is taken from the current directory.")
    p.add_argument("--vision-model", default=DEFAULT_VISION_MODEL,
                   help=f"VLM id (default {DEFAULT_VISION_MODEL}).")
    p.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL,
                   help=f"Diffusion id (default {DEFAULT_IMAGE_MODEL}).")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--fresh", action="store_true", default=True,
                   help="Delete existing run dirs before re-seeding (default on).")
    p.add_argument("--keep", dest="fresh", action="store_false",
                   help="Keep existing runs: seed only a missing image_0.png and resume a partial run "
                        "from its last finished step.")
    p.add_argument("--min-source-size", type=int, default=64,
                   help="Skip source images whose shorter side is below this many pixels (default 64).")
    p.add_argument("--redo-complete", action="store_true",
                   help="Also re-run dirs that already finished all --steps (default: skip them, "
                        "untouched, even under --fresh).")
    p.add_argument("--max-consecutive-failures", type=int, default=3,
                   help="Abort once this many FAIL in A ROW (systemic problem, e.g. OOM). Isolated failures "
                        "are skipped and logged. Default 3. Set 1 to break on the first failure (old behavior).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without loading models, touching the GPU or writing any file.")
    return p


def done_steps(work_dir: Path) -> int:
    """How many loop steps this dir already holds: the largest k with prompt_1..prompt_k present.

    With --start-from 0 the loop describes image_0 -> prompt_1, then renders
    image_k -> prompt_{k+1}, so step k ends when prompt_k.txt is written. A run
    stopped after k steps resumes with --start-from k --steps S-k: loop.py
    renders image_k from prompt_k (or describes image_k if it was already saved).
    """
    k = 0
    while (work_dir / f"prompt_{k + 1}.txt").is_file():
        k += 1
    return k


def run_is_complete(work_dir: Path, steps: int) -> bool:
    """True if loop.py already did all `steps` on this dir."""
    return steps > 0 and done_steps(work_dir) >= steps


def prepare_runs(args) -> Iterator[Tuple[int, Path, Dict[str, Any]]]:
    """Yield (row_index, work_dir, record) after seeding each source image in its own dir.

    Dir names use the DATASET ROW INDEX (5-digit: image_09134) so a run is
    traceable back to its source, shards never collide, and `ls` stays sorted.
    """
    cfg = DATASETS[args.dataset]
    image_column, id_column, category_column = cfg["image_column"], cfg["id_column"], cfg["category_column"]
    split_ds, split_name = load_split(args.data_dir / cfg["dir"])
    meta_ds = split_ds.remove_columns([image_column])   # lazy: metadata reads skip image decoding
    n = len(split_ds)
    picks = select_rows(n, args.start_index, args.num_images, args.seed, args.sequential,
                        args.shard, args.num_shards)
    mode = "sequential" if args.sequential else f"random (seed {args.seed})"
    if not picks:
        print(f"note: --start-index {args.start_index} selects nothing (split has {n} rows, {mode}).",
              file=sys.stderr)
        return
    print(f"split: {split_name!r}  (row count: {n})  ->  {mode}, "
          f"selection positions {args.start_index}..{args.start_index + args.num_images - 1}, "
          f"shard {args.shard}/{args.num_shards}  ({len(picks)} image(s))")

    for selection_pos, row_idx in picks:
        work_dir = args.out / f"image_{row_idx:05d}"
        meta = decode_metadata(split_ds, meta_ds[row_idx])
        raw_image = split_ds[row_idx][image_column]
        try:
            image = open_image(raw_image, args.data_dir)
        except Exception as exc:  # one unreadable source must not take down the whole shard
            print(f"[row {row_idx:05d}] SKIPPED: cannot open source image {raw_image!r}: {exc}",
                  file=sys.stderr, flush=True)
            continue
        size = list(image.size) if hasattr(image, "size") else None
        if size and min(size) < args.min_source_size:
            # e.g. the 1x1 placeholders some VLStereoSet downloads saved instead of a photo
            print(f"[row {row_idx:05d}] SKIPPED: source image {raw_image if isinstance(raw_image, str) else ''} "
                  f"is {size[0]}x{size[1]} (< --min-source-size {args.min_source_size})",
                  file=sys.stderr, flush=True)
            continue
        if id_column is None:
            meta[ID_KEY] = Path(raw_image).stem
        elif id_column != ID_KEY:
            meta[ID_KEY] = meta[id_column]
        if isinstance(raw_image, str):
            meta["source_image"] = raw_image

        record: Dict[str, Any] = {
            "dir": work_dir.name,
            "row": row_idx,
            "selection_position": selection_pos,
            "selection": "sequential" if args.sequential else "random",
            "seed": args.seed,
            "dataset_name": args.dataset,
            "dataset": (args.data_dir / cfg["dir"]).as_posix(),
            "split": split_name,
            "num_rows": n,
            "image_column": image_column,
            "source_size": size,
            **meta,
        }

        size_str = f"{size[0]}x{size[1]}" if size else "?"
        if not args.redo_complete and run_is_complete(work_dir, args.steps):
            # Finished run: leave it (and its source.json) exactly as it is.
            print(f"[row {row_idx:05d}] id={record.get(ID_KEY)} "
                  f"category={record.get(category_column)} -> {work_dir} already has "
                  f"prompt_{args.steps}.txt, skipping")
            yield row_idx, work_dir, record
            continue
        if args.dry_run:
            # Plan only: never wipe, seed or write into a run directory.
            if not work_dir.exists():
                action = "would seed"
            elif args.fresh:
                action = "would wipe+reseed"
            else:
                action = f"would resume from step {done_steps(work_dir)}/{args.steps} in"
            print(f"[row {row_idx:05d}] id={record.get(ID_KEY)} "
                  f"category={record.get(category_column)} -> {action} {work_dir}/image_0.png  (source {size_str})")
            yield row_idx, work_dir, record
            continue

        try:
            if not args.fresh and work_dir.exists():
                work_dir.mkdir(parents=True, exist_ok=True)
                if not (work_dir / "image_0.png").exists():
                    save_seed(image, work_dir / "image_0.png")
            else:
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                save_seed(image, work_dir / "image_0.png")
        except Exception as exc:
            # A failed save can leave a truncated image_0.png that --keep would trust; drop it.
            (work_dir / "image_0.png").unlink(missing_ok=True)
            print(f"[row {row_idx:05d}] SKIPPED: cannot seed {work_dir}/image_0.png: {exc}",
                  file=sys.stderr, flush=True)
            continue

        (work_dir / "source.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"[row {row_idx:05d}] id={record.get(ID_KEY)} "
              f"category={record.get(category_column)} -> seeded {work_dir}/image_0.png  (source {size_str})")
        yield row_idx, work_dir, record


def write_manifest(args, records: List[Dict[str, Any]]) -> Path:
    """Write this shard's selection as JSONL. One file per shard so parallel GPU
    drivers never interleave writes; merge them with `cat data/bbq_runs/manifest/*.jsonl`."""
    manifest_dir = args.out / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"shard-{args.start_index:06d}-{args.shard}of{args.num_shards}-{len(records):04d}-gpu{args.gpu}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def run_loop(work_dir: Path, gpu: int, **kwargs) -> int:
    # A --fresh dir was just wiped, so this is 0 there; a --keep dir resumes after its last step.
    start = min(done_steps(work_dir), kwargs["steps"])
    cmd = [
        sys.executable, str(LOOP_PY), str(work_dir.resolve()),
        "--start-from", str(start),
        "--steps", str(kwargs["steps"] - start),
        "--device", "cuda",
        "--vision-model", kwargs["vision_model"],
        "--image-model", kwargs["image_model"],
        "--width", str(kwargs["width"]),
        "--height", str(kwargs["height"]),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return subprocess.run(cmd, cwd=str(RECREATE_DIR), env=env).returncode


def main() -> int:
    args = build_parser().parse_args()
    if args.count_rows:
        print(len(load_split(args.data_dir / DATASETS[args.dataset]["dir"])[0]))
        return 0
    if args.out is None:
        args.out = args.data_dir / DATASETS[args.dataset]["default_out"]
    if not LOOP_PY.is_file():
        print(f"error: loop.py not found at {LOOP_PY}", file=sys.stderr)
        return 2

    runs = list(prepare_runs(args))
    if runs and not args.dry_run:
        manifest = write_manifest(args, [record for _, _, record in runs])
        print(f"manifest: {manifest}")
    kwargs = dict(
        steps=args.steps,
        vision_model=args.vision_model,
        image_model=args.image_model,
        width=args.width,
        height=args.height,
    )
    skipped = [] if args.redo_complete else [row for row, work_dir, _ in runs
                                             if run_is_complete(work_dir, args.steps)]
    if args.dry_run:
        for row_idx, work_dir, record in runs:
            if row_idx in skipped:
                continue
            print("\n" + "=" * 72)
            print(f"[row {row_idx:05d}] id={record.get(ID_KEY)}  {work_dir}")
            start = 0 if args.fresh else min(done_steps(work_dir), args.steps)
            print(f"  CUDA_VISIBLE_DEVICES={args.gpu} {sys.executable} {LOOP_PY} "
                  f"{work_dir} --start-from {start} --steps {args.steps - start} --device cuda "
                  f"--vision-model {args.vision_model} --image-model {args.image_model} "
                  f" {args.width}x{args.height}")
        print("\ndry-run: no GPU, no models loaded.")
        return 0

    max_fail = max(1, args.max_consecutive_failures)
    failures = []            # (row_index, work_dir, rc)
    consecutive_fail = 0
    for row_idx, work_dir, record in runs:
        if row_idx in skipped:
            continue
        print("\n" + "=" * 72, flush=True)
        print(f"[row {row_idx:05d}] id={record.get(ID_KEY)} starting loop on {work_dir}", flush=True)
        print("=" * 72, flush=True)
        rc = run_loop(work_dir, args.gpu, **kwargs)
        if rc != 0:
            failures.append((row_idx, work_dir, rc))
            consecutive_fail += 1
            print(f"[row {row_idx:05d}] loop exited rc={rc} "
                  f"({consecutive_fail}/{max_fail} consecutive failure(s))", file=sys.stderr, flush=True)
            if consecutive_fail >= max_fail:
                print("aborting: too many consecutive failures (likely a systemic problem).", file=sys.stderr)
                break
        else:
            consecutive_fail = 0

    print("\n" + "-" * 72, flush=True)
    done = len(runs) - len(skipped) - len(failures)
    print(f"summary: {len(runs)} scheduled, {len(skipped)} already complete (skipped), "
          f"{done} OK, {len(failures)} failed  (under {args.out})")
    if failures:
        print(f"FAILED rows: {[r for r, _, _ in failures]} did not complete cleanly.", file=sys.stderr)
        return 1
    print("OK: all loops completed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
