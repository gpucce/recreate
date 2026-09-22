#!/usr/bin/env python
"""Drive recreate/loop.py over a RANDOM sample of the local BBQ-V dataset.

BBQ-V is a HuggingFace `datasets` object on disk (`./BBQ-V`), a DatasetDict
with a single `test` split (14578 rows). Each row's photo is decoded from
column `file_name` to a PIL image. We draw a random sample of rows and feed
each one as the seed image of an independent content-convergence loop.

Selection (random by default, reproducible):
    The rows are shuffled with a seeded permutation (`--seed`, default 0) and
    then sliced by [--start-index, --start-index + --num-images). Sharding
    therefore still works exactly as before: shards that take disjoint slices
    get disjoint rows, PROVIDED every shard is given the same --seed (blank
    SEED in run_bbq_200.sh means "the default 0" -- still consistent).
    Pass --sequential for the old behaviour (rows in dataset order).

Identifying / recovering a source image:
    A run directory is named by its DATASET ROW INDEX, not by selection order:

        bbq_runs/image_<row>/        e.g. bbq_runs/image_09134/

    and each one carries a `source.json` with the row index, the dataset's own
    unique `id` (e.g. "01_01_0000_2_01"), the category, the question, and the
    seed that selected it. A per-shard JSONL manifest of the same records is
    written to `bbq_runs/manifest/`.

    To recover the original image of a run:

        import datasets, json
        rec = json.load(open("bbq_runs/image_09134/source.json"))
        ds  = datasets.load_from_disk(rec["dataset"])[rec["split"]]
        row = ds[rec["row"]]
        assert row["id"] == rec["id"]        # guards against a stale directory
        row["file_name"].save("recovered.png")

Layout per run:

    bbq_runs/image_09134/
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
    ./conda_venv/bin/python run_bbq_loop.py                 # 5 random images, 3 loop steps, GPU 0
    ./conda_venv/bin/python run_bbq_loop.py --num-images 5 --steps 3 --gpu 0
    ./conda_venv/bin/python run_bbq_loop.py --seed 1337     # a different random sample
    ./conda_venv/bin/python run_bbq_loop.py --sequential    # old behaviour: rows in order
    ./conda_venv/bin/python run_bbq_loop.py --start-index 100 --num-images 50 --gpu 2   # sharding
    ./conda_venv/bin/python run_bbq_loop.py --dry-run        # plan only, no GPU, no models
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


HERE = Path(__file__).resolve().parent
RECREATE_DIR = HERE / "recreate"
LOOP_PY = RECREATE_DIR / "loop.py"
BBQ_V_DIR = HERE / "BBQ-V"
IMAGE_COLUMN = "file_name"
ID_COLUMN = "id"
DEFAULT_OUT = HERE / "bbq_runs"
DEFAULT_SEED = 0                                               # shard-safe: all shards share it
DEFAULT_VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"            # code default (NOT cached locally)
CACHED_VISION_MODEL = "Qwen/Qwen3-VL-4B-Instruct"              # cached, fits with Z-Image on A40
DEFAULT_IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"               # cached


def load_bbq_v_split():
    """Return the (single) split's Dataset object, handling both Dataset and DatasetDict."""
    ds = datasets.load_from_disk(BBQ_V_DIR)
    if hasattr(ds, "keys"):  # DatasetDict
        split_name = list(ds.keys())[0]
        return ds[split_name], split_name
    return ds, "<single>"


def select_rows(n: int, start: int, count: int, seed: int, sequential: bool) -> List[int]:
    """Return the dataset row indices this (shard of the) run should process.

    Random mode permutes ALL n rows under `seed`, then takes the slice
    [start, start+count). Disjoint slices therefore stay disjoint rows, so the
    per-GPU sharding in run_bbq_200.sh keeps working unchanged.
    """
    start = max(0, start)
    if sequential:
        order = range(n)
    else:
        order = random.Random(seed).sample(range(n), n)
    return list(order[start:start + max(0, count)])


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


def save_seed(pil_image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = pil_image.convert("RGB")
    pil_image.save(path.as_posix(), format="PNG")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run recreate/loop.py over a random sample of BBQ-V images.")
    p.add_argument("--num-images", type=int, default=5, help="How many BBQ-V images (default 5).")
    p.add_argument("--start-index", type=int, default=0,
                   help="Offset into the SELECTION ORDER, for sharding across GPUs. Runs selection "
                        "positions [start-index, start-index+num-images). Default 0 (from the top).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Seed of the random row permutation (default {DEFAULT_SEED}). Shards must share "
                        "it to stay disjoint. Change it for a different random sample.")
    p.add_argument("--sequential", action="store_true",
                   help="Take rows in dataset order instead of randomly (the old behaviour).")
    p.add_argument("--steps", type=int, default=3,
                   help="Loop iterations per image. Each step is one describe-or-generate cycle (default 3).")
    p.add_argument("--gpu", type=int, default=0,
                   help="Pinned via CUDA_VISIBLE_DEVICES (default 0).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output root (default ./bbq_runs).")
    p.add_argument("--vision-model", default=DEFAULT_VISION_MODEL,
                   help=f"VLM id (code default: {DEFAULT_VISION_MODEL}; cached: {CACHED_VISION_MODEL}).")
    p.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL,
                   help=f"Diffusion id (default {DEFAULT_IMAGE_MODEL}).")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--fresh", action="store_true", default=True,
                   help="Delete existing run dirs before re-seeding (default on).")
    p.add_argument("--keep", dest="fresh", action="store_false",
                   help="Keep existing runs; only seed missing image_0.png.")
    p.add_argument("--max-consecutive-failures", type=int, default=3,
                   help="Abort once this many FAIL in A ROW (systemic problem, e.g. OOM). Isolated failures "
                        "are skipped and logged. Default 3. Set 1 to break on the first failure (old behavior).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without loading models or touching the GPU.")
    return p


def prepare_runs(args) -> Iterator[Tuple[int, Path, Dict[str, Any]]]:
    """Yield (row_index, work_dir, record) after seeding each source image in its own dir.

    Dir names use the DATASET ROW INDEX (5-digit: image_09134) so a run is
    traceable back to its source, shards never collide, and `ls` stays sorted.
    """
    split_ds, split_name = load_bbq_v_split()
    meta_ds = split_ds.remove_columns([IMAGE_COLUMN])   # lazy: metadata reads skip image decoding
    n = len(split_ds)
    rows = select_rows(n, args.start_index, args.num_images, args.seed, args.sequential)
    mode = "sequential" if args.sequential else f"random (seed {args.seed})"
    if not rows:
        print(f"note: --start-index {args.start_index} selects nothing (split has {n} rows, {mode}).",
              file=sys.stderr)
        return
    print(f"split: {split_name!r}  (row count: {n})  ->  {mode}, "
          f"selection positions {args.start_index}..{args.start_index + len(rows) - 1}  "
          f"({len(rows)} image(s))")

    for offset, row_idx in enumerate(rows):
        selection_pos = args.start_index + offset
        work_dir = args.out / f"image_{row_idx:05d}"
        meta = decode_metadata(split_ds, meta_ds[row_idx])
        image = split_ds[row_idx][IMAGE_COLUMN]
        size = list(image.size) if hasattr(image, "size") else None

        record: Dict[str, Any] = {
            "dir": work_dir.name,
            "row": row_idx,
            "selection_position": selection_pos,
            "selection": "sequential" if args.sequential else "random",
            "seed": args.seed,
            "dataset": BBQ_V_DIR.as_posix(),
            "split": split_name,
            "num_rows": n,
            "image_column": IMAGE_COLUMN,
            "source_size": size,
            **meta,
        }

        if not args.fresh and work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)
            if not (work_dir / "image_0.png").exists():
                save_seed(image, work_dir / "image_0.png")
        else:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            save_seed(image, work_dir / "image_0.png")

        (work_dir / "source.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        size_str = f"{size[0]}x{size[1]}" if size else "?"
        print(f"[row {row_idx:05d}] id={record.get(ID_COLUMN)} "
              f"category={record.get('category')} -> seeded {work_dir}/image_0.png  (source {size_str})")
        yield row_idx, work_dir, record


def write_manifest(args, records: List[Dict[str, Any]]) -> Path:
    """Write this shard's selection as JSONL. One file per shard so parallel GPU
    drivers never interleave writes; merge them with `cat bbq_runs/manifest/*.jsonl`."""
    manifest_dir = args.out / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"shard-{args.start_index:06d}-{len(records):04d}-gpu{args.gpu}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def run_loop(work_dir: Path, gpu: int, **kwargs) -> int:
    cmd = [
        sys.executable, str(LOOP_PY), str(work_dir.resolve()),
        "--start-from", "0",
        "--steps", str(kwargs["steps"]),
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
    if not LOOP_PY.is_file():
        print(f"error: loop.py not found at {LOOP_PY}", file=sys.stderr)
        return 2

    runs = list(prepare_runs(args))
    if runs:
        manifest = write_manifest(args, [record for _, _, record in runs])
        print(f"manifest: {manifest}")
    kwargs = dict(
        steps=args.steps,
        vision_model=args.vision_model,
        image_model=args.image_model,
        width=args.width,
        height=args.height,
    )
    if args.dry_run:
        for row_idx, work_dir, record in runs:
            print("\n" + "=" * 72)
            print(f"[row {row_idx:05d}] id={record.get(ID_COLUMN)}  {work_dir}")
            print(f"  CUDA_VISIBLE_DEVICES={args.gpu} {sys.executable} {LOOP_PY} "
                  f"{work_dir} --start-from 0 --steps {args.steps} --device cuda "
                  f"--vision-model {args.vision_model} --image-model {args.image_model} "
                  f" {args.width}x{args.height}")
        print("\ndry-run: no GPU, no models loaded.")
        return 0

    max_fail = max(1, args.max_consecutive_failures)
    failures = []            # (row_index, work_dir, rc)
    consecutive_fail = 0
    for row_idx, work_dir, record in runs:
        print("\n" + "=" * 72, flush=True)
        print(f"[row {row_idx:05d}] id={record.get(ID_COLUMN)} starting loop on {work_dir}", flush=True)
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
    done = len(runs) - len(failures)
    print(f"summary: {len(runs)} scheduled, {done} OK, {len(failures)} failed  (under {args.out})")
    if failures:
        print(f"FAILED rows: {[r for r, _, _ in failures]} did not complete cleanly.", file=sys.stderr)
        return 1
    print("OK: all loops completed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
