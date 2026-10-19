#!/usr/bin/env python
"""Entity-wise visual similarity: embed every entity's box crop with EmbeddingGemma 2 and compare
crops of the SAME entity type, both inside one image and across the steps of a run.

Each entity record carries a step, a type and a pixel box (x0, y0, x1, y1) on image_<step>.png of
its run. The crop is embedded with google/embeddinggemma-2 (vision encoder only, audio is not
loaded) into an L2-normalised 768-d vector, and similarities are cosines between those vectors.

Pairs (only entities of the same type are ever compared):
    within        every pair of same-type entities in the same image_N (person 0 vs person 1):
                  do the people of one scene look alike, and does that change over the steps?
    consecutive   every same-type pair between step N and step N+1 of a run
    anchor        every same-type pair between a fixed step K and every other step
                  (--anchor K; default the first step of each run; --no-anchor turns it off)
    For across-step pairs, best_match.csv also keeps, for each entity of the later step, its most
    similar same-type entity of the earlier step: "is there still something like it?".

Input: entity files as aggregate_scene_description_attributes.py finds them, per experiment:
    per-run files   [<dataset>.]image_<row>[.<experiment>].json  (JSON list or JSON Lines)
        data/outputs/entities_descriptions/vlstereoset/qwen3vl4b_flux2klein/vlstereoset_exp_qwen3vl4b_flux2klein/vlstereoset.image_00002.json
    per-step dirs   image_<row>/image_<step>.<experiment>.json  ({"entities": [...]})
and --images, the run root holding image_<row>/image_<step>.png:
        data/vlstereoset_experiments/qwen3vl4b_flux2klein/20261012-155508

Outputs (under <out>/<experiment>/, --out defaulting to <input>/../entity_similarity/<input name>):
    within_pairs.csv     run, step, type, a, b, similarity
    across_pairs.csv     run, relation, step_a, step_b, type, a, b, similarity
    best_match.csv       run, relation, step_a, step_b, type, entity (at step_b), best (at step_a), similarity
    summary.csv          per (kind, step, type): pairs, mean, std; across kinds also mean best match.
                         For across kinds "step" is the moving side (step_b, or the non-anchor step).
    embeddings/<run>.npz cache of the crop embeddings (reused unless --recompute or the boxes changed)
    crops/<run>/         with --save-crops: the PNG crops that were embedded, to check the boxes

Like the CLIP scores in semantic_validate.py, unrelated crops of the same type still score well
above 0, so read the curves relative to each other, not as absolute values.

Usage:
    ./conda_venv/bin/python content_convergence/similarity_entitiy_wise.py \\
        data/outputs/entities_descriptions/vlstereoset/qwen3vl4b_flux2klein/vlstereoset_exp_qwen3vl4b_flux2klein \\
        --images data/vlstereoset_experiments/qwen3vl4b_flux2klein/20261012-155508 --gpu 0
    ... --dry-run                 count runs, entities and missing images; load no model
    ... --type person --anchor 1  only persons, anchor at step 1
    ... --save-crops              also write the crops for inspection
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_MODEL = "google/embeddinggemma-2"
DEFAULT_BATCH = 32
DEFAULT_MIN_BOX = 8          # px; smaller crops are skipped
ROW_RE = re.compile(r"image_(\d+)")


def _pin_gpu_before_torch() -> None:
    """Honour --gpu by setting CUDA_VISIBLE_DEVICES before torch is imported."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu", type=int, default=None)
    known, _ = pre.parse_known_args()
    if known.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_pin_gpu_before_torch()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_scene_description_attributes import (  # noqa: E402
    STEP_FILE_RE, find_runs, read_records)
from compare_scene_description_entities import normalize_type  # noqa: E402


# ---- loading --------------------------------------------------------------
Entity = Dict  # {"step", "type", "index", "box"}


def run_name(run: Path) -> str:
    """image_<row> for both layouts (vlstereoset.image_00002.json -> image_00002)."""
    m = ROW_RE.search(run.name)
    if not m:
        raise ValueError(f"cannot read the dataset row from {run}")
    return f"image_{m.group(1)}"


def _entity(rec: Dict, step: int, position: int, normalize: bool) -> Optional[Entity]:
    box = rec.get("box")
    etype = rec.get("entity_type") or rec.get("type")
    if not box or len(box) != 4 or not etype:
        return None
    index = next((rec[k] for k in ("entity_index", "instance_index", "index") if k in rec), position)
    return {"step": step, "type": normalize_type(etype, normalize), "index": int(index),
            "box": [float(x) for x in box]}


def load_entities(run: Path, experiment: str, normalize: bool) -> List[Entity]:
    ents: List[Entity] = []
    if run.is_dir():
        for p in sorted(run.glob("image_*." + experiment + ".json")):
            m = STEP_FILE_RE.match(p.name)
            if not m or m.group(2) != experiment:
                continue
            data = read_records(p)
            recs = data.get("entities", []) if isinstance(data, dict) else data
            ents += [e for i, r in enumerate(recs) if (e := _entity(r, int(m.group(1)), i, normalize))]
    else:
        data = read_records(run)
        per_step: Dict[int, int] = defaultdict(int)
        for r in [data] if isinstance(data, dict) else data:
            step = int(r["step"])
            if (e := _entity(r, step, per_step[step], normalize)):
                ents.append(e)
            per_step[step] += 1
    return sorted(ents, key=lambda e: (e["step"], e["type"], e["index"]))


def crop_box(box: List[float], size: Tuple[int, int], pad: float) -> Optional[Tuple[int, int, int, int]]:
    """Pad by a fraction of the box size, clip to the image; None if the crop is degenerate."""
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    w, h = size
    x0, y0 = max(0, math.floor(x0 - px)), max(0, math.floor(y0 - py))
    x1, y1 = min(w, math.ceil(x1 + px)), min(h, math.ceil(y1 + py))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def entity_key(e: Entity) -> str:
    return f"{e['step']}|{e['type']}|{e['index']}|" + ",".join(f"{x:g}" for x in e["box"])


# ---- embedding ------------------------------------------------------------
class CropEmbedder:
    """EmbeddingGemma 2, vision + text towers only, -> one L2-normalised vector per crop."""

    def __init__(self, model_id: str, batch_size: int) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        cuda = torch.cuda.is_available()
        # bfloat16 or float32 only: the model card warns float16 overflows into NaNs.
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
        self.model = SentenceTransformer(model_id, device="cuda" if cuda else "cpu",
                                         model_kwargs={"torch_dtype": dtype},
                                         config_kwargs={"audio_config": None})
        self.batch_size = batch_size

    def __call__(self, images) -> "np.ndarray":
        emb = self.model.encode(images, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_tensor=True, show_progress_bar=False)
        return emb.float().cpu().numpy()


def embed_run(run: Path, ents: List[Entity], args: argparse.Namespace, out: Path,
              embedder: CropEmbedder) -> Tuple[List[Entity], "np.ndarray"]:
    """Embeddings of the run's crops, from the cache when it already holds every one of them.
    Returns the entities that could be cropped and their vectors, row-aligned."""
    import numpy as np
    from PIL import Image

    name = run_name(run)
    cache = out / "embeddings" / f"{name}.npz"
    wanted = [entity_key(e) for e in ents]
    if cache.exists() and not args.recompute:
        z = np.load(cache)
        have = {k: i for i, k in enumerate(z["keys"].tolist())}
        skipped = set(z["skipped"].tolist())
        if all(k in have or k in skipped for k in wanted):
            kept = [e for e, k in zip(ents, wanted) if k in have]
            return kept, z["vectors"][[have[entity_key(e)] for e in kept]].astype(np.float32)

    kept, crops, skipped = [], [], []
    by_step: Dict[int, List[Entity]] = defaultdict(list)
    for e in ents:
        by_step[e["step"]].append(e)
    for step, step_ents in sorted(by_step.items()):
        path = args.images / name / f"image_{step}.png"
        if not path.exists():
            skipped += [entity_key(e) for e in step_ents]
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            for e in step_ents:
                box = crop_box(e["box"], im.size, args.pad)
                if box is None or min(box[2] - box[0], box[3] - box[1]) < args.min_box:
                    skipped.append(entity_key(e))
                    continue
                kept.append(e)
                crops.append(im.crop(box))
    if args.save_crops:
        d = out / "crops" / name
        d.mkdir(parents=True, exist_ok=True)
        for e, c in zip(kept, crops):
            c.save(d / f"step{e['step']:02d}_{e['type'].replace(' ', '_')}_{e['index']}.png")
    vectors = embedder(crops) if crops else np.zeros((0, 768), dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, keys=np.array([entity_key(e) for e in kept]), skipped=np.array(skipped),
             vectors=vectors.astype(np.float16))
    return kept, vectors.astype(np.float16).astype(np.float32)


# ---- pairs ----------------------------------------------------------------
def step_pairs(steps: List[int], relation: str, anchor: Optional[int]) -> List[Tuple[int, int]]:
    if relation == "consecutive":
        return list(zip(steps, steps[1:]))
    if anchor is None or anchor not in steps:
        return []
    return [(anchor, s) if s > anchor else (s, anchor) for s in steps if s != anchor]


def compare_run(name: str, ents: List[Entity], vectors, args: argparse.Namespace, writers: Dict,
                stats: Dict) -> None:
    groups: Dict[Tuple[int, str], List[int]] = defaultdict(list)     # (step, type) -> rows
    for i, e in enumerate(ents):
        groups[e["step"], e["type"]].append(i)
    types = {t for _, t in groups}
    if args.type:
        types &= {normalize_type(t, not args.no_normalize) for t in args.type}
    steps = sorted({s for s, _ in groups})

    def sim(i: int, j: int) -> float:
        return float(vectors[i] @ vectors[j])

    if not args.no_within:
        for (step, etype), rows in sorted(groups.items()):
            if etype not in types:
                continue
            for i, j in combinations(rows, 2):
                s = sim(i, j)
                writers["within"].writerow([name, step, etype, ents[i]["index"], ents[j]["index"], round(s, 4)])
                stats["within", step, etype]["sims"].append(s)

    relations = [] if args.no_across else \
        [r for r in ("consecutive", "anchor") if not (r == "consecutive" and args.no_consecutive)
         and not (r == "anchor" and args.no_anchor)]
    anchor = args.anchor if args.anchor is not None else (steps[0] if steps else None)
    for relation in relations:
        for a, b in step_pairs(steps, relation, anchor):
            moving = b if relation == "consecutive" or a == anchor else a
            for etype in sorted(types):
                rows_a, rows_b = groups.get((a, etype), []), groups.get((b, etype), [])
                if not rows_a or not rows_b:
                    continue
                st = stats[relation, moving, etype]
                for j in rows_b:
                    best_i, best = None, -2.0
                    for i in rows_a:
                        s = sim(i, j)
                        writers["across"].writerow([name, relation, a, b, etype, ents[i]["index"],
                                                    ents[j]["index"], round(s, 4)])
                        st["sims"].append(s)
                        if s > best:
                            best_i, best = i, s
                    writers["best"].writerow([name, relation, a, b, etype, ents[j]["index"],
                                              ents[best_i]["index"], round(best, 4)])
                    st["best"].append(best)


def write_summary(path: Path, stats: Dict) -> List[List]:
    rows = []
    for (kind, step, etype), st in sorted(stats.items()):
        sims, best = st["sims"], st["best"]
        if not sims:
            continue
        mean = sum(sims) / len(sims)
        std = (sum((x - mean) ** 2 for x in sims) / len(sims)) ** 0.5
        rows.append([kind, step, etype, len(sims), round(mean, 4), round(std, 4),
                     round(sum(best) / len(best), 4) if best else ""])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "step", "type", "pairs", "mean", "std", "mean_best_match"])
        w.writerows(rows)
    return rows


# ---- main -----------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="entity dataset directory, run files or run directories")
    ap.add_argument("--images", type=Path, required=True,
                    help="run root holding image_<row>/image_<step>.png")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root; each experiment writes to <out>/<experiment>/")
    ap.add_argument("--experiment", action="append", default=[],
                    help="only these experiments (file suffixes, e.g. person, template)")
    ap.add_argument("--type", action="append", default=[], help="only compare these entity types")
    ap.add_argument("--no-normalize", action="store_true", help="use raw entity type names")
    ap.add_argument("--anchor", type=int, default=None,
                    help="fixed step for anchor pairs (default: the first step of each run)")
    ap.add_argument("--no-anchor", action="store_true", help="skip anchor pairs")
    ap.add_argument("--no-consecutive", action="store_true", help="skip consecutive pairs")
    ap.add_argument("--no-within", action="store_true", help="skip same-image pairs")
    ap.add_argument("--no-across", action="store_true", help="skip across-step pairs")
    ap.add_argument("--pad", type=float, default=0.0, help="pad each box by this fraction of its size")
    ap.add_argument("--min-box", type=int, default=DEFAULT_MIN_BOX,
                    help="skip crops whose shorter side is below this many pixels")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--gpu", type=int, default=None, help="pin this GPU (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--recompute", action="store_true", help="ignore cached embeddings")
    ap.add_argument("--save-crops", action="store_true", help="also write the embedded crops as PNGs")
    ap.add_argument("--dry-run", action="store_true", help="plan only: load no model, write nothing")
    args = ap.parse_args(argv)
    normalize = not args.no_normalize

    experiments = find_runs(args.paths)
    if args.experiment:
        experiments = {e: r for e, r in experiments.items() if e in args.experiment}
    if not experiments:
        print("no entity files found", file=sys.stderr)
        return 1

    # Plan first, so --dry-run costs nothing and a real run reports its size up front.
    plans: Dict[str, List[Tuple[Path, List[Entity]]]] = {}
    for experiment, runs in experiments.items():
        plan, n_ents, missing = [], 0, 0
        for run in runs:
            ents = load_entities(run, experiment, normalize)
            if args.type:
                keep = {normalize_type(t, normalize) for t in args.type}
                ents = [e for e in ents if e["type"] in keep]
            if not ents:
                continue
            name = run_name(run)
            missing += sum(not (args.images / name / f"image_{s}.png").exists() for s in {e["step"] for e in ents})
            n_ents += len(ents)
            plan.append((run, ents))
        plans[experiment] = plan
        print(f"[{experiment}] {len(plan)} runs, {n_ents} entities with boxes, "
              f"{missing} referenced images missing under {args.images}", file=sys.stderr)
    if args.dry_run:
        print("(dry-run: no model loaded, nothing written.)", file=sys.stderr)
        return 0

    embedder = CropEmbedder(args.model, args.batch_size)
    out_root = args.out or args.paths[0].resolve().parent / "entity_similarity" / args.paths[0].resolve().name
    for experiment, plan in plans.items():
        if not plan:
            continue
        out = out_root / experiment
        out.mkdir(parents=True, exist_ok=True)
        stats: Dict = defaultdict(lambda: {"sims": [], "best": []})
        files = {k: open(out / f"{k}.csv", "w", newline="") for k in ("within_pairs", "across_pairs", "best_match")}
        writers = {"within": csv.writer(files["within_pairs"]), "across": csv.writer(files["across_pairs"]),
                   "best": csv.writer(files["best_match"])}
        writers["within"].writerow(["run", "step", "type", "a", "b", "similarity"])
        writers["across"].writerow(["run", "relation", "step_a", "step_b", "type", "a", "b", "similarity"])
        writers["best"].writerow(["run", "relation", "step_a", "step_b", "type", "entity", "best", "similarity"])
        try:
            for k, (run, ents) in enumerate(plan, 1):
                kept, vectors = embed_run(run, ents, args, out, embedder)
                compare_run(run_name(run), kept, vectors, args, writers, stats)
                if k % 50 == 0 or k == len(plan):
                    print(f"[{experiment}] {k}/{len(plan)} runs", file=sys.stderr)
        finally:
            for f in files.values():
                f.close()
        rows = write_summary(out / "summary.csv", stats)

        # Console: mean similarity per kind and type, first and last step side by side.
        by: Dict[Tuple[str, str], List[List]] = defaultdict(list)
        for r in rows:
            by[r[0], r[2]].append(r)
        print(f"######## experiment {experiment}")
        for (kind, etype), rs in sorted(by.items()):
            first, last = rs[0], rs[-1]
            best = f"   best match {first[6]} -> {last[6]}" if first[6] != "" else ""
            print(f"  {kind:<11} {etype:<16} step {first[1]:>2} -> {last[1]:>2}: "
                  f"mean {first[4]:.3f} -> {last[4]:.3f}{best}")
        print(f"[{experiment}] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
