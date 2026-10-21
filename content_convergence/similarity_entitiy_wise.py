#!/usr/bin/env python
"""Entity-wise visual similarity: embed every entity's box crop with EmbeddingGemma 2 and compare
crops of the SAME entity type, both inside one image and across the steps of a run.

Each entity record carries a step, a type, a pixel box (x0, y0, x1, y1) and the path of the image
it was detected on ("image"). The crop is embedded with google/embeddinggemma-2 (vision encoder only, audio is not
loaded) into an L2-normalised 768-d vector, and similarities are cosines between those vectors.

Pairs (only entities of the same type are ever compared):
    within        every pair of same-type entities in the same image_N (person 0 vs person 1):
                  do the people of one scene look alike, and does that change over the steps?
    consecutive   every same-type pair between step N and step N+1 of a run
    anchor        every same-type pair between a fixed step K and every other step
                  (--anchor K; default the first step of each run; --no-anchor turns it off)
    For across-step pairs, best_match.csv also keeps, for each entity of the later step, its most
    similar same-type entity of the earlier step: "is there still something like it?".

Input: one entity file per run, or directories of them:
    [<dataset>.]image_<row>[.<experiment>].json  (JSON list or JSON Lines)
        outputs/describe/vlstereoset/qwen3827b_zimage/image_00000.entities_description.json
Each record is {"step", "type", "bbox" (or "box"), "detected_idx", "image", ...}. The crop is taken
from the record's "image" path; relative paths are resolved against --images (default: the
current directory). Records without "image" fall back to --images/image_<row>/image_<step>.png.

Outputs (under <out>/<experiment>/, --out defaulting to <input>/../entity_similarity/<input name>):
    within_pairs.csv     run, step, type, a, b, similarity
    across_pairs.csv     run, relation, step_a, step_b, type, a, b, similarity
    best_match.csv       run, relation, step_a, step_b, type, entity (at step_b), best (at step_a), similarity
    summary.csv          per (kind, step, type): pairs, mean, std; across kinds also mean best match.
                         For across kinds "step" is the moving side (step_b, or the non-anchor step).
    embeddings/<run>.npz cache of the crop embeddings (reused unless --recompute or the boxes changed)
    crops/<run>/         with --save-crops: the PNG crops that were embedded, to check the boxes
    plots/               (unless --no-plot; --plot-only redraws them from the CSVs, no model loaded)
        curves.pdf         mean +- std per step, one panel per type, one line per kind;
                           dashed: mean best match of the across kinds
        distributions.pdf  per-step box plots, rows = types, columns = kinds
        runs/least_stable_<rank>_<run>.png
                           crop grid of the --plot-runs least stable runs (lowest mean consecutive
                           best match): rows = steps, columns = entities; border colour and caption
                           = best match to the previous step (and which entity it was)
    Only the --plot-types types with most pairs are plotted (or the --type ones).

Like the CLIP scores in semantic_validate.py, unrelated crops of the same type still score well
above 0, so read the curves relative to each other, not as absolute values.

Usage:
    ./conda_venv/bin/python content_convergence/similarity_entitiy_wise.py \\
        outputs/describe/vlstereoset/qwen3827b_zimage --gpu 0
    ... --dry-run                 count runs, entities and missing images; load no model
    ... --type person --anchor 1  only persons, anchor at step 1
    ... --save-crops              also write the crops for inspection
    ... --plot-only               redraw the plots of an existing output, no model loaded
"""
from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_MIN_BOX = 256 * 256          # px; smaller crops are skipped
DEFAULT_PLOT_TYPES = 6
DEFAULT_PLOT_RUNS = 6
MAX_GRID_ENTITIES = 12       # columns of a run's crop grid
KINDS = ("within", "consecutive", "anchor")
KIND_COLORS = {"within": "tab:blue", "consecutive": "tab:orange", "anchor": "tab:green"}
ROW_RE = re.compile(r"image_(\d+)")
RUN_FILE_RE = re.compile(r"^(?:.+\.)?image_(\d+)(?:\.(.+))?\.json$")
DEFAULT_EXPERIMENT = "default"


def _pin_gpu_before_torch() -> None:
    """Honour --gpu by setting CUDA_VISIBLE_DEVICES before torch is imported."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu", type=int, default=None)
    known, _ = pre.parse_known_args()
    if known.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_pin_gpu_before_torch()


# ---- loading --------------------------------------------------------------
Entity = Dict  # {"step", "type", "index", "box", "image"}


def normalize_type(etype: str, normalize: bool) -> str:
    """"Chess_Piece" == "chess piece", "trees" == "tree" (unless --no-normalize)."""
    if not normalize:
        return etype
    t = " ".join(etype.lower().replace("_", " ").replace("-", " ").split())
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def find_runs(paths: List[Path]) -> Dict[str, List[Path]]:
    """experiment -> run files, from run files and/or directories holding them."""
    files: List[Path] = []
    for p in paths:
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    runs: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        m = RUN_FILE_RE.match(f.name)
        if m and f.is_file():
            runs[m.group(2) or DEFAULT_EXPERIMENT].append(f)
    return dict(runs)


def read_records(path: Path) -> List[Dict]:
    text = path.read_text()
    try:
        data = json.loads(text)
    except ValueError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [data] if isinstance(data, dict) else data


def run_name(run: Path) -> str:
    """image_<row> for both layouts (vlstereoset.image_00002.json -> image_00002)."""
    m = ROW_RE.search(run.name)
    if not m:
        raise ValueError(f"cannot read the dataset row from {run}")
    return f"image_{m.group(1)}"


def image_path(rec: Dict, run: Path, step: int, images: Path) -> Path:
    """The record's own "image" (relative paths resolved against --images), else the run-root layout."""
    if rec.get("image"):
        p = Path(rec["image"])
        return p if p.is_absolute() else images / p
    return images / run_name(run) / f"image_{step}.png"


def _entity(rec: Dict, run: Path, step: int, position: int, images: Path,
            normalize: bool) -> Optional[Entity]:
    box = rec.get("bbox") or rec.get("box")
    etype = rec.get("entity_type") or rec.get("type")
    if not box or len(box) != 4 or not etype:
        return None
    index = next((rec[k] for k in ("detected_idx", "entity_index", "instance_index", "index") if k in rec),
                 position)
    return {"step": step, "type": normalize_type(etype, normalize), "index": int(index),
            "box": [float(x) for x in box], "image": image_path(rec, run, step, images)}


def load_entities(run: Path, images: Path, normalize: bool) -> List[Entity]:
    ents: List[Entity] = []
    per_step: Dict[int, int] = defaultdict(int)
    for r in read_records(run):
        step = int(r["step"])
        if (e := _entity(r, run, step, per_step[step], images, normalize)):
            ents.append(e)
        per_step[step] += 1
    # Within an image (and type), left to right by the box's left edge.
    return sorted(ents, key=lambda e: (e["step"], e["type"], e["box"][0], e["index"]))


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
    by_image: Dict[Path, List[Entity]] = defaultdict(list)
    for e in ents:
        by_image[e["image"]].append(e)
    for path, img_ents in by_image.items():
        if not path.exists():
            skipped += [entity_key(e) for e in img_ents]
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            for e in img_ents:
                box = crop_box(e["box"], im.size, args.pad)
                if box is None or (box[2] - box[0]) *  (box[3] - box[1]) < args.min_box:
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


# ---- plots ----------------------------------------------------------------
def read_csv(path: Path) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_types(summary: List[Dict], args: argparse.Namespace) -> List[str]:
    """The --type types, else the --plot-types types with most pairs."""
    if args.type:
        wanted = {normalize_type(t, not args.no_normalize) for t in args.type}
        return sorted(wanted & {r["type"] for r in summary})
    pairs: Dict[str, int] = defaultdict(int)
    for r in summary:
        pairs[r["type"]] += int(r["pairs"])
    return sorted(pairs, key=pairs.get, reverse=True)[:args.plot_types]


def _grid(plt, n: int, w: float, h: float, ncols: int = 3):
    ncols = min(n, ncols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def plot_curves(plt, summary: List[Dict], types: List[str], out: Path, title: str) -> None:
    fig, axes = _grid(plt, len(types), 5, 3.6)
    for ax, etype in zip(axes, types):
        for kind in KINDS:
            rs = sorted((r for r in summary if r["kind"] == kind and r["type"] == etype),
                        key=lambda r: int(r["step"]))
            if not rs:
                continue
            x = [int(r["step"]) for r in rs]
            m = [float(r["mean"]) for r in rs]
            s = [float(r["std"]) for r in rs]
            c = KIND_COLORS[kind]
            ax.plot(x, m, marker="o", ms=3, color=c, label=kind)
            ax.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                            color=c, alpha=0.15)
            best = [(xi, float(r["mean_best_match"])) for xi, r in zip(x, rs) if r["mean_best_match"]]
            if best:
                ax.plot(*zip(*best), ls="--", color=c, label=f"{kind} best match")
        ax.set_title(etype)
        ax.set_xlabel("step")
        ax.set_ylabel("cosine similarity")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "curves.pdf")
    plt.close(fig)


def plot_distributions(plt, out: Path, types: List[str], plots: Path, title: str,
                       anchor: Optional[int]) -> None:
    """Steps as in summary.csv: the moving side of a pair (the default anchor is a run's first step,
    so there the moving side is always step_b)."""
    sims: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for r in read_csv(out / "within_pairs.csv"):
        if r["type"] in types:
            sims["within", r["type"], int(r["step"])].append(float(r["similarity"]))
    for r in read_csv(out / "across_pairs.csv"):
        if r["type"] not in types:
            continue
        a, b = int(r["step_a"]), int(r["step_b"])
        moving = a if r["relation"] == "anchor" and b == anchor else b
        sims[r["relation"], r["type"], moving].append(float(r["similarity"]))
    kinds = [k for k in KINDS if any(key[0] == k for key in sims)]
    if not kinds or not types:
        return
    fig, axes = plt.subplots(len(types), len(kinds), figsize=(5 * len(kinds), 3 * len(types)),
                             squeeze=False)
    for i, etype in enumerate(types):
        for j, kind in enumerate(kinds):
            ax = axes[i][j]
            steps = sorted(s for k, t, s in sims if k == kind and t == etype)
            if steps:
                ax.boxplot([sims[kind, etype, s] for s in steps], positions=steps, widths=0.6,
                           showfliers=False, patch_artist=True,
                           boxprops={"facecolor": KIND_COLORS[kind], "alpha": 0.4})
            ax.set_title(f"{etype}: {kind}", fontsize=9)
            ax.set_xlabel("step")
            ax.set_ylabel("cosine similarity")
            ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plots / "distributions.pdf")
    plt.close(fig)


def _draw_boxes(plt, ax, ents: List[Entity], image: Path, fontsize: float) -> None:
    """Yellow boxes and labels of the entities detected on `image`."""
    for e in ents:
        if e["image"] == image:
            x0, y0, x1, y1 = e["box"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.2, ec="yellow"))
            ax.text(x0, y0, f"{e['type']} {e['index']}", fontsize=fontsize, color="black", va="bottom",
                    bbox={"facecolor": "yellow", "pad": 1, "lw": 0})


def plot_run_grid(plt, run: Path, ents: List[Entity], best: Dict, args: argparse.Namespace,
                  path: Path, title: str, norm) -> None:
    """Top: the run's original image. Then one row per step: the step's full image with its boxes,
    followed by the crops (left to right) with their best match to the previous step."""
    from PIL import Image

    by_step: Dict[int, List[Entity]] = defaultdict(list)
    for e in ents:
        by_step[e["step"]].append(e)
    steps = sorted(by_step)
    ncols = min(MAX_GRID_ENTITIES, max(len(v) for v in by_step.values()))
    full_w = 2.5  # width of the full-image column, in crop columns
    width = max(1.5 * (ncols + full_w), 5.0)
    fig = plt.figure(figsize=(width, 1.7 * len(steps) + 4))
    gs = fig.add_gridspec(len(steps) + 1, ncols + 1, height_ratios=[4 / 1.7] + [1] * len(steps),
                          width_ratios=[full_w] + [1] * ncols)
    cmap = plt.get_cmap("RdYlGn")
    images: Dict[Path, "Image.Image"] = {}

    def load(p: Path):
        if p not in images:
            images[p] = Image.open(p).convert("RGB") if p.exists() else None
        return images[p]

    # Top row: the run's original image (image_0.png, the input the first step is detected on).
    first = by_step[steps[0]][0]["image"]
    orig = first.parent / "image_0.png"
    orig = orig if orig.exists() else first
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    if load(orig) is not None:
        ax.imshow(load(orig))
        _draw_boxes(plt, ax, ents, orig, 6)
    ax.set_title(f"{title}\noriginal: {orig.name}", fontsize=9)

    for i, step in enumerate(steps):
        # Column 0: the full image of this step.
        step_image = by_step[step][0]["image"]
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"step {step}", fontsize=8)
        if load(step_image) is not None:
            ax.imshow(load(step_image))
            _draw_boxes(plt, ax, by_step[step], step_image, 4)
        ax.set_title(step_image.name, fontsize=6)
        for j in range(ncols):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= len(by_step[step]):
                ax.axis("off")
                continue
            e = by_step[step][j]
            im = load(e["image"])
            box = crop_box(e["box"], im.size, args.pad) if im is not None else None
            if box is not None:
                ax.imshow(im.crop(box))
            caption = f"{e['type']} {e['index']}"
            b = best.get((step, e["type"], e["index"]))
            if b is not None:
                caption += f"\n{b[1]:.2f} <- {b[0]}"
                for sp in ax.spines.values():
                    sp.set_edgecolor(cmap(norm(b[1])))
                    sp.set_linewidth(3)
            ax.set_title(caption, fontsize=6)
    for im in images.values():
        if im is not None:
            im.close()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_experiment(out: Path, plan: List[Tuple[Path, List[Entity]]], args: argparse.Namespace) -> None:
    if not (out / "summary.csv").exists():
        print(f"no summary.csv in {out}, nothing to plot", file=sys.stderr)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    summary = read_csv(out / "summary.csv")
    types = plot_types(summary, args)
    plots = out / "plots"
    (plots / "runs").mkdir(parents=True, exist_ok=True)
    title = f"{out.parent.name}/{out.name}"
    plot_curves(plt, summary, types, plots, title)
    plot_distributions(plt, out, types, plots, title, args.anchor)

    # Crop grids of the least stable runs, by mean consecutive best match.
    best: Dict[str, Dict] = defaultdict(dict)
    per_run: Dict[str, List[float]] = defaultdict(list)
    for r in read_csv(out / "best_match.csv"):
        if r["relation"] != "consecutive" or r["type"] not in types:
            continue
        s = float(r["similarity"])
        best[r["run"]][int(r["step_b"]), r["type"], int(r["entity"])] = (int(r["best"]), s)
        per_run[r["run"]].append(s)
    if per_run and args.plot_runs > 0:
        stability = sorted(per_run, key=lambda k: sum(per_run[k]) / len(per_run[k]))
        allv = sorted(v for vs in per_run.values() for v in vs)
        norm = Normalize(allv[len(allv) // 20], allv[-1 - len(allv) // 20], clip=True)  # 5th-95th pct
        runs = {run_name(run): (run, ents) for run, ents in plan}
        for rank, name in enumerate(stability[:args.plot_runs], 1):
            if name not in runs:
                continue
            run, ents = runs[name]
            ents = [e for e in ents if e["type"] in types]
            mean = sum(per_run[name]) / len(per_run[name])
            plot_run_grid(plt, run, ents, best[name], args, plots / "runs" / f"least_stable_{rank}_{name}.png",
                          f"{name} (least stable #{rank}, mean consecutive best match "
                          f"{mean:.3f})", norm)
    print(f"[{out.name}] plots in {plots}", file=sys.stderr)


def compute_experiment(experiment: str, plan: List[Tuple[Path, List[Entity]]], out: Path,
                       args: argparse.Namespace, embedder: CropEmbedder) -> None:
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


# ---- main -----------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="directories of entity run files, or the files")
    ap.add_argument("--images", type=Path, default=Path("."),
                    help="root that relative \"image\" paths are resolved against (default: current "
                         "directory); for records without \"image\", the run root holding "
                         "image_<row>/image_<step>.png")
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
    ap.add_argument("--no-plot", action="store_true", help="do not write the plots")
    ap.add_argument("--plot-only", action="store_true",
                    help="only redraw the plots from an existing output (same --out/--anchor); no model loaded")
    ap.add_argument("--plot-types", type=int, default=DEFAULT_PLOT_TYPES,
                    help="plot the types with most pairs (default: %(default)s; --type overrides)")
    ap.add_argument("--plot-runs", type=int, default=DEFAULT_PLOT_RUNS,
                    help="crop grids for this many least stable runs (default: %(default)s)")
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
            ents = load_entities(run, args.images, normalize)
            if args.type:
                keep = {normalize_type(t, normalize) for t in args.type}
                ents = [e for e in ents if e["type"] in keep]
            if not ents:
                continue
            missing += sum(not p.exists() for p in {e["image"] for e in ents})
            n_ents += len(ents)
            plan.append((run, ents))
        plans[experiment] = plan
        print(f"[{experiment}] {len(plan)} runs, {n_ents} entities with boxes, "
              f"{missing} referenced images missing under {args.images}", file=sys.stderr)
    if args.dry_run:
        print("(dry-run: no model loaded, nothing written.)", file=sys.stderr)
        return 0

    embedder = None if args.plot_only else CropEmbedder(args.model, args.batch_size)
    out_root = args.out or args.paths[0].resolve().parent / "entity_similarity" / args.paths[0].resolve().name
    for experiment, plan in plans.items():
        if not plan:
            continue
        out = out_root / experiment
        if not args.plot_only:
            compute_experiment(experiment, plan, out, args, embedder)
        if not args.no_plot:
            plot_experiment(out, plan, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
