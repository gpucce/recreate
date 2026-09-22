#!/usr/bin/env python
"""Measure and PLOT semantic drift between SUBSEQUENT steps of a content-convergence run.

Same pairs as validate_run_diffs.py -- every consecutive (image_N, image_N+1) and
(prompt_N, prompt_N+1), plus every step against the FIXED step --anchor K (3 by
default; --no-anchor skips it) -- but instead of asking a VLM to describe the
differences in words, each step is turned into an embedding and the pair is scored
with cosine similarity.

The two relations answer different questions, and get one plot panel each:
    consecutive  how far did THIS step move?   A converged loop sits flat and high.
    anchor       how far from step K are we?   This is where small per-step moves show
                 up as accumulated drift: the step-to-step curve can be flat at 0.95
                 while the anchor curve slides steadily downhill.

--anchor takes a literal step index, so with --mode both it pins image_K and prompt_K.
Those are one loop position apart (prompt_{k+1} describes image_k), which is harmless
for reading each curve but worth knowing before comparing the two panels closely. That gives a number per step instead of a verdict, so the convergence
trajectory can be plotted: a run that is converging should show similarity rising
toward 1.0 as N grows.

Two encoders, as the two modalities need different ones:

    images  -- CLIP image tower (openai/clip-vit-large-patch14), pooled image features
    prompts -- a sentence embedder (intfloat/multilingual-e5-large), mean-pooled

The two live in DIFFERENT vector spaces, so image and prompt similarities are each
meaningful over time but are NOT comparable to each other in absolute value. Compare
their shapes, not their heights.

Aligning the two curves
    The loop is  image_k -> prompt_{k+1} (describes image_k) -> image_{k+1}, so the
    prompt pair (prompt_{k+1}, prompt_{k+2}) straddles exactly the same loop iteration
    as the image pair (image_k, image_{k+1}). Both are therefore plotted at
    x = iteration k, which is step_a for images and step_a - 1 for prompts.

Long prompts
    Prompts here run to ~650 words, past e5's 512-token window. Rather than truncate
    and silently drop the back half of a description, the token stream is split into
    windows, each window embedded, and the windows mean-pooled.

This script is READ-ONLY with respect to run directories. It never writes into
image_*/; results go to a sibling `semantic/` directory. It is safe to run against a
live run: pairs whose files do not exist yet are skipped.

Usage:
    ./conda_venv/bin/python semantic_validate.py bbq_runs/image_09134
    ./conda_venv/bin/python semantic_validate.py bbq_runs/image_* --gpu 0
    ./conda_venv/bin/python semantic_validate.py bbq_runs/image_* --mode images
    ./conda_venv/bin/python semantic_validate.py bbq_runs/image_* --anchor 5
    ./conda_venv/bin/python semantic_validate.py bbq_runs/image_* --no-anchor
    ./conda_venv/bin/python semantic_validate.py bbq_runs/image_09134 --per-run-plots
    ./conda_venv/bin/python semantic_validate.py bbq_runs/image_09134 --dry-run

Reading the numbers
    Both encoders squeeze same-domain items into a narrow high band -- two UNRELATED
    photos still score ~0.88 under CLIP, and two unrelated prompts ~0.92 under e5 --
    so an absolute 0.95 says very little on its own. The script therefore samples
    cross-run pairs to estimate that floor and draws it on the plot as a dashed line
    (--baseline-pool 0 turns it off). Read the curve as its distance ABOVE the floor.

Outputs (under --out, default <runs-parent>/semantic):
    <run>.<kind>.sim.jsonl   one record per pair: cosine similarity + provenance
    semantic_summary.csv     mean/std/median cosine per (kind, relation, iteration)
    semantic_baseline.json   the unrelated-pair floor per kind
    semantic_summary.png     the aggregate convergence plot
    plots/<run>.png          per-run curves (only with --per-run-plots)

VRAM note:
    Much lighter than validate_run_diffs.py: CLIP-L/14 + e5-large is roughly 2.5 GB,
    no generative model is loaded. Pin an idle card with --gpu N.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
RECREATE_DIR = HERE / "recreate"

DEFAULT_IMAGE_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_TEXT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_MAX_SIZE = 512      # matches describe.DEFAULT_MAX_SIZE / validate_run_diffs
DEFAULT_TEXT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_BASELINE_POOL = 48   # runs sampled to estimate the unrelated-pair floor
DEFAULT_ANCHOR = 3           # also score every step against this fixed step
E5_PREFIX = "query: "       # e5 is trained with this prefix; omitting it costs accuracy


def _pin_gpu_before_torch() -> None:
    """Honour --gpu by setting CUDA_VISIBLE_DEVICES before torch is imported.

    Once torch initialises CUDA the variable is ignored, so this has to happen
    before the imports below pull torch in.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu", type=int, default=None)
    known, _ = pre.parse_known_args()
    if known.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_pin_gpu_before_torch()

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")                                    # headless: write PNGs, open no window
import matplotlib.pyplot as plt                          # noqa: E402
import torch                                             # noqa: E402
from transformers import AutoModel, AutoProcessor, AutoTokenizer   # noqa: E402

sys.path.insert(0, str(RECREATE_DIR))
import describe  # noqa: E402  (must follow the GPU pin)

# Pair discovery is shared with validate_run_diffs so the two scripts can never drift
# apart on what "subsequent" means.
from validate_run_diffs import (  # noqa: E402,F401
    available_steps, anchor_pairs, iter_pairs, pair_paths, pairs_for,
)


# ---- embedding ------------------------------------------------------------
def _as_tensor(output: Any) -> "torch.Tensor":
    """get_image_features returns a bare tensor on transformers 4 but an output
    object (with the projection written back into pooler_output) on 5."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise ValueError(f"cannot read image features from {type(output).__name__}")
    return pooled


class ImageEmbedder:
    """CLIP image tower -> one L2-normalised vector per image."""

    def __init__(self, model_id: str, device: str, dtype: Any, max_size: int) -> None:
        self.model_id = model_id
        self.device = device
        self.max_size = max_size
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()

    @torch.no_grad()
    def encode(self, paths: Sequence[Path], batch_size: int) -> "torch.Tensor":
        vectors = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images = [describe.load_image(p, self.max_size) for p in chunk]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features = _as_tensor(self.model.get_image_features(**inputs)).float()
            vectors.append(torch.nn.functional.normalize(features, dim=-1).cpu())
        return torch.cat(vectors) if vectors else torch.empty(0)


class TextEmbedder:
    """Sentence embedder -> one L2-normalised vector per prompt.

    Prompts longer than the model's window are split into windows and the windows
    are mean-pooled, so the tail of a long description still reaches the vector.
    """

    def __init__(self, model_id: str, device: str, dtype: Any, max_length: int) -> None:
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
        limit = getattr(self.model.config, "max_position_embeddings", max_length)
        # BERT-family positional tables include the special tokens; leave room for them.
        self.max_length = min(max_length, limit - 2)
        self.prefix = E5_PREFIX if "e5" in model_id.lower() else ""

    def _windows(self, text: str) -> List[str]:
        """Split a prompt into at-most-max_length token windows, returned as text.

        The windows go back through the tokenizer as strings rather than as raw id
        lists: transformers 5 dropped build_inputs_with_special_tokens from the fast
        tokenizers, and re-encoding is the version-proof way to get the model's own
        special tokens and padding.
        """
        ids = self.tokenizer(self.prefix + text, add_special_tokens=False,
                             verbose=False)["input_ids"]
        body = max(1, self.max_length - 2)   # leave room for <s> ... </s>
        if not ids:
            return [self.prefix.strip() or " "]
        return [self.tokenizer.decode(ids[i : i + body], skip_special_tokens=True)
                for i in range(0, len(ids), body)]

    @torch.no_grad()
    def _encode_windows(self, windows: List[str], batch_size: int) -> "torch.Tensor":
        out = []
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            inputs = self.tokenizer(chunk, padding=True, truncation=True,
                                    max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            hidden = self.model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"]
            masked = hidden * mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            out.append(pooled.cpu())
        return torch.cat(out)

    def encode(self, texts: Sequence[str], batch_size: int) -> "torch.Tensor":
        """One vector per text; a text's windows are pooled before normalising."""
        per_text = [self._windows(t) for t in texts]
        flat = [w for windows in per_text for w in windows]
        if not flat:
            return torch.empty(0)
        pooled = self._encode_windows(flat, batch_size)
        vectors, cursor = [], 0
        for windows in per_text:
            take = pooled[cursor : cursor + len(windows)]
            cursor += len(windows)
            vectors.append(take.mean(dim=0))
        stacked = torch.stack(vectors)
        return torch.nn.functional.normalize(stacked, dim=-1)


def cosine(a: "torch.Tensor", b: "torch.Tensor") -> float:
    """Both inputs are already L2-normalised, so the dot product is the cosine."""
    return float(torch.dot(a, b))


# ---- per-run scoring ------------------------------------------------------
def iteration_of(kind: str, step: int) -> int:
    """Map a step onto the loop iteration it belongs to, so the curves line up.

    prompt_{k+1} describes image_k, so a prompt index sits one ahead of the image
    index covering the same iteration.
    """
    return step if kind == "images" else step - 1


def pair_iteration(kind: str, relation: str, a: int, b: int, anchor: Optional[int]) -> int:
    """x position for a pair.

    Consecutive: the iteration the pair straddles, i.e. its earlier step. Anchor: the
    position of the MOVING side, since the other side is pinned -- that makes the anchor
    curve read as "similarity to the reference, over time".
    """
    if relation == "consecutive":
        return iteration_of(kind, a)
    return iteration_of(kind, b if a == anchor else a)


def score_run(
    run_dir: Path,
    kind: str,
    relation: str,
    pairs: List[Tuple[int, int]],
    image_embedder: Optional[ImageEmbedder],
    text_embedder: Optional[TextEmbedder],
    batch_size: int,
    anchor: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Embed every step ONCE, then score each consecutive pair from those vectors."""
    steps = sorted({s for pair in pairs for s in pair})
    paths = {s: pair_paths(run_dir, kind, s, s)[0] for s in steps}

    if kind == "images":
        assert image_embedder is not None
        vectors = image_embedder.encode([paths[s] for s in steps], batch_size)
        model_id = image_embedder.model_id
    else:
        assert text_embedder is not None
        texts = [paths[s].read_text(encoding="utf-8").strip() for s in steps]
        vectors = text_embedder.encode(texts, batch_size)
        model_id = text_embedder.model_id

    index = {s: i for i, s in enumerate(steps)}
    records = []
    for a, b in pairs:
        record: Dict[str, Any] = {
            "run": run_dir.name,
            "kind": kind,
            "relation": relation,
            "step_a": a,
            "step_b": b,
            "distance": b - a,
            "iteration": pair_iteration(kind, relation, a, b, anchor),
            "file_a": paths[a].name,
            "file_b": paths[b].name,
            "model": model_id,
            "cosine": round(cosine(vectors[index[a]], vectors[index[b]]), 6),
        }
        if relation == "anchor":
            record["anchor"] = anchor
        records.append(record)
    return records


# ---- unrelated-pair baseline ----------------------------------------------
def _content_hash(path: Path, kind: str) -> str:
    """Identify byte-identical steps so duplicate seeds cannot inflate the floor."""
    if kind == "images":
        return hashlib.md5(path.read_bytes()).hexdigest()
    return hashlib.md5(path.read_text(encoding="utf-8").strip().encode()).hexdigest()


def baseline_floor(
    run_dirs: List[Path],
    kind: str,
    embedder: Any,
    pool_size: int,
    batch_size: int,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """Mean cosine between steps of DIFFERENT runs -- the "unrelated" floor.

    Both encoders put any two same-domain items in a narrow high band (two unrelated
    photos still score ~0.88 under CLIP), so a 0.95 on the convergence curve means
    nothing until you can see the floor it sits above. One step is sampled per run,
    all cross pairs are scored, and byte-identical steps are dropped: the BBQ-V sample
    contains a few repeated seed images, which would otherwise pull the floor toward 1.
    """
    if pool_size <= 0 or len(run_dirs) < 2:
        return None
    chosen = rng.sample(run_dirs, min(pool_size, len(run_dirs)))
    picks: List[Path] = []
    for run_dir in chosen:
        steps = available_steps(run_dir, kind)
        if steps:
            picks.append(pair_paths(run_dir, kind, rng.choice(steps), 0)[0])
    if len(picks) < 2:
        return None

    if kind == "images":
        vectors = embedder.encode(picks, batch_size)
    else:
        vectors = embedder.encode([p.read_text(encoding="utf-8").strip() for p in picks], batch_size)

    hashes = [_content_hash(p, kind) for p in picks]
    values = [cosine(vectors[i], vectors[j])
              for i, j in itertools.combinations(range(len(picks)), 2)
              if hashes[i] != hashes[j]]
    if not values:
        return None
    return {
        "kind": kind,
        "n_pairs": len(values),
        "n_runs": len(picks),
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 6),
        "p95": round(sorted(values)[int(0.95 * (len(values) - 1))], 6),
    }


# ---- plotting -------------------------------------------------------------
KIND_STYLE = {
    "images":  {"color": "#1f77b4", "marker": "o", "label": "images (CLIP)"},
    "prompts": {"color": "#d62728", "marker": "s", "label": "prompts (e5)"},
}


def _finish_axes(axes: Any, title: str, relation: str = "consecutive",
                 anchor: Optional[int] = None) -> None:
    if relation == "anchor":
        reference = f"step {anchor}" if anchor is not None else "the anchor"
        axes.set_xlabel("loop iteration k  (position of the moving step)")
        axes.set_ylabel(f"cosine similarity to {reference}")
    else:
        axes.set_xlabel("loop iteration k  (image_k -> image_k+1)")
        axes.set_ylabel("cosine similarity of subsequent steps")
    axes.set_title(title)
    axes.grid(alpha=0.3, linestyle=":")
    axes.legend(loc="lower right", fontsize=9)


def relations_present(records: List[Dict[str, Any]]) -> List[str]:
    """consecutive first, anchor second, and only those actually scored."""
    found = {r.get("relation", "consecutive") for r in records}
    return [rel for rel in ("consecutive", "anchor") if rel in found]


def panel_title(relation: str, anchor: Optional[int]) -> str:
    if relation == "anchor":
        return f"drift from the fixed step {anchor}" if anchor is not None else "drift from anchor"
    return "step-to-step (N vs N+1)"


def plot_per_run(run_name: str, records: List[Dict[str, Any]], out_path: Path,
                 anchor: Optional[int] = None) -> None:
    rels = relations_present(records)
    figure, axes_list = plt.subplots(1, len(rels), figsize=(7.5 * len(rels), 4.5),
                                     squeeze=False)
    for axes, relation in zip(axes_list[0], rels):
        for kind, style in KIND_STYLE.items():
            rows = sorted((r for r in records
                           if r["kind"] == kind and r.get("relation", "consecutive") == relation),
                          key=lambda r: r["iteration"])
            if not rows:
                continue
            axes.plot([r["iteration"] for r in rows], [r["cosine"] for r in rows],
                      linewidth=1.8, markersize=4, **style)
        _finish_axes(axes, panel_title(relation, anchor), relation, anchor)
    figure.suptitle(f"semantic drift - {run_name}")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def aggregate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """mean / std / median / n per (kind, relation, iteration), ascending."""
    buckets: Dict[Tuple[str, str, int], List[float]] = {}
    for record in records:
        key = (record["kind"], record.get("relation", "consecutive"), record["iteration"])
        buckets.setdefault(key, []).append(record["cosine"])
    rows = []
    for (kind, relation, iteration), values in sorted(buckets.items()):
        rows.append({
            "kind": kind,
            "relation": relation,
            "iteration": iteration,
            "n": len(values),
            "mean": round(statistics.fmean(values), 6),
            "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 6),
            "median": round(statistics.median(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        })
    return rows


def plot_aggregate(rows: List[Dict[str, Any]], n_runs: int, out_path: Path,
                   baselines: Optional[Dict[str, Dict[str, Any]]] = None,
                   anchor: Optional[int] = None) -> None:
    """One panel per relation: they answer different questions and share no y-scale
    story, so overlaying all four curves on one axes would just be four lines."""
    rels = relations_present(rows)
    figure, axes_list = plt.subplots(1, len(rels), figsize=(8.5 * len(rels), 5),
                                     squeeze=False)
    for axes, relation in zip(axes_list[0], rels):
        for kind, style in KIND_STYLE.items():
            series = [r for r in rows
                      if r["kind"] == kind and r.get("relation", "consecutive") == relation]
            if not series:
                continue
            xs = [r["iteration"] for r in series]
            means = [r["mean"] for r in series]
            lows = [r["mean"] - r["std"] for r in series]
            highs = [r["mean"] + r["std"] for r in series]
            axes.plot(xs, means, linewidth=2, markersize=5, **style)
            axes.fill_between(xs, lows, highs, color=style["color"], alpha=0.15, linewidth=0)
            floor = (baselines or {}).get(kind)
            if floor:
                # Without this line the curve's height is unreadable: it is the score two
                # UNRELATED runs get, i.e. where "no shared content at all" already sits.
                axes.axhline(floor["mean"], color=style["color"], linestyle="--",
                             linewidth=1.2, alpha=0.7,
                             label=f"{kind} unrelated floor ({floor['mean']:.3f})")
        _finish_axes(axes, panel_title(relation, anchor), relation, anchor)
    suffix = "run" if n_runs == 1 else "runs"
    figure.suptitle(f"semantic convergence across {n_runs} {suffix}  (mean +/- 1 sd)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def print_trajectory(run_name: str, kind: str, relation: str,
                     records: List[Dict[str, Any]]) -> None:
    rows = sorted((r for r in records
                   if r["kind"] == kind and r.get("relation", "consecutive") == relation),
                  key=lambda r: r["iteration"])
    if not rows:
        return
    chain = " ".join(f"{r['cosine']:.3f}" for r in rows)
    print(f"  {run_name} [{kind}/{relation}] cos by iteration: {chain}")


# ---- io -------------------------------------------------------------------
def record_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """What makes a pair unique. Both steps are needed now that anchor pairs exist:
    they all share one step_a, so (kind, step_a) would collapse them into one. Records
    written before anchors existed carry no 'relation' and are consecutive."""
    return (record.get("kind"), record.get("relation", "consecutive"),
            record.get("step_a"), record.get("step_b"))


def existing_records(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    """Records already on disk plus their keys, so a rerun resumes."""
    if not path.exists():
        return [], set()
    records, keys = [], set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
            keys.add(record_key(record))
    return records, keys


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = ["kind", "relation", "iteration", "n", "mean", "std", "median", "min", "max"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed subsequent steps of a run and plot their cosine similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("runs", type=Path, nargs="+",
                        help="Run directories (shell globs work: bbq_runs/image_*).")
    parser.add_argument("--mode", choices=("images", "prompts", "both"), default="both",
                        help="Score subsequent images, subsequent prompts, or both (default both).")
    parser.add_argument("--step", type=int, default=None,
                        help="Score only this pair (N vs N+1). Default: every consecutive pair.")
    parser.add_argument("--anchor", type=int, default=DEFAULT_ANCHOR,
                        help="ALSO score every step against this FIXED step (default "
                             "%(default)s: 3-vs-0, 3-vs-1, 3-vs-2, 3-vs-4, ...), plotted as a "
                             "second panel. Measures total drift from one reference rather than "
                             "step-to-step movement; nearly free here, as the embeddings are "
                             "computed either way. --no-anchor turns it off.")
    parser.add_argument("--no-anchor", action="store_true",
                        help="Consecutive pairs only: skip the fixed-step comparison.")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Pin a GPU via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="Execution device. Default: auto.")
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL,
                        help=f"Image embedding model (default {DEFAULT_IMAGE_MODEL}).")
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL,
                        help=f"Text embedding model (default {DEFAULT_TEXT_MODEL}).")
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE,
                        help=f"Resize longest side before embedding (default {DEFAULT_MAX_SIZE}).")
    parser.add_argument("--text-max-length", type=int, default=DEFAULT_TEXT_MAX_LENGTH,
                        help=f"Token window; longer prompts are split and pooled (default {DEFAULT_TEXT_MAX_LENGTH}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Embedding batch size (default {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir. Default: <runs-parent>/semantic.")
    parser.add_argument("--baseline-pool", type=int, default=DEFAULT_BASELINE_POOL,
                        help=f"Runs sampled for the unrelated-pair floor drawn on the plot; "
                             f"0 disables it (default {DEFAULT_BASELINE_POOL}).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the baseline sampling (default 0).")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip the cross-run aggregate (CSV, baseline, summary plot) and "
                             "write only the per-pair JSONL. For parallel shards, which would "
                             "otherwise each overwrite the same summary with a partial one; "
                             "produce it once afterwards with a pass over every run.")
    parser.add_argument("--per-run-plots", action="store_true",
                        help="Also write one PNG per run under <out>/plots (200 runs = 200 files).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo pairs already present in the JSONL (default: skip them).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the pairs that would be scored; load no model.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.no_anchor:
        args.anchor = None   # everything downstream reads args.anchor is None as 'off'

    run_dirs = [r for r in args.runs if r.is_dir()]
    for path in (str(r) for r in args.runs if not r.is_dir()):
        print(f"warning: not a directory, skipping: {path}", file=sys.stderr)
    if not run_dirs:
        print("error: no run directory to work on.", file=sys.stderr)
        return 2

    kinds = ("images", "prompts") if args.mode == "both" else (args.mode,)
    relations = ("consecutive",) if args.anchor is None else ("consecutive", "anchor")
    out_dir = args.out or (run_dirs[0].resolve().parent / "semantic")

    # Plan first, so --dry-run costs nothing and a real run reports its size up front.
    plan: List[Tuple[Path, str, str, List[Tuple[int, int]]]] = []
    missing_anchor: Dict[str, int] = {}
    for run_dir in run_dirs:
        for kind in kinds:
            for relation in relations:
                pairs = pairs_for(run_dir, kind, relation, args.step, args.anchor)
                if pairs:
                    plan.append((run_dir, kind, relation, pairs))
                elif relation == "anchor":
                    # Counted, not printed per run: a prompt anchor of 0 does not exist in
                    # ANY run (prompts start at 1), and 200 identical notes bury the plan.
                    missing_anchor[kind] = missing_anchor.get(kind, 0) + 1
                else:
                    print(f"note: {run_dir.name} [{kind}]: no consecutive pair available yet.",
                          file=sys.stderr)
    for kind, count in sorted(missing_anchor.items()):
        print(f"note: [{kind}] step {args.anchor} is not present in {count} run(s); "
              f"no anchor comparison for those.", file=sys.stderr)
    total_pairs = sum(len(p) for _, _, _, p in plan)
    if total_pairs == 0:
        print("nothing to score.", file=sys.stderr)
        return 1

    anchor_note = "" if args.anchor is None else f"   anchor: step {args.anchor}"
    print(f"runs: {len(run_dirs)}   mode: {args.mode}{anchor_note}   pairs: {total_pairs}")
    print(f"out : {out_dir}")

    if args.dry_run:
        for run_dir, kind, relation, pairs in plan:
            for a, b in pairs:
                path_a, path_b = pair_paths(run_dir, kind, a, b)
                print(f"  [dry-run] {run_dir.name} [{kind}/{relation}] "
                      f"{path_a.name} vs {path_b.name}")
        print("\n(dry-run: no model loaded, nothing written.)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = describe.resolve_device_and_dtype(args.device)
    image_embedder = text_embedder = None
    if any(kind == "images" for _, kind, _, _ in plan):
        print(f"loading image encoder {args.image_model} ...", flush=True)
        image_embedder = ImageEmbedder(args.image_model, device, dtype, args.max_size)
    if any(kind == "prompts" for _, kind, _, _ in plan):
        print(f"loading text encoder {args.text_model} ...", flush=True)
        text_embedder = TextEmbedder(args.text_model, device, dtype, args.text_max_length)
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"loaded on {device}" + (f" (CUDA_VISIBLE_DEVICES={pinned})" if pinned else ""), flush=True)

    all_records: List[Dict[str, Any]] = []
    per_run: Dict[str, List[Dict[str, Any]]] = {}
    scored = 0
    failures = 0

    for run_dir, kind, relation, pairs in plan:
        out_path = out_dir / f"{run_dir.name}.{kind}.sim.jsonl"
        known, seen = ([], set()) if args.overwrite else existing_records(out_path)
        wanted = {(kind, relation, a, b) for a, b in pairs}
        todo = [(a, b) for a, b in pairs if (kind, relation, a, b) not in seen]
        # Recorded pairs still feed the plots, so a resumed run charts the whole curve.
        reused = [r for r in known if record_key(r) in wanted]
        print(f"\n== {run_dir.name} [{kind}/{relation}]  {len(todo)} to score, "
              f"{len(reused)} already recorded -> {out_path.name}", flush=True)

        fresh: List[Dict[str, Any]] = []
        if todo:
            try:
                fresh = score_run(run_dir, kind, relation, todo, image_embedder,
                                  text_embedder, args.batch_size, args.anchor)
            except (ValueError, OSError) as exc:
                # A half-written PNG from a live run lands here; keep going.
                print(f"  [{kind}/{relation}] FAILED: {exc}", file=sys.stderr, flush=True)
                failures += 1
                continue
            with out_path.open("a", encoding="utf-8") as handle:
                for record in fresh:
                    handle.write(json.dumps(record) + "\n")
            scored += len(fresh)

        records = reused + fresh
        for record in records:
            record.setdefault("relation", "consecutive")
            record.setdefault("iteration", pair_iteration(
                record["kind"], record["relation"], record["step_a"], record["step_b"],
                record.get("anchor", args.anchor)))
        all_records.extend(records)
        per_run.setdefault(run_dir.name, []).extend(records)
        print_trajectory(run_dir.name, kind, relation, records)

    if not all_records:
        print("\nnothing scored and nothing on disk to plot.", file=sys.stderr)
        return 1

    if args.per_run_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        for run_name, records in per_run.items():
            plot_per_run(run_name, records, plots_dir / f"{run_name}.png", args.anchor)
        print(f"\nper-run plots: {len(per_run)} file(s) under {plots_dir}")

    if args.no_summary:
        print(f"\n{'-' * 72}")
        print(f"summary: {scored} pair(s) scored, {failures} failed "
              f"(--no-summary: no aggregate written)")
        return 1 if failures else 0

    baselines: Dict[str, Dict[str, Any]] = {}
    if args.baseline_pool > 0 and len(run_dirs) >= 2:
        rng = random.Random(args.seed)
        print("\nestimating unrelated-pair floor ...", flush=True)
        for kind in kinds:
            embedder = image_embedder if kind == "images" else text_embedder
            if embedder is None:
                continue
            floor = baseline_floor(run_dirs, kind, embedder, args.baseline_pool,
                                   args.batch_size, rng)
            if floor:
                baselines[kind] = floor
                print(f"  {kind:>7}: {floor['mean']:.4f} +/- {floor['std']:.4f} "
                      f"over {floor['n_pairs']} cross-run pair(s)")
        if baselines:
            (out_dir / "semantic_baseline.json").write_text(
                json.dumps(baselines, indent=2) + "\n", encoding="utf-8")
    elif len(run_dirs) < 2:
        print("\nnote: unrelated-pair floor needs >= 2 runs; skipping it.", file=sys.stderr)

    rows = aggregate(all_records)
    csv_path = out_dir / "semantic_summary.csv"
    png_path = out_dir / "semantic_summary.png"
    write_summary_csv(rows, csv_path)
    plot_aggregate(rows, len(per_run), png_path, baselines, args.anchor)

    print("\n" + "-" * 72)
    for relation in relations_present(rows):
        print(f"[{panel_title(relation, args.anchor)}]")
        for kind in kinds:
            series = [r for r in rows
                      if r["kind"] == kind and r.get("relation", "consecutive") == relation]
            if len(series) < 2:
                continue
            first, last = series[0], series[-1]
            arrow = "up" if last["mean"] > first["mean"] else "down"
            note = ""
            floor = baselines.get(kind)
            if floor:
                note = (f"  (floor {floor['mean']:.4f}, final is "
                        f"{last['mean'] - floor['mean']:+.4f} over it)")
            print(f"  {kind:>7}: mean cos {first['mean']:.4f} (iter {first['iteration']}) "
                  f"-> {last['mean']:.4f} (iter {last['iteration']})  [{arrow}]{note}")
    print(f"summary: {scored} pair(s) scored, {failures} failed")
    print(f"plot   : {png_path}")
    print(f"table  : {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
