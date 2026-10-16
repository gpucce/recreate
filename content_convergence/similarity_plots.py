"""Heatmap PDFs for the tables produced by similarity_matrices.py and
similarity_across_runs.py (cross matrices as heatmaps, paired curves as line plots).

Writes <name>.pdf next to every <name>.npy/.csv (per experiment and averages),
colour scale from red (lowest value in the table) to green (highest).
Existing PDFs are kept unless --recompute is given.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

EXP_DIR = "experiments"
NAMES = ("images_similarity", "prompts_similarity",
         "images_cross_similarity", "prompts_cross_similarity",
         "images_paired", "prompts_paired")


def _ticks(n):
    return range(0, n, 1 if n <= 20 else 2)


def plot_table(out: Path, name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.load(out / f"{name}.npy")
    if mat.ndim == 1:  # paired curve: similarity of A step t vs B step t
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(len(mat)), mat, marker="o")
        ax.set_xlabel("step")
        ax.set_ylabel("cosine similarity")
        ax.set_title(f"{out.name}: {name}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
        return

    with open(out / f"{name}.csv") as f:
        cols = next(csv.reader(f))[1:]
    with open(out / f"{name}.csv") as f:
        rows = [r[0] for r in list(csv.reader(f))[1:]]
    nr, nc = mat.shape
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=float(mat.min()), vmax=float(mat.max()))
    xt, yt = _ticks(nc), _ticks(nr)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([cols[i] for i in xt], rotation=90, fontsize=6)
    ax.set_yticklabels([rows[i] for i in yt], fontsize=6)
    if "cross" in name:
        ax.set_xlabel("run B")
        ax.set_ylabel("run A")
    if max(nr, nc) <= 20:
        for i in range(nr):
            for j in range(nc):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="k")
    ax.set_title(f"{out.name}: {name}")
    fig.colorbar(im, ax=ax, label="cosine similarity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def plot_all(out_root: Path, recompute: bool):
    exps = sorted(p.name for p in (out_root / EXP_DIR).glob("image_*") if p.is_dir())
    targets = [(out_root / EXP_DIR / e, n) for e in exps for n in NAMES]
    targets += [(out_root, f"average_{n}") for n in NAMES]
    for d, n in targets:
        if (d / f"{n}.npy").exists() and (recompute or not (d / f"{n}.pdf").exists()):
            print(f"Plotting {d}/{n}")
            plot_table(d, n)
        else:
            print(f"Skipping plot of {d}/{n}")
    print("plots done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="./data/similarities/vlstereoset_runs",
                    help="directory produced by similarity_matrices.py or similarity_across_runs.py")
    ap.add_argument("--recompute", action="store_true",
                    help="regenerate PDFs that already exist")
    a = ap.parse_args()
    plot_all(Path(a.sim), a.recompute)


if __name__ == "__main__":
    main()
