"""Heatmap PDFs for the tables produced by similarity_matrices.py.

Writes <name>.pdf next to every <name>.npy/.csv (per experiment and averages),
colour scale from red (lowest value in the table) to green (highest).
Existing PDFs are kept unless --recompute is given.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

EXP_DIR = "experiments"
NAMES = ("images_similarity", "prompts_similarity")


def plot_table(out: Path, name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.load(out / f"{name}.npy")
    with open(out / f"{name}.csv") as f:
        labels = next(csv.reader(f))[1:]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    vmin, vmax = float(mat.min()), float(mat.max())
    im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    step = 1 if n <= 20 else 2
    ticks = range(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=6)
    if n <= 20:
        for i in range(n):
            for j in range(n):
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
                    help="directory produced by similarity_matrices.py")
    ap.add_argument("--recompute", action="store_true",
                    help="regenerate PDFs that already exist")
    a = ap.parse_args()
    plot_all(Path(a.sim), a.recompute)


if __name__ == "__main__":
    main()
