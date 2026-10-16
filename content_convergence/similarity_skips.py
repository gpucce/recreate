"""Skips: consecutive transitions that are markedly less similar than usual.

For every experiment and every table (images, prompts) in --sim, a transition
(e.g. image_3 -> image_4) is a skip if its similarity is more than --threshold
(default 5%) below the mean similarity of the transitions of that experiment, i.e.
    sim < mean * (1 - threshold)
An experiment can contribute zero, one or more skips. Two variants are produced:
all transitions, and excluding the first one (human -> generated content), which is
then also left out of the mean.

Outputs in <sim>/skips/:
    <table>_skips_thr<threshold>[_excl_first].csv  one row per skip: experiment, step, from, to,
                                    similarity, experiment_mean
    <table>_skips_thr<threshold>[_excl_first].pdf  skips per step + cumulative fraction of all skips
"""
import argparse
import csv
from pathlib import Path

import numpy as np

EXP_DIR = "experiments"
TABLES = ("images_similarity", "prompts_similarity")


def labels_of(csv_path: Path):
    with open(csv_path) as f:
        return next(csv.reader(f))[1:]


def find_skips(sim: Path, table: str, threshold: float, skip_first: bool):
    """Returns (skip rows, number of experiments considered)."""
    rows, n_exp = [], 0
    first = 1 if skip_first else 0
    for d in sorted(p for p in (sim / EXP_DIR).glob("image_*") if p.is_dir()):
        npy = d / f"{table}.npy"
        if not npy.exists():
            continue
        m = np.load(npy)
        consecutive = np.diag(m, k=1)[first:]
        if len(consecutive) == 0:
            continue
        n_exp += 1
        labels = labels_of(d / f"{table}.csv")
        mean = float(consecutive.mean())
        for j in np.flatnonzero(consecutive < mean * (1 - threshold)):
            i = first + int(j)
            step = int(labels[i].rsplit("_", 1)[1])
            rows.append((d.name, step, labels[i], labels[i + 1],
                         float(consecutive[j]), mean))
    return rows, n_exp


def plot(rows, n_exp, table: str, out: Path, suffix: str, note: str, threshold: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = min(r[1] for r in rows), max(r[1] for r in rows)
    x = np.arange(lo, hi + 1)
    counts = np.bincount([r[1] - lo for r in rows], minlength=len(x))
    cum = np.cumsum(counts) / counts.sum()
    ticks = [f"{i}→{i + 1}" for i in x]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, counts, color="lightsteelblue", label="skips at step")
    ax.set_xlabel("step (consecutive pair)")
    ax.set_ylabel("number of skips")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=90, fontsize=7)
    ax2 = ax.twinx()
    ax2.step(x, cum, where="mid", color="crimson", marker="o",
             label="cumulative fraction")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("cumulative fraction of skips")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    with_skips = len({r[0] for r in rows})
    ax.set_title(f"{table}: skips (>{threshold:.0%} below experiment mean){note}\n"
                 f"{len(rows)} skips in {with_skips}/{n_exp} experiments", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{table}_{suffix}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="./data/similarities/vlstereoset_runs",
                    help="directory produced by similarity_matrices.py")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="relative drop below the experiment's mean transition "
                         "similarity that counts as a skip (default: 0.05 = 5%%)")
    a = ap.parse_args()
    sim = Path(a.sim)
    out = sim / "skips"
    out.mkdir(parents=True, exist_ok=True)

    tag = f"skips_thr{a.threshold:g}"
    variants = ((tag, False, ""),
                (f"{tag}_excl_first", True, ", first transition excluded"))
    for table in TABLES:
        for suffix, skip_first, note in variants:
            rows, n_exp = find_skips(sim, table, a.threshold, skip_first)
            if not n_exp:
                print(f"{table} ({suffix}): no tables found")
                continue
            with open(out / f"{table}_{suffix}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["experiment", "step", "from", "to", "similarity",
                            "experiment_mean"])
                for r in rows:
                    w.writerow([*r[:4], f"{r[4]:.6f}", f"{r[5]:.6f}"])
            print(f"{table} ({suffix}): {len(rows)} skips in "
                  f"{len({r[0] for r in rows})}/{n_exp} experiments")
            if rows:
                plot(rows, n_exp, table, out, suffix, note, a.threshold)


if __name__ == "__main__":
    main()
