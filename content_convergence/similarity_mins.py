"""Where does the minimum consecutive similarity occur in each sequence?

For every experiment and every table (images, prompts) in --sim, finds the pair of
consecutive elements (e.g. image_3 -> image_4) with the lowest similarity, then
plots across all experiments how many minima have occurred up to each step.

Outputs in <sim>/mins/:
    <table>_mins.csv / .pdf             minimum over all consecutive pairs
    <table>_mins_excl_first.csv / .pdf same, ignoring the first transition (human -> generated)
CSV: one row per experiment (experiment, step, from, to, similarity); PDF: histogram of
minimum positions + cumulative fraction curve.
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


def find_mins(sim: Path, table: str, skip_first: bool = False):
    rows = []
    for d in sorted(p for p in (sim / EXP_DIR).glob("image_*") if p.is_dir()):
        npy = d / f"{table}.npy"
        if not npy.exists():
            continue
        m = np.load(npy)
        first = 1 if skip_first else 0
        if m.shape[0] < 2 + first:
            continue
        labels = labels_of(d / f"{table}.csv")
        consecutive = np.diag(m, k=1)
        i = first + int(np.argmin(consecutive[first:]))
        step = int(labels[i].rsplit("_", 1)[1])
        rows.append((d.name, step, labels[i], labels[i + 1], float(consecutive[i])))
    return rows


def plot(rows, table: str, out: Path, suffix: str, note: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = min(r[1] for r in rows), max(r[1] for r in rows)
    x = np.arange(lo, hi + 1)
    counts = np.bincount([r[1] - lo for r in rows], minlength=len(x))
    cum = np.cumsum(counts)
    ticks = [f"{i}→{i + 1}" for i in x]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, counts, color="lightsteelblue", label="minima at step")
    ax.set_xlabel("step (consecutive pair)")
    ax.set_ylabel("number of experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=90, fontsize=7)
    ax2 = ax.twinx()
    ax2.step(x, cum / len(rows), where="mid", color="crimson", marker="o",
             label="cumulative fraction")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("cumulative fraction of minima")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.set_title(f"{table}: position of minimum consecutive similarity "
                 f"({len(rows)} experiments{note})")
    fig.tight_layout()
    fig.savefig(out / f"{table}_{suffix}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="./data/similarities/vlstereoset_runs",
                    help="directory produced by similarity_matrices.py")
    a = ap.parse_args()
    sim = Path(a.sim)
    out = sim / "mins"
    out.mkdir(parents=True, exist_ok=True)

    variants = (("mins", False, ""),
                ("mins_excl_first", True, ", first transition excluded"))
    for table in TABLES:
        for suffix, skip_first, note in variants:
            rows = find_mins(sim, table, skip_first)
            if not rows:
                print(f"{table} ({suffix}): no tables found")
                continue
            with open(out / f"{table}_{suffix}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["experiment", "step", "from", "to", "similarity"])
                for r in rows:
                    w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6f}"])
            plot(rows, table, out, suffix, note)
            print(f"{table} ({suffix}): {len(rows)} experiments -> {out}")


if __name__ == "__main__":
    main()
