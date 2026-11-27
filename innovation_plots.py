"""Where do the entities of each step come from? Splits the per-document innovation rate into a local and a global part.

    python innovation_plots.py                                 # bbc-news, erc-grants, acl with qwen3.8-27b
    python innovation_plots.py --datasets acl --format png

For document d at step k, every entity of the step is assigned to exactly one source:
    self-recycled: already in d's own chain at an earlier step
    borrowed:      not in d's chain, but produced by some other document at an earlier step
    new:           never produced by any document at an earlier step
Shares are computed per document and then averaged over documents, so borrowed + new = the per-document innovation
rate of plot_entities.py (per_document/) and self-recycled = 1 - that rate. A growing borrowed share means the chains
drift toward a vocabulary shared across the corpus (global effect); a growing self-recycled share means each chain
repeats itself (local effect).

Output: <out-dir>/<dataset>/all_types_{odd,even}.<fmt>, <out-dir>/<dataset>/by_type/<type>_{odd,even}.<fmt>,
<out-dir>/legend.<fmt> and <out-dir>/<dataset>/innovation_sources.csv
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_entities import (ALL_TYPES, DATA_ROOT, FORMAT, STEP_PANELS, format_step_axis, get_iterations,
                           legend_figure, load_dataset, save, slug, split_steps, step_figures, top_entity_types)

OUT_DIR = "/raid/homes/giovanni.puccetti/Latex/acl_content_convergence/plots/textonly/innovation_only"

# stacked bottom to top: the local part first, then the two parts that make up the innovation rate
SOURCES = {"self": "Self-recycled", "borrowed": "Borrowed from other documents", "new": "New to the corpus"}
SOURCE_COLORS = {"self": "C0", "borrowed": "C1", "new": "C2"}


def innovation_sources(data):
    """Per (document, type, iteration): the share of the step's entities in each source. ALL_TYPES pools the
    entity strings of every type (as in plot_entities). Documents with no entity of a type at a step are skipped."""
    iterations = get_iterations(data)
    by_type = {etype: g.groupby(["iteration", "document_ids"])["entity"].agg(set) for etype, g in data.groupby("type")}
    by_type[ALL_TYPES] = data.groupby(["iteration", "document_ids"])["entity"].agg(set)

    rows = []
    for etype, sets in by_type.items():
        own_seen = {}          # document -> entities of its earlier steps
        corpus_seen = set()    # entities of every document at earlier steps
        for it in iterations:
            step = sets.get(it)
            if step is None:
                continue
            for doc, ents in step.items():
                own = own_seen.get(doc, set())
                n_self = len(ents & own)
                n_borrowed = len((ents - own) & corpus_seen)
                n = len(ents)
                rows.append((doc, etype, it, n_self / n, n_borrowed / n, (n - n_self - n_borrowed) / n, n))
            # update only after the whole step, so a step is never compared with itself or with its own other documents
            for doc, ents in step.items():
                own_seen.setdefault(doc, set()).update(ents)
                corpus_seen |= ents
    return pd.DataFrame(rows, columns=["document_ids", "type", "iteration", *SOURCES, "n_entities"])


def plot_sources(means, iterations, title=None, figsize=(8, 6)):
    """Stacked areas of the mean source shares (rows = iterations, columns = SOURCES); odd and even steps apart."""
    steps = split_steps(iterations)
    figs = step_figures(figsize)
    for (step, _, marker), (_, ax) in zip(STEP_PANELS, figs.values()):
        xs = steps[step]
        ys = means.reindex([iterations[x] for x in xs])
        ax.stackplot(xs, *(ys[s].to_numpy() for s in SOURCES), colors=[SOURCE_COLORS[s] for s in SOURCES],
                     alpha=0.8, edgecolor="black", linewidth=0.5)
        # dashed line = per-document innovation rate (borrowed + new)
        ax.plot(xs, (ys["borrowed"] + ys["new"]).to_numpy(), color="black", linestyle="--", marker=marker,
                markeredgewidth=0.5, linewidth=1.5)
        format_step_axis(ax, xs, iterations)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of the step's entities\n(mean per document)")
        if title:
            ax.set_title(title)
    return {step: fig for step, (fig, _) in figs.items()}


def sources_legend():
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SOURCE_COLORS[s], alpha=0.8, edgecolor="black", linewidth=0.5)
               for s in SOURCES]
    handles.append(plt.Line2D([], [], color="black", linestyle="--", linewidth=1.5))
    return legend_figure(handles, [*SOURCES.values(), "Innovation rate (borrowed + new)"], ncol=2)


def run(name, data_type, data_entity, out: Path, fmt: str, top_types: int | None = None):
    iterations = get_iterations(data_entity)
    shares = innovation_sources(data_entity)
    means = shares.groupby(["type", "iteration"])[list(SOURCES)].mean()
    (out / name).mkdir(parents=True, exist_ok=True)
    means.round(4).to_csv(out / name / "innovation_sources.csv")

    save(plot_sources(means.loc[ALL_TYPES], iterations), out / name / f"all_types.{fmt}")
    types = top_entity_types(data_type, top_types) if top_types else sorted(data_entity["type"].unique())
    for etype in types:
        if etype in means.index.get_level_values("type"):
            save(plot_sources(means.loc[etype], iterations, title=etype), out / name / "by_type" / f"{slug(etype)}.{fmt}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["bbc-news", "erc-grants", "acl"])
    parser.add_argument("--model", default="qwen_qwen3.8_27b", help="Generator folder name ('-' and '_' are interchangeable).")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path(OUT_DIR))
    parser.add_argument("--format", default=FORMAT, help="Figure format, e.g. pdf or png.")
    parser.add_argument("--top-types", type=int, default=None, help="Only the N most frequent entity types per dataset.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save({None: sources_legend()}, args.out_dir / f"legend.{args.format}")
    for name in args.datasets:
        data_type, data_entity = load_dataset(name, args.model, args.data_root)
        run(name, data_type, data_entity, args.out_dir, args.format, args.top_types)


if __name__ == "__main__":
    main()
