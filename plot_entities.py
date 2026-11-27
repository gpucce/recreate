"""Entity plots from _plotter.ipynb as a script: saves every figure (and the summary tables as CSV) to the paper's plots dir.

    python plot_entities.py                                   # bbc-news, erc-grants, acl with qwen3.8-27b
    python plot_entities.py --datasets acl --format png
    python plot_entities.py --bertscore                       # also the ERC entity-BERTScore plots (GPU, downloads models)

For every dataset: entity-type counts per iteration, one sample's entity heatmap, retention / novelty / innovation rate
(all types together and one plot per type). For erc-grants only: entities by call year, iteration plots split by
call year, real vs simulated abstracts and entity overlap (and entity BERTScore with --bertscore).
Output: <out-dir>/<dataset>/... and <out-dir>/erc-grants/by_year/...
"""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# larger text in every figure (matplotlib default is 10)
plt.rcParams.update({
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 18,     # only the overlap column headers still have titles
    "legend.fontsize": 16,
})
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import kruskal, mannwhitneyu

DATA_ROOT = Path(__file__).resolve().parent / "data_bkp/da_mettere_in_homeprogetto/outputs/textonly_dataframes"
ERC_CSV = "/raid/datasets/content_convergence/original_datasets/ERC_data_filtered.csv"
OUT_DIR = "/raid/homes/giovanni.puccetti/Latex/acl_content_convergence/plots/textonly"
FORMAT = "pdf"
# datasets whose original column is ignored (ACL: the paper, turned once into the first research plan)
DROP_ORIGINAL = {"acl"}
# per dataset: entity types whose count exceeds this at ANY iteration are dropped entirely from the type plots
# (a per-row filter would cut single points and make a line drop to 0); None keeps every type
TYPE_COUNT_MAX = {
    "bbc-news": None,
    "erc-grants": None,
    "acl": None,
}


# per dataset: misspelled entity types emitted by the extractor, renamed (and merged) into the right type before filtering
TYPE_ALIASES = {
    "acl": {"novelry_claim": "novelty_claim"},
}

# per dataset: the only entity types used by every plot (filtered at load time); a dataset not listed keeps all types
ENTITY_TYPES = {
    "acl": [
        "task","topic","language","model","method","paradigm","dataset","metric","eval_protocol","baseline","domain",
        "resource","data_construction","linguistic_phenomenon","population","software","hardware","statistical_analysis",
        "modality","limitation_or_risk","novelty_claim",],
    "bbc-news": [
        "person", "organization", "geopolitical_entity", "location", "facility",
        "time", "currency", "event", "product", "work_of_art", "law",],
    "erc-grants": [
        "system", "theory_model", "method_experimental", "method_computational", "instrument",
        "quantity_regime", "open_problem", "application", "field", "stakeholder",],
}

# ----------------------------------------------------------------------------------------------------------------- io

def model_dir(dataset: str, model: str, root=DATA_ROOT) -> Path:
    """Folder of one dataset/model; '-' and '_' are interchangeable (ACL uses qwen_qwen3.8-27b, the others qwen_qwen3.8_27b)."""
    norm = lambda s: s.replace("-", "_")
    matches = [d for d in (Path(root) / dataset).iterdir() if d.is_dir() and norm(d.name) == norm(model)]
    if not matches:
        raise FileNotFoundError(f"no {model} folder for {dataset} in {root}")
    return matches[0]


def merge_rows(df, keys):
    """Collapse rows that share `keys` (e.g. after renaming a type): counts are summed, list-valued document_ids
    are united. Iteration order of the groups is kept."""
    agg = {"count": "sum"}
    if "document_ids" in df.columns and "document_ids" not in keys:
        agg["document_ids"] = lambda ids: sorted({i for lst in ids for i in lst})
    return df.groupby(keys, sort=False, as_index=False).agg(agg)


def load_dataset(dataset: str, model: str = "qwen_qwen3.8_27b", root=DATA_ROOT):
    """Return (data_type, data_entity) count dataframes for one dataset/model."""
    base = model_dir(dataset, model, root)
    df_type, df_entity = pd.read_json(base / "counts_type.json"), pd.read_json(base / "counts_entity.json")
    if dataset in DROP_ORIGINAL:
        # the original is converted only once into the first generation: the chain starts at iteration_00
        df_type = df_type[df_type["iteration"] != "original"]
        df_entity = df_entity[df_entity["iteration"] != "original"]
    if dataset in TYPE_ALIASES:
        aliases = TYPE_ALIASES[dataset]
        df_type = merge_rows(df_type.assign(entity=df_type["entity"].replace(aliases)), ["iteration", "entity"])
        df_entity = merge_rows(df_entity.assign(type=df_entity["type"].replace(aliases)),
                               ["iteration", "entity", "type", "document_ids"])
    if dataset in ENTITY_TYPES:
        # counts_type stores the type in "entity", counts_entity in "type"
        keep = ENTITY_TYPES[dataset]
        df_type = df_type[df_type["entity"].isin(keep)]
        df_entity = df_entity[df_entity["type"].isin(keep)]
        missing = sorted(set(keep) - set(df_type["entity"]))
        if missing:
            print(dataset, "listed entity types not found in the data:", missing)
    print(dataset, "type", df_type.shape[0])
    type_max = TYPE_COUNT_MAX.get(dataset)
    if type_max is not None:
        too_large = df_type.groupby("entity")["count"].max().loc[lambda m: m > type_max].index
        df_type = df_type[~df_type["entity"].isin(too_large)]
        print(dataset, f"types with count > {type_max} dropped:", sorted(too_large))
    print(dataset, "type", df_type.shape[0])
    entity_mask = df_entity["count"] <= 50
    print(dataset, "entity", df_entity.shape[0])
    df_entity = df_entity.loc[entity_mask, :]
    print(dataset, "entity", df_entity.shape[0])
    return df_type, df_entity


def save(figs, path: Path):
    """Save what a plot function returns and close the figures: a (fig, axes) pair is saved to `path`,
    a {part: fig} dict (e.g. odd / even / legend) to <stem>_<part><suffix> next to it."""
    if figs is None:
        return
    parts = figs if isinstance(figs, dict) else {None: figs[0]}
    path.parent.mkdir(parents=True, exist_ok=True)
    for part, fig in parts.items():
        out = path if part is None else path.with_name(f"{path.stem}_{part}{path.suffix}")
        # figures with a layout engine already fit their labels: save them at exactly their figsize, so every
        # plot of a kind has the same size; legend-only figures are cropped to their content
        tight = (part == "legend" and not getattr(fig, "keep_size", False)) or fig.get_layout_engine() is None
        fig.savefig(out, bbox_inches="tight" if tight else None, dpi=200)
        plt.close(fig)
        print("saved", out)


def legend_figure(handles, labels, ncol=1, figsize=None, fontsize=None):
    """A figure holding only the legend, to be placed separately in the paper. By default it is cropped to the
    legend; with figsize it keeps that size (legend centred), so legends of the same figsize scale alike in LaTeX."""
    fig = plt.figure(figsize=figsize or (1, 1))
    # fixed-size legends use tighter spacing, which leaves room for a larger font
    spacing = dict(columnspacing=1.0, handlelength=1.6, handletextpad=0.5) if figsize else {}
    leg = fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False, fontsize=fontsize, **spacing)
    if figsize:
        # keep the page size; if the legend is too wide or tall for it, shrink its font in 0.5 pt steps
        fig.keep_size = True
        size = leg.get_texts()[0].get_fontsize() if labels else 0
        while True:
            box = leg.get_window_extent(fig.canvas.get_renderer())
            if (box.width <= 0.98 * fig.bbox.width and box.height <= fig.bbox.height) or size <= 6:
                break
            leg.remove()
            size -= 0.5
            leg = fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False, fontsize=size, **spacing)
        print(f"legend font {size:g} pt: {', '.join(labels)}")
    return fig


def slug(s: str) -> str:
    return str(s).replace(" ", "_").replace("/", "-")


# ------------------------------------------------------------------------------------------------ iteration plots

# helpers to split the two-step process into separate panels
# position 0 = original, position k = iteration_{k-1:02d}
#   odd positions  (1, 3, 5, ...) -> first generation prompt  (iteration_00, _02, ...)
#   even positions (0, 2, 4, ...) -> second generation prompt (original, iteration_01, _03, ...)
# without an original (ACL), position k = iteration_{k:02d}:
#   even positions (0, 2, 4, ...) -> research plans (iteration_00, _02, ...), odd positions -> reviews
STEP_PANELS = (
    ("odd", "Odd steps", "s"),
    ("even", "Even steps", "o"),
)


def get_iterations(data):
    """Chain in order: original (if present) then iteration_00, iteration_01, ..."""
    present = set(data.iteration.unique())
    return (["original"] if "original" in present else []) + sorted(c for c in present if c.startswith("iteration_"))


def split_steps(iterations):
    positions = list(range(len(iterations)))
    return {"odd": positions[1::2], "even": positions[0::2]}


def tick_labels(positions, iterations):
    # plain step numbers: no "original" label, so the axes have the same size with and without an original text
    return [str(p) for p in positions]


def format_step_axis(ax, positions, iterations):
    ax.set_xticks(positions, tick_labels(positions, iterations))
    if positions and iterations[positions[0]] == "original":
        # the grey band marks the original text (position 0)
        ax.axvspan(-1, 1, alpha=0.7, color="lightgray")
    ax.set_xlim(positions[0] - 1, positions[-1] + 1)
    ax.set_xlabel("Iterations")
    ax.grid(which="major", linestyle="--", alpha=0.7)


def step_figures(figsize):
    """One figure per step type: {step: (fig, ax)} in STEP_PANELS order."""
    return {step: plt.subplots(figsize=figsize, layout="constrained") for step, _, _ in STEP_PANELS}


def share_ylim(axes):
    """Same y range on every axis, so odd and even figures stay comparable once separated."""
    lims = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))


# plot enity TYPES at different iterations: odd steps, even steps and the legend each in its own figure
def plot_entity_types(data, entity_types: list | None = None, figsize=(8, 6), ylabel='Count',
                      color_order: list | None = None):
    line_style = {"linewidth": 2, "markeredgecolor": "black", "markeredgewidth": 0.5}

    iterations = get_iterations(data)
    steps = split_steps(iterations)
    figs = step_figures(figsize)
    axes = [ax for _, ax in figs.values()]

    entity_types = entity_types if entity_types else sorted(data['entity'].unique())
    # tab20 (dark shades first, then light) so >10 types never repeat a color
    colors = [plt.cm.tab20(i) for i in list(range(0, 20, 2)) + list(range(1, 20, 2))]
    # color follows the type name, so it is identical across figures that share color_order
    color_of = {t: colors[i % len(colors)] for i, t in enumerate(color_order or entity_types)}
    for et in entity_types:
        counts = [
            data[(data["iteration"] == it) & (data["entity"] == et)]["count"].sum()
            for it in iterations
        ]

        color = color_of[et]  # same color for the entity type in both panels
        for ax, (step, _, marker) in zip(axes, STEP_PANELS):
            xs = steps[step]
            ax.plot(xs, [counts[x] for x in xs], label=et, marker=marker, color=color, **line_style)

    for ax, (step, _, _) in zip(axes, STEP_PANELS):
        format_step_axis(ax, steps[step], iterations)
        ax.set_ylabel(ylabel)
    share_ylim(axes)

    handles, labels = axes[0].get_legend_handles_labels()
    out = {step: fig for step, (fig, _) in figs.items()}
    out["legend"] = legend_figure(handles, labels, ncol=max(1, len(entity_types) // 2))
    return out


def plot_sample_trajectory(data, sample_id, figsize=None):
    sample = data[data["document_ids"] == sample_id]
    if sample.empty:
        print(f"No sample found with document ID {sample_id}.")
        return

    iterations = get_iterations(data)
    steps = split_steps(iterations)
    vocab = sample.entity.unique()

    # create a heatmap of the sample's entity counts across iterations
    heatmap_data = (
        sample.groupby(["entity", "iteration"])["count"].sum()
        .unstack(fill_value=0)
        .reindex(index=vocab, columns=iterations, fill_value=0)
        .astype(float)
    )
    vmin, vmax = heatmap_data.values.min(), heatmap_data.values.max()

    # one heatmap figure per step type, sharing the color scale; the colorbar is its own figure
    # room for the entity labels on the left (~0.13 in per character at 16pt) and ~0.3 in per heatmap row
    longest = max(len(str(v)) for v in vocab)
    n_cols = max(len(p) for p in steps.values())
    figsize = figsize or (0.13 * longest + 0.4 * n_cols + 1.5, max(6, 0.3 * len(vocab) + 2))
    figs = step_figures(figsize)
    for (step, _, _), (_, ax) in zip(STEP_PANELS, figs.values()):
        positions = steps[step]
        im = ax.imshow(heatmap_data.iloc[:, positions], cmap='viridis', aspect="equal", vmin=vmin, vmax=vmax)

        ax.set_yticks(range(len(vocab)), vocab)
        ax.set_xticks(range(len(positions)), tick_labels(positions, iterations), rotation=90)
        ax.set_xlabel("Iterations")

        # cell grid
        ax.set_yticks(np.arange(-0.5, len(vocab), 1), minor=True)
        ax.set_xticks(np.arange(-0.5, len(positions), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="minor", length=0)

    out = {step: fig for step, (fig, _) in figs.items()}
    out["legend"] = plt.figure(figsize=(1, 6))
    mappable = matplotlib.cm.ScalarMappable(norm=im.norm, cmap=im.cmap)   # same scale as the heatmaps
    out["legend"].colorbar(mappable, cax=out["legend"].add_axes([0.1, 0.05, 0.25, 0.9]), label="Count")
    return out


def compare_counters(src: Counter, tgt: Counter, seen: set | None = None):
    """
    src: original counter; tgt: counter of the step being compared.
    seen: union of all entities from earlier steps (original included).
          Empty/None for the original step itself.
    """
    seen = seen or set()

    ents = sorted(set(src) | set(tgt))
    p = np.array([src.get(e, 0) for e in ents], dtype=float)
    q = np.array([tgt.get(e, 0) for e in ents], dtype=float)

    S, T = set(src), set(tgt)
    div = lambda a, b: a / b if b else np.nan   # empty sets (e.g. a rare entity type) -> NaN, not a crash

    return {
        "jaccard_dist": 1 - div(len(S & T), len(S | T)),
        "js_divergence": jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2 if p.sum() and q.sum() else np.nan,
        "retention": div(len(S & T), len(S)),
        "retained_mass": div(sum(src[e] for e in S & T), sum(src.values())),
        "novelty": div(len(T - S), len(T)),
        "innovation_rate": div(len(T - seen), len(T)),   # entities never seen in any earlier step
        "cum_vocab": len(seen | T),
    }


def plot_retention_novelty(data, entity_type: None | str = None):
    iterations = get_iterations(data)

    if entity_type:
        data = data[data["type"] == entity_type]

    # reference = first text of the chain (original, or iteration_00 when the original is ignored)
    src = data[data["iteration"] == iterations[0]]["entity"].value_counts().to_dict()

    # innovation rate at step k is computed against the union of all steps n < k
    metrics = []
    seen = set()
    for iteration in iterations:
        tgt = data[data["iteration"] == iteration]["entity"].value_counts().to_dict()
        metrics.append(compare_counters(src, tgt, seen))
        seen |= set(tgt)          # update only after comparing, so step k is never compared to itself

    steps = split_steps(iterations)
    figs = step_figures((8, 6))

    series = (
        ("retention", "Retention", {}),
        ("novelty", "Novelty", {}),
        ("innovation_rate", "Innovation rate", {"linestyle": "--"}),
    )
    for (step, _, marker), (_, ax) in zip(STEP_PANELS, figs.values()):
        xs = steps[step]
        for c_idx, (key, label, line_kwargs) in enumerate(series):
            ax.plot(
                xs, [metrics[x][key] for x in xs], label=label, color=f"C{c_idx}",
                marker=marker, markeredgecolor="black", markeredgewidth=0.5, **line_kwargs,
            )
        format_step_axis(ax, xs, iterations)
        ax.set_ylim(-0.02, 1.02)

    # the legend uses plain lines (markers differ between odd and even figures)
    handles = [plt.Line2D([], [], color=f"C{c_idx}", linewidth=2, **line_kwargs)
               for c_idx, (_, _, line_kwargs) in enumerate(series)]
    out = {step: fig for step, (fig, _) in figs.items()}
    out["legend"] = legend_figure(handles, [label for _, label, _ in series], ncol=len(series))
    return out


PER_DOC_METRICS = {"retention": "Retention", "novelty": "Novelty", "innovation_rate": "Innovation rate"}
ALL_TYPES = "all types"


def per_document_metrics(data):
    """Retention / novelty / innovation rate of every document's own chain, per entity type and for all types together.

    Unlike plot_retention_novelty (one pooled vocabulary per iteration), each document is compared only with itself:
    step k vs the document's first step, and innovation vs the union of the document's own earlier steps.
    Returns one row per (document, type, iteration); a metric with an empty denominator is NaN (skipped by the mean).
    """
    iterations = get_iterations(data)
    div = lambda a, b: a / b if b else np.nan
    chains = {}   # (document, type) -> {iteration: set of entities}
    for (doc, etype, it), ents in data.groupby(["document_ids", "type", "iteration"])["entity"].agg(set).items():
        chains.setdefault((doc, etype), {})[it] = ents
    for (doc, it), ents in data.groupby(["document_ids", "iteration"])["entity"].agg(set).items():
        chains.setdefault((doc, ALL_TYPES), {})[it] = ents

    rows = []
    for (doc, etype), sets in chains.items():
        src, seen = sets.get(iterations[0], set()), set()
        for it in iterations:
            tgt = sets.get(it, set())
            rows.append((doc, etype, it, div(len(src & tgt), len(src)), div(len(tgt - src), len(tgt)),
                         div(len(tgt - seen), len(tgt))))
            seen |= tgt
    return pd.DataFrame(rows, columns=["document_ids", "type", "iteration", *PER_DOC_METRICS])


def pooled_metrics(data):
    """Corpus-level retention / novelty / innovation rate (as in plot_retention_novelty), per entity type and for all
    types together: every iteration is one vocabulary pooled over all documents. One row per (type, iteration)."""
    iterations = get_iterations(data)
    groups = [(etype, g) for etype, g in data.groupby("type")] + [(ALL_TYPES, data)]
    rows = []
    for etype, g in groups:
        vocab = g.groupby("iteration")["entity"].agg(set)
        src, seen = vocab.get(iterations[0], set()), set()
        for it in iterations:
            tgt = vocab.get(it, set())
            m = compare_counters(Counter(src), Counter(tgt), seen)
            rows.append((etype, it, *(m[k] for k in PER_DOC_METRICS)))
            seen |= tgt
    return pd.DataFrame(rows, columns=["type", "iteration", *PER_DOC_METRICS])


AVERAGE = "avg. of all types"
# legends of plot_metric_by_type: as wide as the plot and of fixed height, so they scale like the plot in LaTeX
METRIC_LEGEND_HEIGHT = 2.1
METRIC_LEGEND_FONTSIZE = 22


def plot_metric_by_type(per_type, metric, ylabel_suffix="", entity_types: list | None = None, figsize=(8, 6),
                        legend_ncol: int = 2):
    """One metric over iterations for six entity types, the top two, middle two and bottom two by the metric's mean
    over all iterations, plus a black dashed line for the unweighted average of all entity types' lines.
    per_type: rows of (type, iteration, metric), averaged per (type, iteration) (a no-op for pooled_metrics).
    entity_types: the candidate types (default: all). Odd steps, even steps and the legend each in its own figure;
    the same six types are shown in both step figures."""
    iterations = get_iterations(per_type)
    steps = split_steps(iterations)
    means = per_type.groupby(["type", "iteration"])[metric].mean()
    entity_types = entity_types or sorted(t for t in per_type["type"].unique() if t != ALL_TYPES)
    table = means.unstack("iteration").reindex(index=entity_types, columns=iterations)   # type x iteration
    ranked = table.mean(axis=1).dropna().sort_values(ascending=False).index.tolist()
    mid = max(0, (len(ranked) - 2) // 2)
    picked = list(dict.fromkeys(ranked[:2] + ranked[mid:mid + 2] + ranked[-2:]))      # top 2, middle 2, bottom 2
    average = table.mean(axis=0)                                                          # iteration -> mean
    roles = ["tab:blue", "tab:cyan", "tab:orange", "tab:olive", "tab:green", "tab:purple"]  # top, middle, bottom
    styles = {t: {"color": c, "linewidth": 2, "linestyle": "-"} for t, c in zip(picked, roles)}
    styles[AVERAGE] = {"color": "black", "linewidth": 3, "linestyle": "--"}

    figs = step_figures(figsize)
    for (step, _, marker), (_, ax) in zip(STEP_PANELS, figs.values()):
        xs = steps[step]
        for etype, style in styles.items():
            row = average if etype == AVERAGE else table.loc[etype]
            ax.plot(xs, [row[iterations[x]] for x in xs], label=etype, marker=marker, markeredgecolor="black",
                    markeredgewidth=0.5, **style)
        format_step_axis(ax, xs, iterations)
        ax.set_ylabel(f"{PER_DOC_METRICS[metric]}{ylabel_suffix}")
        ax.set_ylim(-0.02, 1.02)

    handles = [plt.Line2D([], [], **style) for style in styles.values()]
    out = {step: fig for step, (fig, _) in figs.items()}
    out["legend"] = legend_figure(handles, list(styles), ncol=legend_ncol, figsize=(figsize[0], METRIC_LEGEND_HEIGHT),
                                  fontsize=METRIC_LEGEND_FONTSIZE)
    return out


def top_entity_types(data_type, top_k: int | None = None):
    """Entity types sorted by total mentions over all iterations (most frequent first), the first top_k if given."""
    totals = data_type.groupby("entity")["count"].sum().sort_values(ascending=False)
    return totals.index[:top_k].tolist()


def show_entity_types(datasets: dict, entity_types: list | None = None, top_k: int | None = None, ylabel="Count",
                      shared_colors=True):
    """Entity-type counts per iteration for every dataset in {name: (data_type, data_entity)}.
    entity_types: explicit list of types; otherwise the top_k most frequent types of each dataset (all if None).
    shared_colors: a type gets the same color in every figure (for subsets of one dataset, e.g. call years);
    False colors each dataset on its own (different datasets have different types), so <= 20 types never repeat."""
    selected = {name: entity_types or top_entity_types(data_type, top_k) for name, (data_type, _) in datasets.items()}
    shared_order = sorted(set().union(*selected.values()))
    return {
        name: plot_entity_types(data_type, entity_types=selected[name], ylabel=ylabel,
                                color_order=shared_order if shared_colors else sorted(selected[name]))
        for name, (data_type, _) in datasets.items()
    }


# ----------------------------------------------------------------------------- ERC grants: entities by call year

def erc_entities_per_document(df, data_entity, iteration="original", year_col="Call Year"):
    """One row per ERC project: entity counts at `iteration` + call year from the original df.
    `document_ids` is the positional row of `df` (textonly_loop reads df["Abstract"] in order).
    Documents with no extracted entities are kept with 0."""
    ents = data_entity[data_entity["iteration"] == iteration]
    per_doc = pd.DataFrame({
        "n_mentions": ents.groupby("document_ids")["count"].sum(),   # total entity mentions
        "n_unique": ents.groupby("document_ids")["entity"].nunique(),  # distinct entities
    }).reindex(range(len(df)), fill_value=0)
    per_type = ents.pivot_table(index="document_ids", columns="type", values="count", aggfunc="sum", fill_value=0)
    per_type = per_type.reindex(range(len(df)), fill_value=0)

    per_doc["year"] = df[year_col].to_numpy()
    per_doc["n_words"] = df["Abstract"].str.split().str.len().to_numpy()
    per_doc["mentions_per_100_words"] = 100 * per_doc["n_mentions"] / per_doc["n_words"]
    return per_doc, per_type


def summarize_entities_by_year(per_doc, metric="n_mentions"):
    """Per-year stats for `metric`, difference of the mean vs the first year, and a Kruskal-Wallis test."""
    summary = per_doc.groupby("year")[metric].agg(["count", "mean", "median", "std"])
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["mean_diff_vs_first"] = summary["mean"] - summary["mean"].iloc[0]
    summary["pct_diff_vs_first"] = 100 * summary["mean_diff_vs_first"] / summary["mean"].iloc[0]
    h, pval = kruskal(*[g[metric].to_numpy() for _, g in per_doc.groupby("year")])
    print(f"{metric}: Kruskal-Wallis H = {h:.2f}, p = {pval:.3g}")
    return summary


def plot_entities_by_year(per_doc, per_type, metric="n_mentions", figsize=(16, 6)):
    """Left: distribution of `metric` per document for each call year (box + mean).
    Right: mean mentions per document for each entity type, by call year."""
    years = sorted(per_doc["year"].unique())
    labels = {"n_mentions": "Entity mentions per abstract",
              "n_unique": "Distinct entities per abstract",
              "mentions_per_100_words": "Entity mentions per 100 words"}

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize, layout="constrained")

    groups = [per_doc.loc[per_doc["year"] == y, metric] for y in years]
    ax_l.boxplot(groups, tick_labels=[f"{y}\n(n={len(g)})" for y, g in zip(years, groups)],
                 showfliers=True, flierprops={"markersize": 3, "alpha": 0.4},
                 medianprops={"color": "C0", "linewidth": 2})
    means = [g.mean() for g in groups]
    ax_l.plot(range(1, len(years) + 1), means, marker="o", color="C1", markeredgecolor="black",
              markeredgewidth=0.5, linewidth=2, label="Mean")
    for x, m in zip(range(1, len(years) + 1), means):
        ax_l.annotate(f"{m:.1f}", (x, m), textcoords="offset points", xytext=(10, 4))
    ax_l.set_xlabel("Call year")
    ax_l.set_ylabel(labels.get(metric, metric))
    ax_l.grid(axis="y", linestyle="--", alpha=0.7)
    ax_l.legend()

    type_means = per_type.groupby(per_doc["year"].to_numpy()).mean().T   # rows: type, cols: year
    type_means = type_means.loc[type_means.mean(axis=1).sort_values(ascending=False).index]
    # tab20 (dark shades first, then light) so >10 types never repeat a color
    colors = [plt.cm.tab20(i) for i in list(range(0, 20, 2)) + list(range(1, 20, 2))]
    for i, (et, row) in enumerate(type_means.iterrows()):
        ax_r.plot(years, row[years], marker="o", linewidth=2, color=colors[i % len(colors)],
                  markeredgecolor="black", markeredgewidth=0.5, label=et)
    ax_r.set_xticks(years)
    ax_r.set_xlabel("Call year")
    ax_r.set_ylabel("Mean mentions per abstract")
    ax_r.grid(linestyle="--", alpha=0.7)
    ax_r.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    return fig, (ax_l, ax_r)


# ------------------------------------------------------------------ ERC grants: iteration plots split by call year

def type_counts_from_entities(data_entity, n_docs: int | None = None):
    """counts_type-like frame (iteration, entity=<type>, count) from per-document entity counts.
    If n_docs is given, counts are divided by it (mean mentions per document)."""
    out = data_entity.groupby(["iteration", "type"], as_index=False)["count"].sum().rename(columns={"type": "entity"})
    if n_docs:
        out["count"] = out["count"] / n_docs
    return out


def split_by_year(df, data_entity, name="erc-grants", year_col="Call Year", per_document=True):
    """{f"{name} {year}": (data_type, data_entity)} restricted to documents whose original abstract is from `year`.
    document_ids are positional rows of df. per_document=True normalizes type counts by the number of
    documents in that year, so years with different sizes are comparable."""
    years = df[year_col].to_numpy()
    out = {}
    for year in sorted(np.unique(years)):
        ids = np.flatnonzero(years == year)
        ent = data_entity[data_entity["document_ids"].isin(ids)]
        out[f"{name} {year}"] = (type_counts_from_entities(ent, len(ids) if per_document else None), ent)
    return out


# -------------------------------------------------------- ERC grants: real abstracts by year vs. simulation from 2024

def per_doc_metrics(ent, doc_ids):
    """Per-document metrics for `doc_ids` (docs with no entities -> 0): total mentions, distinct entities, mentions per type."""
    g = ent.groupby("document_ids")
    out = pd.DataFrame({"n_mentions": g["count"].sum(), "n_unique": g["entity"].nunique()})
    out = out.join(ent.pivot_table(index="document_ids", columns="type", values="count", aggfunc="sum"))
    return out.reindex(doc_ids).fillna(0)


def real_vs_simulated(df, data_entity, start_year=2024,
                      sim_iterations=("iteration_01", "iteration_03", "iteration_05"), year_col="Call Year"):
    """Long frame (one row per document) with source in {real, simulated} and the year it stands for.
    Simulated generation k (k-th entry of sim_iterations) is placed at start_year + k."""
    years = df[year_col].to_numpy()
    orig = data_entity[data_entity["iteration"] == "original"]
    parts = []
    for y in sorted(np.unique(years)):
        m = per_doc_metrics(orig, np.flatnonzero(years == y))
        parts.append(m.assign(source="real", year=y, iteration="original"))
    start_ids = np.flatnonzero(years == start_year)
    for k, it in enumerate(["original", *sim_iterations]):
        m = per_doc_metrics(data_entity[data_entity["iteration"] == it], start_ids)
        parts.append(m.assign(source="simulated", year=start_year + k, iteration=it))
    return pd.concat(parts, ignore_index=True).fillna(0)


def summarize_real_vs_simulated(rvs, metric="n_mentions"):
    """Mean per year for real and simulated, their difference, and a Mann-Whitney U test where both exist."""
    stats = rvs.groupby(["year", "source"])[metric].mean().unstack("source")
    stats["sim_minus_real"] = stats["simulated"] - stats["real"]
    stats["p_mannwhitney"] = [
        mannwhitneyu(rvs.loc[(rvs.year == y) & (rvs.source == "real"), metric],
                     rvs.loc[(rvs.year == y) & (rvs.source == "simulated"), metric]).pvalue
        if not np.isnan(stats.loc[y, "real"]) and y != stats.index.min() else np.nan
        for y in stats.index
    ]
    return stats


def plot_real_vs_simulated(rvs, metrics=("n_mentions", "n_unique"), ncols=3, panel_size=(5.5, 4.5)):
    """One panel per metric: mean +/- 95% CI per year, real (solid) vs simulated from the first year (dashed).
    Metrics: "n_mentions", "n_unique" or an entity type name."""
    labels = {"n_mentions": "Entity mentions per abstract", "n_unique": "Distinct entities per abstract"}
    styles = {"real": dict(color="C0", marker="o", linestyle="-", label="Real abstracts"),
              "simulated": dict(color="C1", marker="s", linestyle="--", label="Simulated from 2024")}

    ncols = min(ncols, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             layout="constrained", squeeze=False)
    sim_its = rvs[rvs.source == "simulated"].groupby("year")["iteration"].first()
    all_years = sorted(rvs["year"].unique())

    for ax, metric in zip(axes.flat, metrics):
        for source, style in styles.items():
            g = rvs[rvs.source == source].groupby("year")[metric]
            mean, ci = g.mean(), 1.96 * g.sem()
            ax.errorbar(mean.index, mean.values, yerr=ci.values, capsize=4, linewidth=2,
                        markeredgecolor="black", markeredgewidth=0.5, **style)
        ax.set_xticks(all_years, [f"{y}\n({sim_its[y].replace('iteration_', 'it ')})" if sim_its.get(y, "original") != "original"
                                  else str(y) for y in all_years])
        ax.set_ylabel(labels.get(metric, metric))
        ax.grid(linestyle="--", alpha=0.7)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)

    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="outside lower center", ncol=2)
    return fig, axes


# ------------------------------------------------- ERC grants: entity overlap between real and simulated abstracts

def entity_counters_by_type(data_entity, iteration, doc_ids):
    """{type: Counter(entity -> mentions)} plus "ALL", for the given iteration and documents."""
    s = data_entity[(data_entity["iteration"] == iteration) & data_entity["document_ids"].isin(doc_ids)]
    by_type = {t: Counter(g.groupby("entity")["count"].sum().to_dict()) for t, g in s.groupby("type")}
    by_type["ALL"] = Counter(s.groupby("entity")["count"].sum().to_dict())
    return by_type


def overlap_pairs(df, data_entity, start_year=2024, sim_map=None, year_col="Call Year",
                  n_docs=None, n_repeats=20, seed=0):
    """Yield (year, sim_iteration, comparison, repeat, A, B), A/B = {type: Counter} (A is the reference).
    sim_map: {year: simulated iteration}, default {2025: iteration_01, 2026: iteration_03}.
    Both sides use n_docs documents (default: the smallest group); same seed -> same samples."""
    sim_map = sim_map or {start_year + 1: "iteration_01", start_year + 2: "iteration_03"}
    years = df[year_col].to_numpy()
    ids_of = {y: np.flatnonzero(years == y) for y in np.unique(years)}
    n_docs = n_docs or min(len(ids_of[y]) for y in [start_year, *sim_map])
    rng = np.random.default_rng(seed)

    for year, sim_it in sim_map.items():
        for rep in range(n_repeats):
            real_y = rng.choice(ids_of[year], n_docs, replace=False)
            src_a = rng.choice(ids_of[start_year], n_docs, replace=False)   # real 2024 docs for the baseline
            src_b = rng.choice(ids_of[start_year], n_docs, replace=False)   # 2024 docs the simulation starts from
            real_y_c = entity_counters_by_type(data_entity, "original", real_y)
            sim_c = entity_counters_by_type(data_entity, sim_it, src_b)
            yield year, sim_it, "Real Y vs Sim Y", rep, real_y_c, sim_c
            yield year, sim_it, "Real Y vs Real 2024 (baseline)", rep, real_y_c, entity_counters_by_type(data_entity, "original", src_a)
            yield year, sim_it, f"Real {start_year} vs Sim Y (same projects)", rep, entity_counters_by_type(data_entity, "original", src_b), sim_c


def real_vs_sim_overlap(df, data_entity, **pair_kwargs):
    """Tidy frame: one row per (year, comparison, type, repeat) with jaccard, js_divergence and
    retained_mass (share of A's mentions whose entity also appears in B). kwargs -> overlap_pairs."""
    rows = []
    for year, sim_it, comp, rep, A, B in overlap_pairs(df, data_entity, **pair_kwargs):
        for t in set(A) | set(B):
            a, b = A.get(t, Counter()), B.get(t, Counter())
            if not a or not b:
                continue
            m = compare_counters(a, b)
            rows.append({"year": year, "sim_iteration": sim_it, "comparison": comp, "type": t, "repeat": rep,
                         "jaccard": 1 - m["jaccard_dist"], "js_divergence": m["js_divergence"],
                         "retained_mass": m["retained_mass"]})
    return pd.DataFrame(rows)


def plot_overlap_by_type(overlap, metrics=("jaccard", "retained_mass", "js_divergence")):
    """Rows: metrics, columns: years. Horizontal bars per entity type (mean over repeats, +/- std)."""
    labels = {"jaccard": "Jaccard similarity (vocabulary)  ↑",
              "retained_mass": "Share of A's mentions found in B  ↑",
              "js_divergence": "Jensen-Shannon divergence  ↓",
              "bertscore_f1": "Entity BERTScore F1  ↑",
              "bertscore_recall": "Entity BERTScore recall (A covered by B)  ↑",
              "bertscore_precision": "Entity BERTScore precision (B matched in A)  ↑"}
    comps = list(dict.fromkeys(overlap["comparison"]))
    years = sorted(overlap["year"].unique())
    agg = overlap.groupby(["year", "comparison", "type"])[list(metrics)].agg(["mean", "std"])
    # order types by the first metric of the first comparison in the first year, ALL on top
    order = (agg.loc[(years[0], comps[0])][(metrics[0], "mean")].drop("ALL", errors="ignore")
             .sort_values().index.tolist() + ["ALL"])

    fig, axes = plt.subplots(len(metrics), len(years), figsize=(7 * len(years), 0.45 * len(order) * len(metrics) + 1),
                             layout="constrained", sharey=True, squeeze=False)
    h = 0.8 / len(comps)
    for r, metric in enumerate(metrics):
        for c, year in enumerate(years):
            ax = axes[r, c]
            for k, comp in enumerate(comps):
                sub = agg.loc[(year, comp)].reindex(order)
                ax.barh(np.arange(len(order)) + (k - (len(comps) - 1) / 2) * h, sub[(metric, "mean")],
                        xerr=sub[(metric, "std")], height=h, color=f"C{k}", label=comp, error_kw={"lw": 0.8})
            sim_it = overlap.loc[overlap.year == year, "sim_iteration"].iloc[0]
            ax.set_title(f"{year} (Sim = {sim_it})" if r == 0 else "")
            ax.set_xlabel(labels.get(metric, metric))
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.axhline(len(order) - 1.5, color="black", linewidth=0.8)
    axes[0, 0].set_yticks(range(len(order)), order)
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="outside lower center", ncol=len(comps))
    return fig, axes


# --------------------------------------------------------------------------- ERC grants: entity BERTScore (optional)

def make_embedders():
    """Embedder classes need torch/transformers, so they are only imported with --bertscore."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    def best_device():   # GPU with the most free memory
        return (f"cuda:{max(range(torch.cuda.device_count()), key=lambda d: torch.cuda.mem_get_info(d)[0])}"
                if torch.cuda.is_available() else "cpu")

    class EntityEmbedder:
        """BERTScore-style contextual embeddings of entity strings (default: roberta-large, layer 17 as in bert_score),
        mean-pooled over the entity's tokens (special tokens excluded), L2-normalized and cached per string."""

        def __init__(self, model_name="roberta-large", layer=17, device=None, batch_size=512):
            self.device, self.layer, self.batch_size = device or best_device(), layer, batch_size
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
            self.cache = {}

        @torch.no_grad()
        def _embed(self, batch):
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64,
                           return_special_tokens_mask=True).to(self.device)
            special = enc.pop("special_tokens_mask")
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[self.layer]
            mask = (enc["attention_mask"] * (1 - special)).unsqueeze(-1).to(hidden.dtype)
            return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)

        def __call__(self, entities):
            missing = [e for e in dict.fromkeys(entities) if e not in self.cache]
            for i in range(0, len(missing), self.batch_size):
                batch = missing[i:i + self.batch_size]
                self.cache.update(zip(batch, self._embed(batch).cpu()))
            return torch.stack([self.cache[e] for e in entities]).to(self.device)

        def random_pair_baseline(self, entities, n=4000, seed=0):
            """Mean cosine between random pairs of distinct entities: the b of BERTScore baseline rescaling."""
            pool = np.asarray(list(dict.fromkeys(entities)), dtype=object)
            sample = list(np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False))
            S = self(sample) @ self(sample).T
            return S[~torch.eye(len(sample), dtype=torch.bool, device=S.device)].mean().item()

    class SentenceEntityEmbedder(EntityEmbedder):
        """Same interface, but embeds each entity with a sentence-transformers model (its own pooling + normalization),
        e.g. "sentence-transformers/all-mpnet-base-v2", which is trained for phrase/sentence similarity."""

        def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device=None, batch_size=512):
            from sentence_transformers import SentenceTransformer
            self.device, self.batch_size = device or best_device(), batch_size
            self.model = SentenceTransformer(model_name, device=self.device)
            self.cache = {}

        def _embed(self, batch):
            return self.model.encode(batch, batch_size=self.batch_size, convert_to_tensor=True,
                                     normalize_embeddings=True, show_progress_bar=False)

    return EntityEmbedder, SentenceEntityEmbedder


def entity_bertscore(A: Counter, B: Counter, embedder, weighting="count", baseline=None):
    """BERTScore greedy matching between two entity multisets (A = reference, B = candidate).
    recall: each A entity matched to its most similar B entity; precision: vice versa.
    weighting: "count" (by mentions) or "uniform" (each distinct entity once).
    baseline: if given, scores are rescaled as (s - b) / (1 - b), so random entity pairs -> 0, identical sets -> 1."""
    import torch
    a_ents, b_ents = list(A), list(B)
    sim = embedder(a_ents) @ embedder(b_ents).T
    w = lambda C, ents: torch.tensor([C[e] if weighting == "count" else 1.0 for e in ents],
                                     device=sim.device, dtype=sim.dtype)
    wa, wb = w(A, a_ents), w(B, b_ents)
    recall = ((sim.max(dim=1).values * wa).sum() / wa.sum()).item()
    precision = ((sim.max(dim=0).values * wb).sum() / wb.sum()).item()
    f1 = 2 * precision * recall / (precision + recall)
    if baseline is not None:
        precision, recall, f1 = ((x - baseline) / (1 - baseline) for x in (precision, recall, f1))
    return {"bertscore_precision": precision, "bertscore_recall": recall, "bertscore_f1": f1}


def real_vs_sim_bertscore(df, data_entity, embedder, weighting="count", baseline=None, **pair_kwargs):
    """Same comparisons and document samples as real_vs_sim_overlap (same pair_kwargs/seed), scored with entity BERTScore."""
    rows = []
    for year, sim_it, comp, rep, A, B in overlap_pairs(df, data_entity, **pair_kwargs):
        for t in set(A) | set(B):
            a, b = A.get(t, Counter()), B.get(t, Counter())
            if not a or not b:
                continue
            rows.append({"year": year, "sim_iteration": sim_it, "comparison": comp, "type": t, "repeat": rep,
                         **entity_bertscore(a, b, embedder, weighting, baseline)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------- main

def run_dataset_plots(datasets: dict, out: Path, sample_id: int, fmt: str, top_types: int | None = None):
    """Plots of cells 5-8 of the notebook, for every dataset. top_types: only the most frequent entity types."""
    for name, fa in show_entity_types(datasets, top_k=top_types, shared_colors=False).items():
        save(fa, out / name / f"entity_types.{fmt}")
    for name, (data_type, data_entity) in datasets.items():
        save(plot_sample_trajectory(data_entity, sample_id=sample_id),
             out / name / f"sample_{sample_id}_trajectory.{fmt}")
        save(plot_retention_novelty(data_entity), out / name / f"retention_novelty.{fmt}")
        # one figure per metric, a line per entity type, in two versions:
        #   by_metric/:    pooled vocabulary of the whole corpus (same numbers as retention_novelty_by_type/)
        #   per_document/: each document's chain compared only with itself, then averaged over documents
        doc_types = top_entity_types(data_type, top_types) if top_types else None
        for folder, per_type, suffix in (
            ("by_metric", pooled_metrics(data_entity), ""),
            ("per_document", per_document_metrics(data_entity), " (mean per document)"),
        ):
            for metric in PER_DOC_METRICS:
                # top, middle and bottom entity type plus the average of all types
                save(plot_metric_by_type(per_type, metric, suffix, entity_types=doc_types),
                     out / name / folder / f"{metric}.{fmt}")
            (per_type.groupby(["type", "iteration"])[list(PER_DOC_METRICS)].mean().round(4)
             .to_csv(out / name / folder / "means.csv"))
        for entity_type in top_entity_types(data_type, top_types) if top_types else sorted(data_entity["type"].unique()):
            save(plot_retention_novelty(data_entity, entity_type=entity_type),
                 out / name / "retention_novelty_by_type" / f"{slug(entity_type)}.{fmt}")


def run_erc_plots(erc_data_entity, out: Path, fmt: str, erc_csv: str, bertscore: bool, top_types: int | None = None):
    """ERC-only analyses (call years of the original abstracts), cells 9-26 of the notebook."""
    df = pd.read_csv(erc_csv)
    out = out / "erc-grants" / "by_year"

    erc_per_doc, erc_per_type = erc_entities_per_document(df, erc_data_entity)
    for metric in ("n_mentions", "mentions_per_100_words"):   # the second controls for abstract length
        save(plot_entities_by_year(erc_per_doc, erc_per_type, metric=metric), out / f"entities_by_year_{metric}.{fmt}")
        summarize_entities_by_year(erc_per_doc, metric=metric).to_csv(out / f"entities_by_year_{metric}.csv")

    erc_by_year = split_by_year(df, erc_data_entity)
    for name, fa in show_entity_types(erc_by_year, top_k=top_types, ylabel="Mean mentions per document").items():
        save(fa, out / f"entity_types_{slug(name)}.{fmt}")
    for name, (_, data_entity) in erc_by_year.items():
        save(plot_retention_novelty(data_entity), out / f"retention_novelty_{slug(name)}.{fmt}")

    erc_rvs = real_vs_simulated(df, erc_data_entity, start_year=2024,
                                sim_iterations=("iteration_01", "iteration_03", "iteration_05"))
    save(plot_real_vs_simulated(erc_rvs, metrics=("n_mentions", "n_unique")), out / f"real_vs_simulated.{fmt}")
    summarize_real_vs_simulated(erc_rvs, "n_mentions").to_csv(out / "real_vs_simulated_n_mentions.csv")
    # per entity type, most frequent types in the real abstracts first
    top_types = (erc_rvs[erc_rvs.source == "real"]
                 .drop(columns=["n_mentions", "n_unique", "source", "year", "iteration"])
                 .mean().sort_values(ascending=False).index[:9])
    save(plot_real_vs_simulated(erc_rvs, metrics=top_types, ncols=3), out / f"real_vs_simulated_by_type.{fmt}")

    erc_overlap = real_vs_sim_overlap(df, erc_data_entity, start_year=2024, n_repeats=20)
    save(plot_overlap_by_type(erc_overlap), out / f"overlap_by_type.{fmt}")
    (erc_overlap.groupby(["year", "type", "comparison"])[["jaccard", "retained_mass", "js_divergence"]].mean()
     .unstack("comparison").round(3)).to_csv(out / "overlap_by_type.csv")

    if not bertscore:
        return
    EntityEmbedder, SentenceEntityEmbedder = make_embedders()
    baseline_entities = erc_data_entity.loc[
        erc_data_entity["iteration"].isin(["original", "iteration_01", "iteration_03"]), "entity"]
    bs_metrics = ["bertscore_f1", "bertscore_recall", "bertscore_precision"]
    results = {}
    for label, file_tag, embedder in (
        ("roberta-large L17", "roberta", lambda: EntityEmbedder("roberta-large", layer=17)),
        ("all-mpnet-base-v2", "mpnet", lambda: SentenceEntityEmbedder("sentence-transformers/all-mpnet-base-v2")),
    ):
        emb = embedder()
        # raw cosines of roberta-large are all ~0.9+, so rescale against random entity pairs (as bert_score does)
        b = emb.random_pair_baseline(baseline_entities)
        print(f"{label}: random-pair baseline b = {b:.3f}")
        bs = real_vs_sim_bertscore(df, erc_data_entity, emb, baseline=b, start_year=2024, n_repeats=20)
        save(plot_overlap_by_type(bs, metrics=tuple(bs_metrics)),
             out / f"bertscore_by_type_{file_tag}.{fmt}")
        bs.groupby(["year", "type", "comparison"])[bs_metrics].mean().unstack("comparison").round(3) \
            .to_csv(out / f"bertscore_by_type_{file_tag}.csv")
        results[label] = bs
    # side by side: rescaled scores, all entity types together
    pd.concat(results, names=["embedder"]).query("type == 'ALL'") \
        .groupby(["embedder", "year", "comparison"])[bs_metrics].mean().unstack("comparison").round(3) \
        .to_csv(out / "bertscore_all_types.csv")


def main():
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["bbc-news", "erc-grants", "acl"])
    parser.add_argument("--model", default="qwen_qwen3.8_27b", help="Generator folder name ('-' and '_' are interchangeable).")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path(OUT_DIR))
    parser.add_argument("--format", default=FORMAT, help="Figure format, e.g. pdf or png.")
    parser.add_argument("--sample-id", type=int, default=1000, help="Document shown in the sample trajectory heatmap.")
    parser.add_argument("--erc-csv", default=ERC_CSV, help="Original ERC data (Call Year, Abstract), row i = document_ids i.")
    parser.add_argument("--no-erc-years", action="store_true", help="Skip the ERC call-year analyses.")
    parser.add_argument("--bertscore", action="store_true", help="Also compute the ERC entity BERTScore plots (GPU).")
    parser.add_argument("--top-types", type=int, default=None,
                        help="Only plot the N most frequent entity types of each dataset (entity-type and per-type plots).")
    args = parser.parse_args()

    datasets = {name: load_dataset(name, args.model, args.data_root) for name in args.datasets}
    run_dataset_plots(datasets, args.out_dir, args.sample_id, args.format, args.top_types)
    if "erc-grants" in datasets and not args.no_erc_years:
        run_erc_plots(datasets["erc-grants"][1], args.out_dir, args.format, args.erc_csv, args.bertscore, args.top_types)


if __name__ == "__main__":
    main()
