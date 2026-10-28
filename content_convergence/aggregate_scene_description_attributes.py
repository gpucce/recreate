#!/usr/bin/env python
"""Aggregate attribute-value PROPORTIONS per generation step across all runs of a dataset.

No entity is paired across steps. Instead, for every step N the entities of
image_N from EVERY run are pooled, and for each (entity type, attribute key) the
script counts the values and reports each value's proportion at that step:

    step  type    key             value   count  total  proportion
    0     person  clothing.shirt  white   12     40     0.30
    1     person  clothing.shirt  white   9      38     0.237
    ...

Following one (type, key) down the steps shows whether the value distribution
drifts or collapses as the loop goes on.

Input: a dataset directory of runs (or the runs themselves), in either layout:

    per-step files, one run directory per image:
        data/outputs/entities_descriptions/vlstereoset/image_00002/image_0.visual_entities.fine.json
    per-run files, one JSON (list) or JSON Lines file per run, each record carrying its step:
        data/debug.template/entities_descriptions/vlstereoset/vlstereoset.image_00002.json
        data/outputs/entities_descriptions/vlstereoset/qwen3827b_zimage/image_01139.person.json

Experiments: the suffix between "image_<n>." and ".json" names the experiment
("visual_entities.fine", "person", ...); per-run files without a suffix are the
"template" experiment. Each experiment is aggregated separately, so files with
different suffixes are never pooled. Records may use the fine schema
({type, attributes, description}) or the template schema ({entity_type, parsed,
response}); records with neither (plain detections) count only toward entity types.

Choices:
    Keys are scoped by entity type ("person / clothing.shirt") unless --pool-types,
    which merges the same key across all types.
    Nested attributes are flattened to dotted keys: {"clothing": {"shirt": ..}} ->
    "clothing.shirt".
    Values are lowercased and whitespace-trimmed. With --tokens each value is split
    into words and every word is counted instead, which gives usable distributions
    for free-text values ("mauve short-sleeved top" -> mauve, short, sleeved, top).
    Counting unit: by default one count per entity. With --per-image a value
    counts at most once per image per (type, key), so an image with 15 trees
    does not outweigh one with a single tree.
    Entity types go through the same normalisation as compare_scene_description_entities.py
    ("chess_piece" == "chess piece", "trees" == "tree"); --no-normalize turns it off.
    Entities whose attributes failed to parse are recovered from their description
    (template files: from their raw response). In template files, null attributes are
    dropped, so they count neither as a value nor as key coverage.
    Small entities are dropped before anything is counted: an entity whose bounding box
    area (w x h, from "bbox"/"box", else the record's "area") is below --min-area px^2
    (default 128 x 128) is ignored everywhere, entity types included. Entities without
    a box are kept; --min-area 0 keeps everything.

Outputs (under <out>/<experiment>/, --out defaulting to <dataset>/../attribute_stats/<dataset name>):
    attribute_proportions.csv   one row per (step, type, key, value)
    entity_types.csv            per step: entity type counts, proportion of entities,
                                fraction of images containing that type, and the
                                ';'-separated paths of the runs (directories or
                                template files) whose image contains it
    key_coverage.csv            per step: how many entities / images have each key
    attribute_distributions.pdf        one row per entity type, one panel per attribute: a 100%
                                       stacked bar of value shares per step (top --plot-top-values
                                       values, the rest folded into "other"; n above each bar)
    attribute_distributions_lines.pdf  the same panels as one line per value
    panels/<bars|lines>/<type>__<key>.pdf         each panel of the two plots above as its own PDF
    panels/<bars|lines>/<type>__<key>.legend.pdf  its legend, saved separately (three columns when the
                                                  panel has more than two values)
    Each (type, key) panel stacks its values in one order, with one color per value, shared by every
    run through value_order.json (next to the experiment folders, by default attribute_stats/value_order.json;
    --value-order to choose it): values already in it keep their place and color, new ones are appended.
    The order is therefore the order in which values were first plotted; delete the file to rebuild it.
    The plots need matplotlib; --no-plot skips them. Rows are the --type types (default the
    --plot-n-types types with most counts), panels the --key keys (default the
    --plot-keys-per-type keys with most counts), both laid out alphabetically.

Usage:
    python content_convergence/aggregate_scene_description_attributes.py data/outputs/entities_descriptions/vlstereoset
    python content_convergence/aggregate_scene_description_attributes.py data/outputs/entities_descriptions/vlstereoset --per-image --tokens
    python content_convergence/aggregate_scene_description_attributes.py data/outputs/entities_descriptions/vlstereoset --type person --key clothing.shirt
    python content_convergence/aggregate_scene_description_attributes.py data/debug.template/entities_descriptions/vlstereoset
    python content_convergence/aggregate_scene_description_attributes.py data/outputs/entities_descriptions/vlstereoset/qwen3827b_zimage --experiment person
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_scene_description_entities import (  # noqa: E402
    flatten, load_step, normalize_type, recover_attributes)

STEP_FILE_RE = re.compile(r"^image_(\d+)\.(.+)\.json$")                # image_<step>.<experiment>.json in a run dir
RUN_FILE_RE = re.compile(r"^(?:[^.]+\.)?image_\d+(?:\.(.+))?\.json$")   # [<dataset>.]image_<row>[.<experiment>].json
RUN_DIR_RE = re.compile(r"^image_\d+$")
TEMPLATE_EXPERIMENT = "template"                                        # per-run files with no experiment suffix
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {"a", "an", "the", "and", "or", "of", "with", "in", "on", "at", "to", "is", "its"}
POOLED = "*"
DEFAULT_MIN_AREA = 256 * 256 # px^2


def step_experiments(run_dir: Path) -> List[str]:
    """Experiment names of the per-step files in a run directory (e.g. "visual_entities.fine")."""
    return sorted({m.group(2) for p in run_dir.iterdir() if (m := STEP_FILE_RE.match(p.name))})


def run_file_experiment(p: Path) -> Optional[str]:
    m = RUN_FILE_RE.match(p.name) if p.is_file() else None
    return (m.group(1) or TEMPLATE_EXPERIMENT) if m else None


def find_runs(paths: List[Path]) -> Dict[str, List[Path]]:
    """experiment -> runs. Accepts runs (per-step directories image_<row>/ or per-run files) and/or
    dataset directories containing them. A run directory yields one run per file suffix it holds."""
    runs: Dict[str, List[Path]] = defaultdict(list)

    def add(p: Path) -> None:
        if p.is_dir() and RUN_DIR_RE.match(p.name):
            for exp in step_experiments(p):
                runs[exp].append(p)
        elif (exp := run_file_experiment(p)) is not None:
            runs[exp].append(p)

    for p in paths:
        if p.is_dir() and not RUN_DIR_RE.match(p.name):
            for child in sorted(p.iterdir()):
                add(child)
        else:
            add(p)
    return dict(sorted(runs.items()))


def read_records(path: Path):
    """A JSON document, or JSON Lines."""
    text = path.read_text()
    try:
        return json.loads(text)
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def template_attributes(rec: Dict) -> Dict[str, str]:
    parsed = rec.get("parsed")
    if not isinstance(parsed, dict) or not parsed.get("parse_ok", True):
        try:
            parsed = json.loads(rec.get("response", ""))
        except (TypeError, ValueError):
            parsed = None
    if not isinstance(parsed, dict):
        return {}
    attrs = {k: v for k, v in parsed.items() if k != "parse_ok" and v is not None}
    return {k: v for k, v in flatten(attrs).items() if v.strip()}


def record_entity(rec: Dict, normalize: bool) -> Dict:
    """One entity record of either schema: fine ({type, attributes, description}) or
    template ({entity_type, parsed, response}). Records with neither (plain detections) get no attributes."""
    if "attributes" in rec or "description" in rec:
        attrs = flatten(recover_attributes(rec))
    else:
        attrs = template_attributes(rec)
    return {"type": normalize_type(rec.get("entity_type") or rec["type"], normalize), "attributes": attrs,
            "box": rec.get("bbox") or rec.get("box"), "area": rec.get("area")}


def entity_area(e: Dict) -> Optional[float]:
    """Bounding box area in px^2 (w x h of the box, else the record's "area"); None without either."""
    box = e.get("box")
    if box and len(box) == 4:
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
    return float(e["area"]) if e.get("area") is not None else None


def large_enough(e: Dict, min_area: float) -> bool:
    area = entity_area(e)
    return area is None or area >= min_area


def load_run(run: Path, experiment: str, normalize: bool) -> Dict[int, List[Dict]]:
    """step -> entities ({"type", "attributes"}), for either layout."""
    if run.is_dir():
        steps = {}
        for p in run.glob("image_*." + experiment + ".json"):
            m = STEP_FILE_RE.match(p.name)
            if not m or m.group(2) != experiment:
                continue
            data = read_records(p)
            if isinstance(data, dict):
                steps[int(m.group(1))] = load_step(p, normalize)
            else:
                steps[int(m.group(1))] = [record_entity(r, normalize) for r in data]
        return dict(sorted(steps.items()))
    data = read_records(run)
    by_step: Dict[int, List[Dict]] = defaultdict(list)
    for rec in [data] if isinstance(data, dict) else data:      # a one-line JSONL file parses as one record
        by_step[int(rec["step"])].append(record_entity(rec, normalize))
    return dict(sorted(by_step.items()))


def value_units(value: str, use_tokens: bool) -> List[str]:
    v = " ".join(value.lower().split())
    if not use_tokens:
        return [v] if v else []
    return [t for t in TOKEN_RE.findall(v) if t not in STOPWORDS]


# Categorical palette in fixed order (blue, orange, aqua, yellow, magenta, green, violet), gray for "other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
# Further colors for values that come later in the shared value order (see --value-order).
EXTRA_PALETTE = ["#8c564b", "#17becf", "#bcbd22", "#d62728", "#1f3b73", "#a6611a", "#5ab4ac", "#b2182b",
                 "#7570b3", "#66a61e", "#e6ab02", "#a6761d", "#e7298a"]
COLORS = PALETTE + EXTRA_PALETTE
OTHER_COLOR = "#c4c4c4"
OTHER = "other"


def value_color(order: List[str], value: str) -> str:
    return COLORS[order.index(value) % len(COLORS)]


def panel_table(values: Dict[Tuple[int, str, str], Counter], steps: List[int], etype: str, key: str,
                top_values: int, min_step_total: int,
                order: List[str]) -> Tuple[Dict[str, List[float]], List[int], List[str]]:
    """value -> share per step (NaN when the step has < min_step_total counts), totals per step, top values.
    The top values (by count over all steps) are laid out in the shared `order` of this (type, key), which
    is extended in place with values not seen before; the rest are folded into "other"."""
    overall: Counter = Counter()
    for step in steps:
        overall.update(values.get((step, etype, key), {}))
    top = [v for v, _ in overall.most_common(top_values)]
    order.extend(v for v in top if v not in order)
    top.sort(key=order.index)
    columns = top + ([OTHER] if len(overall) > len(top) else [])
    props: Dict[str, List[float]] = {col: [] for col in columns}
    totals = []
    for step in steps:
        c = values.get((step, etype, key), Counter())
        total = sum(c.values())
        totals.append(total)
        counts = {v: c[v] for v in top}
        if OTHER in props:
            counts[OTHER] = total - sum(counts.values())
        for col in columns:
            props[col].append(counts[col] / total if total >= min_step_total and total else float("nan"))
    return props, totals, top


def plot_distributions(values: Dict[Tuple[int, str, str], Counter], args: argparse.Namespace, out: Path,
                       title: str) -> None:
    """Write attribute_distributions.pdf (stacked bars) and attribute_distributions_lines.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalize = not args.no_normalize
    top_values = min(args.plot_top_values, len(COLORS))
    steps = sorted({step for step, _, _ in values})
    group_tot: Counter = Counter()
    type_tot: Counter = Counter()
    for (_, etype, key), c in values.items():
        group_tot[etype, key] += sum(c.values())
        type_tot[etype] += sum(c.values())
    # Pick types/keys by count, but lay them out alphabetically so panels line up across runs.
    if args.type:
        types = [t for t in dict.fromkeys(normalize_type(t, normalize) for t in args.type) if t in type_tot]
    else:
        types = sorted(t for t, _ in type_tot.most_common(args.plot_n_types))
    panels = {}
    for t in types:
        keys = [k for (et, k), _ in group_tot.most_common() if et == t and (not args.key or k in args.key)]
        if keys:
            panels[t] = sorted(keys[:args.plot_keys_per_type])
    if not panels:
        print(f"no attribute values to plot in {out}", file=sys.stderr)
        return

    nrows, ncols = len(panels), max(len(k) for k in panels.values())
    xticks = steps[::max(1, len(steps) // 8)]
    for kind, name in (("bars", "attribute_distributions.pdf"), ("lines", "attribute_distributions_lines.pdf")):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.9 * nrows), squeeze=False,
                                 sharey=True, layout="constrained")
        for r, (etype, keys) in enumerate(panels.items()):
            for c in range(ncols):
                ax = axes[r, c]
                if c >= len(keys):
                    ax.set_visible(False)
                    continue
                draw_panel(ax, kind, values, steps, etype, keys[c], top_values, args, xticks,
                           ylabel=c == 0, xlabel=r == nrows - 1)
        fig.suptitle(f"Attribute value distributions per generation step — {title}", fontsize=12)
        fig.savefig(out / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # The same panels, one PDF each, with their legend in a PDF of its own:
        # panels/<bars|lines>/<type>__<key>.pdf and <type>__<key>.legend.pdf
        panel_dir = out / "panels" / kind
        panel_dir.mkdir(parents=True, exist_ok=True)
        for etype, keys in panels.items():
            for key in keys:
                stem = f"{panel_file_name(etype)}__{panel_file_name(key)}"
                fig, ax = plt.subplots(figsize=(3.6, 3.6), layout="constrained")
                handles, labels = draw_panel(ax, kind, values, steps, etype, key, top_values, args, xticks,
                                             legend=False)
                fig.savefig(panel_dir / f"{stem}.pdf", bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(3.6, 0.5))
                fig.legend(handles, labels, loc="center", **legend_style(kind, labels))
                fig.savefig(panel_dir / f"{stem}.legend.pdf", bbox_inches="tight")
                plt.close(fig)


def panel_file_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "_"


def draw_panel(ax, kind: str, values: Dict[Tuple[int, str, str], Counter], steps: List[int], etype: str,
               key: str, top_values: int, args: argparse.Namespace, xticks: List[int], ylabel: bool = True,
               xlabel: bool = True, legend: bool = True) -> Tuple[List, List[str]]:
    """One (type, key) panel: stacked bars ("bars") or one line per value ("lines").
    Returns the legend handles and labels; legend=False leaves the legend off the axes."""
    from matplotlib.patches import Patch

    order = args.value_orders.setdefault(f"{etype} / {key}", [])
    props, totals, top = panel_table(values, steps, etype, key, top_values, args.plot_min_step_total, order)
    colors = [value_color(order, v) for v in top] + [OTHER_COLOR] * (len(props) - len(top))
    if kind == "bars":
        bottom = [0.0] * len(steps)
        for (col, shares), color in zip(props.items(), colors):
            h = [0.0 if x != x else x for x in shares]          # NaN -> no bar
            ax.bar(steps, h, bottom=bottom, width=0.85, color=color, edgecolor="white", linewidth=0.6)
            bottom = [b + x for b, x in zip(bottom, h)]
        handles = [Patch(color=k) for k in colors]
        ax.set_ylim(0, 1.0)
    else:
        for (col, shares), color in zip(props.items(), colors):
            other = col == OTHER and col not in top
            ax.plot(steps, shares, color=color, linewidth=1.5 if other else 2,
                    linestyle="--" if other else "-", marker="o", markersize=3)
        handles = ax.get_lines()
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
    for step, n in zip(steps, totals):
        if n and n >= args.plot_min_step_total:
            ax.text(step, 1.01 if kind == "bars" else 1.03, str(n), ha="center", va="bottom",
                    fontsize=6, color="#666666")
    ax.set_box_aspect(1)                                    # square plotting area
    ax.set_title(f"{etype} / {key}", fontsize=9, pad=12)
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=7, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    if ylabel:
        ax.set_ylabel("share of counts", fontsize=8)
    if xlabel:
        ax.set_xlabel("generation step", fontsize=8)
    labels = [v if len(v) <= 22 else v[:21] + "…" for v in props]
    if legend:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), **legend_style(kind, labels))
    return handles, labels


def legend_style(kind: str, labels: List[str]) -> Dict:
    """Legend layout: three columns when there are more than two values, else one row."""
    return dict(ncol=3 if len(labels) > 2 else max(1, len(labels)), fontsize=6.5, frameon=False,
                handlelength=1 if kind == "bars" else 1.5, columnspacing=0.8)


def aggregate(runs: List[Path], experiment: str, args: argparse.Namespace, out: Path) -> None:
    normalize = not args.no_normalize
    values: Dict[Tuple[int, str, str], Counter] = defaultdict(Counter)     # (step, type, key) -> value counts
    key_ents: Dict[Tuple[int, str, str], int] = Counter()                  # entities having the key
    key_imgs: Dict[Tuple[int, str, str], int] = Counter()                  # images having the key
    type_ents: Dict[Tuple[int, str], int] = Counter()
    type_imgs: Dict[Tuple[int, str], int] = Counter()
    type_runs: Dict[Tuple[int, str], List[str]] = defaultdict(list)       # paths of runs whose image has the type
    n_images: Dict[int, int] = Counter()
    n_entities: Dict[int, int] = Counter()
    n_small = n_unboxed = 0

    for run in runs:
        for step, ents in load_run(run, experiment, normalize).items():
            n_unboxed += sum(entity_area(e) is None for e in ents)
            kept = [e for e in ents if large_enough(e, args.min_area)]
            n_small += len(ents) - len(kept)
            ents = kept
            n_images[step] += 1
            n_entities[step] += len(ents)
            seen_units, seen_keys, seen_types = set(), set(), set()
            for e in ents:
                type_ents[step, e["type"]] += 1
                seen_types.add(e["type"])
                etype = POOLED if args.pool_types else e["type"]
                for key, val in e["attributes"].items():
                    g = (step, etype, key)
                    key_ents[g] += 1
                    seen_keys.add(g)
                    for unit in value_units(val, args.tokens):
                        if args.per_image:
                            if (g, unit) in seen_units:
                                continue
                            seen_units.add((g, unit))
                        values[g][unit] += 1
            for g in seen_keys:
                key_imgs[g] += 1
            for t in seen_types:
                type_imgs[step, t] += 1
                type_runs[step, t].append(run.as_posix())

    out.mkdir(parents=True, exist_ok=True)

    with open(out / "attribute_proportions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "type", "key", "value", "count", "total", "proportion"])
        for (step, etype, key), c in sorted(values.items()):
            total = sum(c.values())
            for val, n in c.most_common():
                w.writerow([step, etype, key, val, n, total, round(n / total, 4)])

    with open(out / "entity_types.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "type", "entities", "entity_proportion", "images_with_type", "image_fraction",
                    "images"])
        for (step, t), n in sorted(type_ents.items()):
            w.writerow([step, t, n, round(n / n_entities[step], 4),
                        type_imgs[step, t], round(type_imgs[step, t] / n_images[step], 4),
                        ";".join(sorted(type_runs[step, t]))])

    with open(out / "key_coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "type", "key", "entities_with_key", "images_with_key", "image_fraction"])
        for (step, etype, key), n in sorted(key_ents.items()):
            w.writerow([step, etype, key, n, key_imgs[step, etype, key],
                        round(key_imgs[step, etype, key] / n_images[step], 4)])

    if not args.no_print:
        print(f"######## experiment {experiment}")
        types = {normalize_type(t, normalize) for t in args.type}
        by_group: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for step, etype, key in sorted(values):
            by_group[etype, key].append(step)
        for (etype, key), steps in sorted(by_group.items()):
            if (types and etype not in types) or (args.key and key not in args.key):
                continue
            lines = []
            for step in steps:
                c = values[step, etype, key]
                total = sum(c.values())
                if total < args.min_total:
                    continue
                top = ", ".join(f"{v} {n / total:.2f}" for v, n in c.most_common(args.top))
                lines.append(f"  step {step:>2} (n={total:>3}): {top}")
            if lines:
                print(f"== {etype} / {key}")
                print("\n".join(lines))

    if not args.no_plot:
        try:
            plot_distributions(values, args, out, f"{out.parent.name} / {experiment}")
        except ImportError as e:
            print(f"skipping plots ({e}); install matplotlib or pass --no-plot", file=sys.stderr)

    steps = f"steps {min(n_images)}..{max(n_images)}" if n_images else "no steps"
    print(f"[{experiment}] {len(runs)} runs, {steps}, {sum(n_entities.values())} entities kept, {n_small} "
          f"dropped with box area < {args.min_area:g} px^2, {n_unboxed} without a box (kept) -> wrote {out}",
          file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="dataset directory or run directories")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root; each experiment writes its CSVs to <out>/<experiment>/")
    ap.add_argument("--experiment", action="append", default=[],
                    help="only aggregate these experiments (file suffixes, e.g. person, visual_entities.fine)")
    ap.add_argument("--pool-types", action="store_true", help="merge the same key across entity types")
    ap.add_argument("--tokens", action="store_true", help="count words of values instead of whole values")
    ap.add_argument("--per-image", action="store_true",
                    help="count each value at most once per image per (type, key)")
    ap.add_argument("--no-normalize", action="store_true", help="use raw entity type names")
    ap.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA,
                    help="drop entities whose bounding box area is below this many px^2 "
                         "(default: %(default)g = 128 x 128; 0 keeps all)")
    ap.add_argument("--type", action="append", default=[], help="only print / plot these entity types")
    ap.add_argument("--key", action="append", default=[], help="only print / plot these attribute keys")
    ap.add_argument("--top", type=int, default=5, help="values printed per (step, type, key)")
    ap.add_argument("--min-total", type=int, default=5,
                    help="only print (step, type, key) groups with at least this many counts")
    ap.add_argument("--no-print", action="store_true", help="do not print the value tables")
    ap.add_argument("--no-plot", action="store_true", help="do not write the PDF plots")
    ap.add_argument("--plot-n-types", type=int, default=4,
                    help="entity types plotted (most counts) when --type is not given")
    ap.add_argument("--plot-keys-per-type", type=int, default=10, help="attributes plotted per entity type")
    ap.add_argument("--plot-top-values", type=int, default=7,
                    help=f"values shown per panel, at most {len(COLORS)}; the rest become 'other'")
    ap.add_argument("--value-order", type=Path, default=None,
                    help="JSON file with the order (hence colors) of the values of each (type, key) panel, "
                         "shared by every run that uses it and extended with new values "
                         "(default: <out>/../value_order.json, i.e. attribute_stats/value_order.json)")
    ap.add_argument("--plot-min-step-total", type=int, default=1,
                    help="hide a step's bar when it has fewer counts than this")
    args = ap.parse_args(argv)

    experiments = find_runs(args.paths)
    if args.experiment:
        experiments = {e: r for e, r in experiments.items() if e in args.experiment}
    if not experiments:
        print("no runs found (image_<row>/image_<step>.<experiment>.json or [<dataset>.]image_<row>[.<experiment>].json)",
              file=sys.stderr)
        return 1

    out_root = args.out or args.paths[0].resolve().parent / "attribute_stats" / args.paths[0].resolve().name
    order_path = args.value_order or out_root.resolve().parent / "value_order.json"
    args.value_orders = json.loads(order_path.read_text()) if order_path.exists() else {}
    for experiment, runs in experiments.items():
        aggregate(runs, experiment, args, out_root / experiment)
    if not args.no_plot and args.value_orders:
        order_path.parent.mkdir(parents=True, exist_ok=True)
        order_path.write_text(json.dumps(args.value_orders, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
