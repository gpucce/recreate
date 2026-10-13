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
    template files, one JSON per run holding a list of {step, entity_type, box, response, parsed}:
        data/debug.template/entities_descriptions/vlstereoset/vlstereoset.image_00002.json

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

Outputs (under --out, default <dataset>/../attribute_stats/<dataset name>):
    attribute_proportions.csv   one row per (step, type, key, value)
    entity_types.csv            per step: entity type counts, proportion of entities,
                                fraction of images containing that type, and the
                                ';'-separated paths of the runs (directories or
                                template files) whose image contains it
    key_coverage.csv            per step: how many entities / images have each key

Usage:
    python content_convergence/aggregate_scene_description_attributes.py data/outputs/entities_descriptions/vlstereoset
    python content_convergence/aggregate_scene_description_attributes.py data/outputs/entities_descriptions/vlstereoset --per-image --tokens
    python content_convergence/aggregate_scene_description_attributes.py data/outputs/entities_descriptions/vlstereoset --type person --key clothing.shirt
    python content_convergence/aggregate_scene_description_attributes.py data/debug.template/entities_descriptions/vlstereoset
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
    FINE_SUFFIX, available_steps, flatten, load_step, normalize_type)

TEMPLATE_RE = re.compile(r"^(?:.+\.)?image_\d+\.json$")   # e.g. vlstereoset.image_00002.json
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {"a", "an", "the", "and", "or", "of", "with", "in", "on", "at", "to", "is", "its"}
POOLED = "*"


def is_template_file(p: Path) -> bool:
    return p.is_file() and TEMPLATE_RE.match(p.name) is not None and not p.name.endswith(FINE_SUFFIX)


def find_runs(paths: List[Path]) -> List[Path]:
    """Accept runs (per-step directories or template files) and/or dataset directories containing them."""
    runs = []
    for p in paths:
        if is_template_file(p):
            runs.append(p)
        elif not p.is_dir():
            continue
        elif any(p.glob("image_*" + FINE_SUFFIX)):
            runs.append(p)
        else:
            runs += sorted(d for d in p.iterdir() if (d.is_dir() and any(d.glob("image_*" + FINE_SUFFIX)))
                           or is_template_file(d))
    return runs


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


def load_run(run: Path, normalize: bool) -> Dict[int, List[Dict]]:
    """step -> entities ({"type", "attributes"}), for either layout."""
    if run.is_dir():
        return {step: load_step(path, normalize) for step, path in available_steps(run).items()}
    steps: Dict[int, List[Dict]] = defaultdict(list)
    for rec in json.loads(run.read_text()):
        steps[int(rec["step"])].append({"type": normalize_type(rec["entity_type"], normalize),
                                        "attributes": template_attributes(rec)})
    return dict(sorted(steps.items()))


def value_units(value: str, use_tokens: bool) -> List[str]:
    v = " ".join(value.lower().split())
    if not use_tokens:
        return [v] if v else []
    return [t for t in TOKEN_RE.findall(v) if t not in STOPWORDS]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="dataset directory or run directories")
    ap.add_argument("--out", type=Path, default=None, help="output directory for the CSVs")
    ap.add_argument("--pool-types", action="store_true", help="merge the same key across entity types")
    ap.add_argument("--tokens", action="store_true", help="count words of values instead of whole values")
    ap.add_argument("--per-image", action="store_true",
                    help="count each value at most once per image per (type, key)")
    ap.add_argument("--no-normalize", action="store_true", help="use raw entity type names")
    ap.add_argument("--type", action="append", default=[], help="only print these entity types")
    ap.add_argument("--key", action="append", default=[], help="only print these attribute keys")
    ap.add_argument("--top", type=int, default=5, help="values printed per (step, type, key)")
    ap.add_argument("--min-total", type=int, default=5,
                    help="only print (step, type, key) groups with at least this many counts")
    ap.add_argument("--no-print", action="store_true", help="only write the CSVs")
    args = ap.parse_args(argv)
    normalize = not args.no_normalize

    runs = find_runs(args.paths)
    if not runs:
        print("no runs with *" + FINE_SUFFIX + " or *.image_<row>.json found", file=sys.stderr)
        return 1

    values: Dict[Tuple[int, str, str], Counter] = defaultdict(Counter)     # (step, type, key) -> value counts
    key_ents: Dict[Tuple[int, str, str], int] = Counter()                  # entities having the key
    key_imgs: Dict[Tuple[int, str, str], int] = Counter()                  # images having the key
    type_ents: Dict[Tuple[int, str], int] = Counter()
    type_imgs: Dict[Tuple[int, str], int] = Counter()
    type_runs: Dict[Tuple[int, str], List[str]] = defaultdict(list)       # paths of runs whose image has the type
    n_images: Dict[int, int] = Counter()
    n_entities: Dict[int, int] = Counter()

    for run in runs:
        for step, ents in load_run(run, normalize).items():
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

    out = args.out or args.paths[0].resolve().parent / "attribute_stats" / args.paths[0].resolve().name
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

    print(f"{len(runs)} runs, steps {min(n_images)}..{max(n_images)} -> wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
