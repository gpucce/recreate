#!/usr/bin/env python
"""Compare the fine-grained ENTITIES and ATTRIBUTES of SUBSEQUENT steps of a content-convergence run.

Input: a run directory of fine descriptions, e.g.

    data/outputs/entities_descriptions/vlstereoset/image_00002/
        image_0.visual_entities.fine.json
        image_1.visual_entities.fine.json
        ...

each holding an `entities` list whose items carry `type`, `index`, `box`,
`description` and a (possibly nested) `attributes` dict.

Pairs (same meaning as validate_run_diffs.py / semantic_validate.py):
    consecutive  every (image_N, image_N+1)
    anchor       every image_N against the fixed step --anchor K (off unless given)

For each pair the script reports
    entity level     which entity TYPES appear / disappear, and how their instance
                     counts change (type-set Jaccard as a single number)
    instance level   instances of the same type are matched one-to-one between the two
                     steps (greedy on box IoU + attribute similarity); unmatched ones
                     are reported as added / removed
    attribute level  for each matched pair, the nested attributes are flattened to
                     dotted keys (e.g. "clothing.shirt") and split into
                     added / removed / changed / unchanged, with a token-Jaccard
                     similarity for each changed value

It also chains the consecutive matches into TRACKS (one per entity followed across
steps), so you can see how e.g. person_0's "clothing.shirt" evolves along the run.

Type names are normalised before comparison ("chess_piece" == "chess piece",
"trees" == "tree"); pass --no-normalize to compare raw types. Entities whose
attributes failed to parse (parse_ok == false) are recovered from the JSON string
in their `description` when possible.

Everything here is stdlib only and READ-ONLY with respect to the input directory.

Usage:
    python content_convergence/compare_scene_description_entities.py data/outputs/entities_descriptions/vlstereoset/image_00002
    python content_convergence/compare_scene_description_entities.py data/outputs/entities_descriptions/vlstereoset/image_* --anchor 0
    python content_convergence/compare_scene_description_entities.py data/outputs/entities_descriptions/vlstereoset/image_00002 --track person
    python content_convergence/compare_scene_description_entities.py data/outputs/entities_descriptions/vlstereoset/image_* --out cmp/ --quiet

Outputs (only with --out DIR):
    <run>.pairs.json    full per-pair comparison (entities, matches, attribute diffs)
    <run>.tracks.json   entity tracks with their attributes at every step
    summary.csv         one row per (run, relation, step_a, step_b) with the headline numbers
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FINE_SUFFIX = ".visual_entities.fine.json"
STEP_RE = re.compile(r"^image_(\d+)" + re.escape(FINE_SUFFIX) + "$")
TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_BOX_WEIGHT = 0.5      # weight of box IoU vs attribute similarity when matching instances
DEFAULT_MIN_MATCH = 0.1       # below this combined score two instances are NOT the same entity


# ---- loading --------------------------------------------------------------
def available_steps(run_dir: Path) -> Dict[int, Path]:
    steps = {}
    for p in run_dir.iterdir():
        m = STEP_RE.match(p.name)
        if m:
            steps[int(m.group(1))] = p
    return dict(sorted(steps.items()))


def singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "ches", "shes", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_type(t: str, enabled: bool = True) -> str:
    if not enabled:
        return t
    words = re.sub(r"[_\s]+", " ", t.strip().lower()).split(" ")
    words[-1] = singularize(words[-1])
    return " ".join(words)


def recover_attributes(ent: Dict[str, Any]) -> Dict[str, Any]:
    """Return the entity's attributes, recovering them from a JSON `description`
    when the describer failed to parse its own output."""
    attrs = ent.get("attributes") or {}
    if attrs or ent.get("parse_ok", True):
        return attrs
    try:
        parsed = json.loads(ent.get("description", ""))
    except (TypeError, ValueError):
        return {}
    return parsed.get("attributes", {}) if isinstance(parsed, dict) else {}


def flatten(obj: Any, prefix: str = "") -> Dict[str, str]:
    """{"clothing": {"shirt": "red"}} -> {"clothing.shirt": "red"}"""
    if isinstance(obj, dict):
        out: Dict[str, str] = {}
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
        return out
    if isinstance(obj, list):
        return {prefix: ", ".join(str(x) for x in obj)}
    return {prefix: "" if obj is None else str(obj)}


def load_step(path: Path, normalize: bool) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    ents = []
    for e in data.get("entities", []):
        ents.append({
            "id": f"{e['type']}_{e['index']}",
            "type": normalize_type(e["type"], normalize),
            "raw_type": e["type"],
            "box": e.get("box"),
            "description": e.get("description", "") if e.get("parse_ok", True) else "",
            "attributes": flatten(recover_attributes(e)),
            "parse_ok": e.get("parse_ok", True),
        })
    return ents


# ---- similarity -----------------------------------------------------------
def tokens(s: str) -> set:
    return set(TOKEN_RE.findall(s.lower()))


def text_sim(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def box_iou(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def attr_sim(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Mean value similarity over the union of keys (a missing key scores 0)."""
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    return sum(text_sim(a[k], b[k]) for k in keys if k in a and k in b) / len(keys)


def diff_attributes(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, Any]:
    changed = {}
    unchanged = []
    for k in sorted(set(a) & set(b)):
        if a[k].strip().lower() == b[k].strip().lower():
            unchanged.append(k)
        else:
            changed[k] = {"from": a[k], "to": b[k], "sim": round(text_sim(a[k], b[k]), 3)}
    return {
        "added": {k: b[k] for k in sorted(set(b) - set(a))},
        "removed": {k: a[k] for k in sorted(set(a) - set(b))},
        "changed": changed,
        "unchanged": unchanged,
        "similarity": round(attr_sim(a, b), 3),
    }


# ---- matching -------------------------------------------------------------
def match_instances(ea: List[Dict], eb: List[Dict], box_weight: float,
                    min_match: float) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching of same-type entities; returns (i_a, i_b, score)."""
    cands = []
    for i, x in enumerate(ea):
        for j, y in enumerate(eb):
            if x["type"] != y["type"]:
                continue
            content = attr_sim(x["attributes"], y["attributes"])
            if not x["attributes"] and not y["attributes"]:
                content = text_sim(x["description"], y["description"])
            score = box_weight * box_iou(x["box"], y["box"]) + (1 - box_weight) * content
            cands.append((score, i, j))
    cands.sort(reverse=True)
    used_a, used_b, out = set(), set(), []
    for score, i, j in cands:
        if i in used_a or j in used_b or score < min_match:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, score))
    return out


def count_types(ents: Iterable[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in ents:
        out[e["type"]] = out.get(e["type"], 0) + 1
    return out


def compare_steps(ea: List[Dict], eb: List[Dict], box_weight: float,
                  min_match: float) -> Dict[str, Any]:
    ca, cb = count_types(ea), count_types(eb)
    ta, tb = set(ca), set(cb)
    matches = match_instances(ea, eb, box_weight, min_match)
    matched_a = {i for i, _, _ in matches}
    matched_b = {j for _, j, _ in matches}

    matched = []
    for i, j, score in sorted(matches, key=lambda m: (ea[m[0]]["type"], m[0])):
        d = diff_attributes(ea[i]["attributes"], eb[j]["attributes"])
        matched.append({"a": ea[i]["id"], "b": eb[j]["id"], "type": ea[i]["type"],
                        "match_score": round(score, 3),
                        "box_iou": round(box_iou(ea[i]["box"], eb[j]["box"]), 3), **d})

    n_keys = sum(len(m["changed"]) + len(m["unchanged"]) for m in matched)
    n_unch = sum(len(m["unchanged"]) for m in matched)
    return {
        "types": {
            "added": sorted(tb - ta),
            "removed": sorted(ta - tb),
            "shared": sorted(ta & tb),
            "jaccard": round(len(ta & tb) / len(ta | tb), 3) if ta | tb else 1.0,
            "count_changes": {t: [ca.get(t, 0), cb.get(t, 0)]
                              for t in sorted(ta | tb) if ca.get(t, 0) != cb.get(t, 0)},
        },
        "instances": {
            "n_a": len(ea), "n_b": len(eb), "n_matched": len(matches),
            "added": [eb[j]["id"] for j in range(len(eb)) if j not in matched_b],
            "removed": [ea[i]["id"] for i in range(len(ea)) if i not in matched_a],
        },
        "matched": matched,
        "summary": {
            "type_jaccard": round(len(ta & tb) / len(ta | tb), 3) if ta | tb else 1.0,
            "instance_match_rate": round(2 * len(matches) / (len(ea) + len(eb)), 3)
                                   if ea or eb else 1.0,
            "mean_attr_similarity": round(sum(m["similarity"] for m in matched) / len(matched), 3)
                                    if matched else None,
            "shared_attr_keys": n_keys,
            "unchanged_attr_frac": round(n_unch / n_keys, 3) if n_keys else None,
        },
    }


# ---- tracks ---------------------------------------------------------------
def build_tracks(steps: Dict[int, List[Dict]], box_weight: float,
                 min_match: float) -> List[Dict[str, Any]]:
    """Chain consecutive matches: each track follows one entity for as long as it
    keeps being matched; an unmatched entity starts a new track."""
    order = list(steps)
    tracks: List[Dict[str, Any]] = []
    live: Dict[int, int] = {}      # entity index in the previous step -> track id
    for n, step in enumerate(order):
        ents = steps[step]
        new_live: Dict[int, int] = {}
        if n > 0:
            for i, j, _ in match_instances(steps[order[n - 1]], ents, box_weight, min_match):
                if i in live:
                    new_live[j] = live[i]
        for j, e in enumerate(ents):
            if j not in new_live:
                new_live[j] = len(tracks)
                tracks.append({"track": len(tracks), "type": e["type"], "steps": {}})
            tracks[new_live[j]]["steps"][step] = {"id": e["id"], "attributes": e["attributes"]}
        live = new_live
    for t in tracks:
        t["length"] = len(t["steps"])
    return tracks


# ---- reporting ------------------------------------------------------------
def fmt_pair(run: str, rel: str, a: int, b: int, cmp: Dict[str, Any], max_changes: int) -> str:
    s, ty, inst = cmp["summary"], cmp["types"], cmp["instances"]
    lines = [f"=== {run}  [{rel}]  image_{a} -> image_{b}",
             f"  types: jaccard={s['type_jaccard']}  +{ty['added']}  -{ty['removed']}"]
    if ty["count_changes"]:
        lines.append("  counts: " + ", ".join(f"{t} {x}->{y}" for t, (x, y)
                                             in ty["count_changes"].items()))
    lines.append(f"  instances: {inst['n_a']} -> {inst['n_b']}, matched {inst['n_matched']} "
                 f"(rate {s['instance_match_rate']}), mean attr sim {s['mean_attr_similarity']}, "
                 f"unchanged attr frac {s['unchanged_attr_frac']}")
    shown = 0
    for m in cmp["matched"]:
        if not (m["changed"] or m["added"] or m["removed"]):
            continue
        if shown >= max_changes:
            lines.append("    ...")
            break
        shown += 1
        lines.append(f"    {m['a']} -> {m['b']}  (sim {m['similarity']}, iou {m['box_iou']})")
        for k, v in m["changed"].items():
            lines.append(f"      ~ {k}: {v['from']!r} -> {v['to']!r}")
        for k, v in m["added"].items():
            lines.append(f"      + {k}: {v!r}")
        for k, v in m["removed"].items():
            lines.append(f"      - {k}: {v!r}")
    return "\n".join(lines)


def fmt_tracks(tracks: List[Dict], type_filter: str, normalize: bool) -> str:
    want = normalize_type(type_filter, normalize)
    lines = [f"--- tracks of type {want!r}"]
    for t in tracks:
        if t["type"] != want:
            continue
        steps = sorted(t["steps"])
        lines.append(f"  track {t['track']}: steps {steps[0]}..{steps[-1]} "
                     f"({t['length']} steps) ids " + " ".join(t["steps"][s]["id"] for s in steps))
        keys = sorted({k for s in steps for k in t["steps"][s]["attributes"]})
        for k in keys:
            vals = [t["steps"][s]["attributes"].get(k, "·") for s in steps]
            lines.append(f"    {k}: " + " | ".join(vals))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run directories holding image_N" + FINE_SUFFIX)
    p.add_argument("--anchor", type=int, default=None,
                   help="also compare every step against this fixed step")
    p.add_argument("--no-consecutive", action="store_true", help="skip consecutive pairs")
    p.add_argument("--track", action="append", default=[], metavar="TYPE",
                   help="print the attribute trajectory of every track of TYPE (repeatable)")
    p.add_argument("--no-normalize", action="store_true", help="compare raw entity type names")
    p.add_argument("--box-weight", type=float, default=DEFAULT_BOX_WEIGHT,
                   help="weight of box IoU (vs attribute similarity) when matching instances")
    p.add_argument("--min-match", type=float, default=DEFAULT_MIN_MATCH,
                   help="minimum combined score for two instances to be matched")
    p.add_argument("--max-changes", type=int, default=10,
                   help="max matched instances with attribute diffs printed per pair")
    p.add_argument("--out", type=Path, default=None, help="write JSON/CSV results here")
    p.add_argument("--quiet", action="store_true", help="print only per-pair headline lines")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    normalize = not args.no_normalize
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for run_dir in args.runs:
        if not run_dir.is_dir():
            continue
        paths = available_steps(run_dir)
        if len(paths) < 2:
            print(f"skip {run_dir}: fewer than 2 steps", file=sys.stderr)
            continue
        run = run_dir.name
        steps = {n: load_step(p, normalize) for n, p in paths.items()}
        order = list(steps)

        pairs: List[Tuple[str, int, int]] = []
        if not args.no_consecutive:
            pairs += [("consecutive", a, b) for a, b in zip(order, order[1:])]
        if args.anchor is not None:
            if args.anchor not in steps:
                print(f"{run}: anchor step {args.anchor} missing", file=sys.stderr)
            else:
                pairs += [("anchor", args.anchor, n) for n in order if n != args.anchor]

        results = []
        for rel, a, b in pairs:
            cmp = compare_steps(steps[a], steps[b], args.box_weight, args.min_match)
            results.append({"relation": rel, "step_a": a, "step_b": b, **cmp})
            rows.append({"run": run, "relation": rel, "step_a": a, "step_b": b, **cmp["summary"],
                         "types_added": len(cmp["types"]["added"]),
                         "types_removed": len(cmp["types"]["removed"]),
                         "instances_added": len(cmp["instances"]["added"]),
                         "instances_removed": len(cmp["instances"]["removed"])})
            if args.quiet:
                s = cmp["summary"]
                print(f"{run} {rel:11s} {a:>3}->{b:<3} type_jacc={s['type_jaccard']:.3f} "
                      f"inst_match={s['instance_match_rate']:.3f} attr_sim={s['mean_attr_similarity']} "
                      f"unchanged={s['unchanged_attr_frac']}")
            else:
                print(fmt_pair(run, rel, a, b, cmp, args.max_changes))

        tracks = build_tracks(steps, args.box_weight, args.min_match)
        for t in args.track:
            print(fmt_tracks(tracks, t, normalize))

        if args.out:
            (args.out / f"{run}.pairs.json").write_text(json.dumps(results, indent=2))
            (args.out / f"{run}.tracks.json").write_text(json.dumps(tracks, indent=2))

    if args.out and rows:
        with open(args.out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
