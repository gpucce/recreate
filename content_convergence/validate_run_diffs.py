#!/usr/bin/env python
"""Compare SUBSEQUENT steps of a content-convergence run with a multi-image VLM.

A run directory produced by run_bbq_loop.py looks like:

    bbq_runs/image_09134/
        image_0.png      <- seed
        prompt_1.txt     <- VLM description of image_0
        image_1.png      <- generated from prompt_1
        prompt_2.txt     <- VLM description of image_1
        ...

This script takes each CONSECUTIVE pair -- (image_N, image_N+1) or
(prompt_N, prompt_N+1) -- shows BOTH to the vision model IN ONE CALL, and asks
it to name the main differences. It ALSO compares every step against the FIXED
step --anchor K (3 by default), which is a different question: consecutive pairs
can all read "minor" while the run drifts a long way from where it started, and
only the anchor comparison sees that accumulation. That doubles the number of VLM
calls; --no-anchor goes back to consecutive pairs only. Two images in a single message is the point:
the model sees them side by side rather than describing each in isolation, so
"the jacket changed from brown to black" is a thing it can actually say.

The model is asked for JSON, and the answer is stored as JSONL, one record per
pair. A run that is converging should show verdicts decaying from major/moderate
toward minor/identical as N grows -- that trajectory is the validation signal.

This script is READ-ONLY with respect to run directories. It never writes into
image_*/; results go to a sibling `validation/` directory. It is safe to run
against a live run: pairs whose files do not exist yet are skipped.

Usage:
    ./conda_venv/bin/python content_convergence/validate_run_diffs.py bbq_runs/image_09134
    ./conda_venv/bin/python content_convergence/validate_run_diffs.py bbq_runs/image_* --gpu 0
    ./conda_venv/bin/python content_convergence/validate_run_diffs.py bbq_runs/image_09134 --step 3
    ./conda_venv/bin/python content_convergence/validate_run_diffs.py bbq_runs/image_09134 --mode both
    ./conda_venv/bin/python content_convergence/validate_run_diffs.py bbq_runs/image_09134 --anchor 5
    ./conda_venv/bin/python content_convergence/validate_run_diffs.py bbq_runs/image_09134 --no-anchor
    ./conda_venv/bin/python content_convergence/validate_run_diffs.py bbq_runs/image_09134 --dry-run

VRAM note:
    Qwen3-VL-4B needs roughly 9-10 GB. Pin an idle card with --gpu N; check
    free memory first, since a 200-image run keeps every GPU near capacity.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


HERE = Path(__file__).resolve().parent
RECREATE_DIR = HERE.parent   # repo root: holds describe.py / loop.py

DEFAULT_VISION_MODEL = "Qwen/Qwen3-VL-4B-Instruct"   # cached; same model the loop uses
DEFAULT_MAX_SIZE = 512                               # matches describe.DEFAULT_MAX_SIZE
DEFAULT_MAX_NEW_TOKENS = 768                         # a difference list, not an essay
VERDICTS = ("identical", "minor", "moderate", "major")
DEFAULT_ANCHOR = 3          # also compare every step against this fixed step


def _pin_gpu_before_torch() -> None:
    """Honour --gpu by setting CUDA_VISIBLE_DEVICES before torch is imported.

    Once torch initialises CUDA the variable is ignored, so this has to happen
    before `import describe` pulls torch in.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu", type=int, default=None)
    known, _ = pre.parse_known_args()
    if known.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_pin_gpu_before_torch()

sys.path.insert(0, str(RECREATE_DIR))
import describe  # noqa: E402  (must follow the GPU pin and the sys.path insert)


IMAGE_COMPARE_REQUEST = """
You are shown TWO images from an iterative image-regeneration loop. The FIRST image is
step {a}. The SECOND image is step {b}: it was produced by describing the first image in
words and giving that description to an image generator, so it is an imperfect copy.

Name the MAIN differences between the first image and the second. Compare subjects and
how many there are, their pose, gaze, clothing, age and appearance, the composition and
camera angle, lighting, colour palette, materials and textures, background, image style,
and any legible text or symbols. Ignore differences too small to notice at a glance.

Reply with ONLY a JSON object, no other text:
{{"verdict": "<identical|minor|moderate|major>",
  "differences": ["<one concrete difference>", "..."],
  "summary": "<one sentence>"}}

verdict: "identical" if you cannot tell them apart, "minor" if only details moved,
"moderate" if a clearly visible attribute changed, "major" if the subject or scene is
no longer the same thing. differences: at most 8 entries, each naming ONE change and
stating both sides, e.g. "the mug moved from the left of the desk to the right".
Use an empty list if the images are identical.
""".strip()

PROMPT_COMPARE_REQUEST = """
You are shown TWO text prompts from an iterative image-regeneration loop. Prompt A is
step {a}. Prompt B is step {b}: it describes the image that was generated from prompt A,
so it is an imperfect copy of A passed through an image.

Name the MAIN differences in what the two prompts describe. Compare the subjects and how
many there are, their pose, clothing and appearance, composition and camera angle,
lighting, colour palette, materials, background, style, and any quoted text. Ignore pure
rewording that describes the same scene.

--- PROMPT A (step {a}) ---
{text_a}

--- PROMPT B (step {b}) ---
{text_b}

Reply with ONLY a JSON object, no other text:
{{"verdict": "<identical|minor|moderate|major>",
  "differences": ["<one concrete difference>", "..."],
  "summary": "<one sentence>"}}

verdict: "identical" if they describe the same scene, "minor" if only details moved,
"moderate" if a clearly visible attribute changed, "major" if the scene is no longer the
same thing. differences: at most 8 entries, each naming ONE change and stating both
sides. Use an empty list if nothing meaningful changed.
""".strip()

IMAGE_ANCHOR_REQUEST = """
You are shown TWO images from an iterative image-regeneration loop. The FIRST image is
step {a}. The SECOND image is step {b}, {distance} loop steps later: each step describes
the previous image in words and regenerates it from that description, so whatever
changed at each step has accumulated between these two.

Name the MAIN differences between the first image and the second. Compare subjects and
how many there are, their pose, gaze, clothing, age and appearance, the composition and
camera angle, lighting, colour palette, materials and textures, background, image style,
and any legible text or symbols. Ignore differences too small to notice at a glance.

Reply with ONLY a JSON object, no other text:
{{"verdict": "<identical|minor|moderate|major>",
  "differences": ["<one concrete difference>", "..."],
  "summary": "<one sentence>"}}

verdict: "identical" if you cannot tell them apart, "minor" if only details moved,
"moderate" if a clearly visible attribute changed, "major" if the subject or scene is
no longer the same thing. differences: at most 8 entries, each naming ONE change and
stating both sides, e.g. "the mug moved from the left of the desk to the right".
Use an empty list if the images are identical.
""".strip()

PROMPT_ANCHOR_REQUEST = """
You are shown TWO text prompts from an iterative image-regeneration loop. Prompt A is
step {a}. Prompt B is step {b}, {distance} loop steps later: each step describes the
image generated from the previous prompt, so whatever changed at each step has
accumulated between these two.

Name the MAIN differences in what the two prompts describe. Compare the subjects and how
many there are, their pose, clothing and appearance, composition and camera angle,
lighting, colour palette, materials, background, style, and any quoted text. Ignore pure
rewording that describes the same scene.

--- PROMPT A (step {a}) ---
{text_a}

--- PROMPT B (step {b}) ---
{text_b}

Reply with ONLY a JSON object, no other text:
{{"verdict": "<identical|minor|moderate|major>",
  "differences": ["<one concrete difference>", "..."],
  "summary": "<one sentence>"}}

verdict: "identical" if they describe the same scene, "minor" if only details moved,
"moderate" if a clearly visible attribute changed, "major" if the scene is no longer the
same thing. differences: at most 8 entries, each naming ONE change and stating both
sides. Use an empty list if nothing meaningful changed.
""".strip()

TEMPLATES = {
    ("images", "consecutive"): IMAGE_COMPARE_REQUEST,
    ("images", "anchor"): IMAGE_ANCHOR_REQUEST,
    ("prompts", "consecutive"): PROMPT_COMPARE_REQUEST,
    ("prompts", "anchor"): PROMPT_ANCHOR_REQUEST,
}



# ---- pair discovery -------------------------------------------------------
def available_steps(run_dir: Path, kind: str) -> List[int]:
    """Indices present on disk for image_N.png / prompt_N.txt, ascending."""
    stem, suffix = ("image", ".png") if kind == "images" else ("prompt", ".txt")
    pattern = re.compile(rf"{stem}_(\d+)\{suffix}$")
    steps = []
    for path in run_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            steps.append(int(match.group(1)))
    return sorted(steps)


def pair_paths(run_dir: Path, kind: str, a: int, b: int) -> Tuple[Path, Path]:
    stem, suffix = ("image", ".png") if kind == "images" else ("prompt", ".txt")
    return run_dir / f"{stem}_{a}{suffix}", run_dir / f"{stem}_{b}{suffix}"


def iter_pairs(run_dir: Path, kind: str, step: Optional[int]) -> Iterator[Tuple[int, int]]:
    """Yield consecutive (a, b) step pairs that are both present on disk.

    A gap in the sequence is not bridged: only (N, N+1) is ever compared, which
    is what "two subsequent images" means. A live run simply yields fewer pairs.
    """
    steps = available_steps(run_dir, kind)
    if step is not None:
        if step in steps and (step + 1) in steps:
            yield step, step + 1
        return
    for a, b in zip(steps, steps[1:]):
        if b == a + 1:
            yield a, b


def anchor_pairs(run_dir: Path, kind: str, anchor: Optional[int]) -> Iterator[Tuple[int, int]]:
    """Yield every present step paired with the FIXED step `anchor`, ascending.

    The consecutive pairs answer "how much did this step move?". These answer "how far
    has the run drifted from step `anchor`?", which is the question a chain of small
    steps cannot settle: each step can look minor while the total travel is major.

    The pair is always ordered (earlier, later) so the "the second was regenerated from
    the first" framing stays true no matter which side the anchor is on.
    """
    if anchor is None:
        return
    steps = available_steps(run_dir, kind)
    if anchor not in steps:
        return
    for step in steps:
        if step != anchor:
            yield (anchor, step) if step > anchor else (step, anchor)


def pairs_for(run_dir: Path, kind: str, relation: str, step: Optional[int],
              anchor: Optional[int]) -> List[Tuple[int, int]]:
    if relation == "consecutive":
        return list(iter_pairs(run_dir, kind, step))
    return list(anchor_pairs(run_dir, kind, anchor))


# ---- model ----------------------------------------------------------------
def load_vlm(model_id: str, device_choice: str, max_new_tokens: int) -> Tuple[Any, str]:
    device, dtype = describe.resolve_device_and_dtype(device_choice)
    vlm = describe.load_vlm_pipeline(model_id=model_id, device=device, dtype=dtype)
    for generation_config in (
        getattr(vlm, "generation_config", None),
        getattr(vlm.model, "generation_config", None),
    ):
        if generation_config is not None:
            generation_config.max_new_tokens = max_new_tokens
            generation_config.max_length = None
    return vlm, device


def call_vlm(vlm: Any, message_variants: List[List[Dict[str, Any]]]) -> str:
    """Try each message shape until one is accepted (mirrors describe.py).

    Transformers' chat schema differs between versions in whether an image is
    carried under "image" or "url", so both are attempted.
    """
    last_error: Optional[Exception] = None
    for messages in message_variants:
        try:
            outputs = vlm(
                text=messages,
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )
            return describe.extract_generated_text(outputs)
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"vision model rejected every message shape. Last error: {last_error}")


def compare_images(vlm: Any, image_a: Any, image_b: Any, text: str) -> str:
    """Both images go in ONE message, in order, so the model can compare them."""
    variants = [
        [{"role": "user", "content": [
            {"type": "image", "image": image_a},
            {"type": "image", "image": image_b},
            {"type": "text", "text": text},
        ]}],
        [{"role": "user", "content": [
            {"type": "image", "url": image_a},
            {"type": "image", "url": image_b},
            {"type": "text", "text": text},
        ]}],
    ]
    return call_vlm(vlm, variants)


def compare_prompts(vlm: Any, text: str) -> str:
    variants = [
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        [{"role": "user", "content": text}],
    ]
    return call_vlm(vlm, variants)


# ---- answer parsing -------------------------------------------------------
def parse_answer(raw: str) -> Dict[str, Any]:
    """Pull the JSON object out of the model's reply, leniently.

    A refusal to parse is recorded rather than raised: one malformed answer in a
    200-run sweep should not lose the other 199.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).strip().lower()
            differences = parsed.get("differences") or []
            if not isinstance(differences, list):
                differences = [str(differences)]
            return {
                "verdict": verdict if verdict in VERDICTS else None,
                "differences": [str(d).strip() for d in differences if str(d).strip()],
                "summary": str(parsed.get("summary", "")).strip(),
                "parse_ok": verdict in VERDICTS,
            }
    return {"verdict": None, "differences": [], "summary": raw.strip()[:500], "parse_ok": False}


# ---- reporting ------------------------------------------------------------
def verdict_label(record: Dict[str, Any]) -> str:
    return record["verdict"] or "UNPARSED"


# One glyph per verdict for the trajectory chain. Not first letters: "minor" and
# "major" both start with M, which made the chain unreadable exactly where it matters.
VERDICT_GLYPH = {"identical": "=", "minor": ".", "moderate": "o", "major": "X"}


def verdict_glyph(record: Dict[str, Any]) -> str:
    return VERDICT_GLYPH.get(verdict_label(record), "?")


def relation_tag(record: Dict[str, Any]) -> str:
    """'con' / 'anc' -- legacy records predate the field and are all consecutive."""
    return "anc" if record.get("relation") == "anchor" else "con"


def print_record(record: Dict[str, Any]) -> None:
    head = (f"  [{record['kind'][:3]}/{relation_tag(record)} "
            f"{record['step_a']:>2}->{record['step_b']:<2}] {verdict_label(record):<9}")
    print(f"{head} {record['summary']}", flush=True)
    for difference in record["differences"]:
        print(f"{'':<8}- {difference}", flush=True)


def print_trajectory(run_name: str, kind: str, relation: str,
                     records: List[Dict[str, Any]]) -> None:
    """The convergence signal: verdicts in step order, plus totals."""
    if not records:
        return
    chain = " ".join(verdict_glyph(r) for r in records)
    counts: Dict[str, int] = {}
    for record in records:
        counts[verdict_label(record)] = counts.get(verdict_label(record), 0) + 1
    tally = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    label = "drift from anchor" if relation == "anchor" else "step-to-step"
    print(f"  {run_name} [{kind}/{relation}] {label} (first -> last): {chain}"
          f"    [= identical  . minor  o moderate  X major  ? unparsed]")
    print(f"  {'':<{len(run_name)}} {tally}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a multi-image VLM what changed between subsequent steps of a run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("runs", type=Path, nargs="+",
                        help="Run directories (shell globs work: bbq_runs/image_*).")
    parser.add_argument("--mode", choices=("images", "prompts", "both"), default="images",
                        help="Compare subsequent images, subsequent prompts, or both (default images).")
    parser.add_argument("--step", type=int, default=None,
                        help="Compare only this pair (N vs N+1). Default: every consecutive pair.")
    parser.add_argument("--anchor", type=int, default=DEFAULT_ANCHOR,
                        help="ALSO compare every step against this FIXED step (default "
                             "%(default)s: scores 3-vs-0, 3-vs-1, 3-vs-2, 3-vs-4, ...). Measures "
                             "total drift from one reference rather than step-to-step movement. "
                             "Roughly DOUBLES the number of VLM calls; --no-anchor turns it off.")
    parser.add_argument("--no-anchor", action="store_true",
                        help="Consecutive pairs only: skip the fixed-step comparison.")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Pin a GPU via CUDA_VISIBLE_DEVICES. Check free VRAM first.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="Execution device. Default: auto.")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL,
                        help=f"VLM id (default {DEFAULT_VISION_MODEL}).")
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE,
                        help=f"Resize longest side before comparing (default {DEFAULT_MAX_SIZE}).")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help=f"Generation cap per comparison (default {DEFAULT_MAX_NEW_TOKENS}).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir for JSONL. Default: <runs-parent>/validation.")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Store the model's raw reply alongside the parsed fields.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo pairs already present in the JSONL (default: skip them).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the pairs that would be compared; load no model.")
    return parser


def record_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """What makes a comparison unique. Both steps are needed now that anchor pairs
    exist: they all share one step_a, so (kind, step_a) would collapse them into one.
    Records written before anchors existed carry no 'relation' and are consecutive."""
    return (record.get("kind"), record.get("relation", "consecutive"),
            record.get("step_a"), record.get("step_b"))


def existing_keys(path: Path) -> set:
    """Comparisons already recorded, so a rerun resumes rather than repeats."""
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(record_key(record))
    return keys


def main() -> int:
    args = build_parser().parse_args()
    if args.no_anchor:
        args.anchor = None   # everything downstream reads args.anchor is None as 'off'

    run_dirs = [r for r in args.runs if r.is_dir()]
    missing = [str(r) for r in args.runs if not r.is_dir()]
    for path in missing:
        print(f"warning: not a directory, skipping: {path}", file=sys.stderr)
    if not run_dirs:
        print("error: no run directory to work on.", file=sys.stderr)
        return 2

    kinds = ("images", "prompts") if args.mode == "both" else (args.mode,)
    relations = ("consecutive",) if args.anchor is None else ("consecutive", "anchor")
    out_dir = args.out or (run_dirs[0].resolve().parent / "validation")

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
        print("nothing to compare.", file=sys.stderr)
        return 1

    anchor_note = "" if args.anchor is None else f"   anchor: step {args.anchor}"
    print(f"runs: {len(run_dirs)}   mode: {args.mode}{anchor_note}   comparisons: {total_pairs}")
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

    print(f"loading {args.vision_model} ...", flush=True)
    vlm, device = load_vlm(args.vision_model, args.device, args.max_new_tokens)
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"loaded on {device}" + (f" (CUDA_VISIBLE_DEVICES={pinned})" if pinned else ""), flush=True)

    done = 0
    failures = 0
    for run_dir, kind, relation, pairs in plan:
        out_path = out_dir / f"{run_dir.name}.{kind}.jsonl"
        seen = set() if args.overwrite else existing_keys(out_path)
        template = TEMPLATES[(kind, relation)]
        records: List[Dict[str, Any]] = []
        print(f"\n== {run_dir.name} [{kind}/{relation}]  {len(pairs)} pair(s) "
              f"-> {out_path.name}", flush=True)

        with out_path.open("a", encoding="utf-8") as handle:
            for a, b in pairs:
                tag = f"{kind[:3]}/{relation[:3]} {a:>2}->{b:<2}"
                if (kind, relation, a, b) in seen:
                    print(f"  [{tag}] already recorded, skipping.", flush=True)
                    continue
                path_a, path_b = pair_paths(run_dir, kind, a, b)
                try:
                    if kind == "images":
                        image_a = describe.load_image(path_a, args.max_size)
                        image_b = describe.load_image(path_b, args.max_size)
                        text = template.format(a=a, b=b, distance=b - a)
                        raw = compare_images(vlm, image_a, image_b, text)
                    else:
                        text_a = path_a.read_text(encoding="utf-8").strip()
                        text_b = path_b.read_text(encoding="utf-8").strip()
                        text = template.format(a=a, b=b, distance=b - a,
                                               text_a=text_a, text_b=text_b)
                        raw = compare_prompts(vlm, text)
                except (ValueError, OSError) as exc:
                    # A half-written PNG from a live run lands here; keep going.
                    print(f"  [{tag}] FAILED: {exc}", file=sys.stderr, flush=True)
                    failures += 1
                    continue

                record: Dict[str, Any] = {
                    "run": run_dir.name,
                    "kind": kind,
                    "relation": relation,
                    "step_a": a,
                    "step_b": b,
                    "distance": b - a,
                    "file_a": path_a.name,
                    "file_b": path_b.name,
                    "model": args.vision_model,
                    **parse_answer(raw),
                }
                if relation == "anchor":
                    record["anchor"] = args.anchor
                if args.keep_raw:
                    record["raw"] = raw
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)
                print_record(record)
                done += 1

        print_trajectory(run_dir.name, kind, relation, records)

    print("\n" + "-" * 72)
    print(f"summary: {done} comparison(s) written, {failures} failed  (under {out_dir})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
