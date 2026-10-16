#!/usr/bin/env python
"""Faster drop-in for run_bbq_loop.py: same flags, same outputs, models loaded once.

Differences from run_bbq_loop.py (selection, sharding, run-dir layout,
source.json and the manifest are unchanged; it reuses that module's helpers):

  * In-process: the VLM and the diffusion model are loaded ONCE per process
    instead of once per row (run_bbq_loop.py spawns a fresh loop.py for every
    row, ~20-25 s of torch import + model load each time). --subprocess
    restores that behaviour.
  * Lazy seeding: a row's image_0.png is (re)seeded right before its loop
    runs, not for the whole shard up front, so a stopped shard never leaves
    wiped-but-unrun dirs behind. Finished rows are skipped without
    reading their image, and the size check reads only the image header.
  * --batch-size B (default 1): advance B chains in lockstep, one batched
    render and one batched VLM call per step. VLM decoding at batch 1 is
    memory-bound (~half of each ~30 s step), so B=2-4 raises throughput.
    A slot is refilled as soon as its chain finishes. B>1 can change
    greedy VLM outputs slightly (padding), so keep B=1 to stay comparable
    with existing runs; batch_size is recorded in source.json.

A chain step matches loop.py: render image_k from prompt_k (unless image_k
already exists), then describe image_k -> prompt_{k+1}. SIGINT/SIGTERM stop
after the current step. Unfinished runs are never resumed: on the next
start every selected row without prompt_<steps>.txt is wiped and redone from
image_0 (--keep and --fresh are accepted but behave the same), so a run's
steps always share one batch size and setting.

Output layout: each job writes into its own run directory <out>/<run-name>/,
where the run name defaults to the job start time (YYYYmmdd-HHMMSS), and
writes <out>/<run-name>/config.json describing the experimental setup
(selection, models, render and describe settings, git commit, command line).
Give parallel shards the same --run-name so they share one run directory and
one config.json; a job that finds a config.json whose "experiment" settings
differ from its own refuses to run. Rerun with an existing --run-name to
finish an interrupted job. --no-run-dir writes straight into --out as before.

Usage:
    ./conda_venv/bin/python content_convergence/run_loop_optimized.py --num-images 4 --steps 2 --gpu 0 --dry-run
    ./conda_venv/bin/python content_convergence/run_loop_optimized.py --num-images 8 --steps 16 --gpu 3 \\
        --batch-size 4 --out data/bbq_runs_smoke                  # -> data/bbq_runs_smoke/<start time>/
    ./conda_venv/bin/python content_convergence/run_loop_optimized.py --run-name 20261012-153000 ...  # resume / share
"""
from __future__ import annotations

import ast
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
from PIL import Image

import run_bbq_loop as rbl
from run_bbq_loop import ID_KEY

CONFIG_NAME = "config.json"
RUN_NAME_FORMAT = "%Y%m%d-%H%M%S"
# Module-level constants of create.py / describe.py that loop.py renders and describes with.
CREATE_CONSTANTS = ("DEFAULT_STEPS", "DEFAULT_GUIDANCE_SCALE")
DESCRIBE_CONSTANTS = ("DEFAULT_MAX_SIZE", "DEFAULT_MAX_NEW_TOKENS", "PROMPT_REQUEST")


@dataclass
class Planned:
    row: int
    work_dir: Path
    record: Dict[str, Any]
    raw_image: Any          # undecoded column value: {"bytes", "path"} dict or a path string
    complete: bool


@dataclass
class Chain:
    row: int
    work_dir: Path
    record: Dict[str, Any]
    idx: int
    remaining: int
    started: float = field(default_factory=time.time)


def open_image_lazy(value, data_dir: Path) -> Image.Image:
    """Open without decoding pixels (size comes from the header). Call .load() before use."""
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        return Image.open(value["path"])
    if isinstance(value, (str, Path)):
        return Image.open(data_dir / value)
    return value


def build_parser():
    p = rbl.build_parser()
    p.description = "Run the recreate loop over a random dataset sample, models loaded once per process."
    p.set_defaults(steps=16)   # the length of the existing runs; a larger value would wipe and redo them
    for action in p._actions:
        if action.dest == "steps":
            action.help = "Loop iterations per image (default 16, the length of the existing runs)."
    p.add_argument("--batch-size", type=int, default=1,
                   help="Chains advanced in lockstep per GPU (default 1 = same outputs as run_bbq_loop.py).")
    p.add_argument("--subprocess", action="store_true",
                   help="Run each row through a separate loop.py process, as run_bbq_loop.py does.")
    p.add_argument("--render-steps", type=int, default=None,
                   help="Diffusion steps per image (default create.DEFAULT_STEPS). Distilled models such as "
                        "FLUX.2-klein are meant to run with few steps.")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="Diffusion guidance scale (default create.DEFAULT_GUIDANCE_SCALE).")
    p.add_argument("--vlm-backend", choices=["transformers", "vllm"], default="transformers",
                   help="How the VLM runs: the transformers pipeline (default, as in loop.py) or vLLM, which "
                        "is much faster but needs an env with vllm installed (./vllm_venv).")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4,
                   help="Share of the first GPU's memory vLLM may take for weights + KV cache (default 0.4, "
                        "leaving room for the diffusion model).")
    p.add_argument("--vllm-max-model-len", type=int, default=8192,
                   help="vLLM context length (default 8192: one 512px image, the request and 2048 new tokens).")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1,
                   help="Split the VLM over this many GPUs of --gpus with vLLM (use instead of --vlm-max-memory).")
    p.add_argument("--vlm-dtype", choices=["float16", "bfloat16"], default=None,
                   help="VLM weight dtype (default describe.py's choice, float16 on CUDA). Use bfloat16 for "
                        "models released in bf16, which can overflow in float16.")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPUs this process may use, set as CUDA_VISIBLE_DEVICES "
                        "(default: --gpu). Needed when the models do not fit on one card.")
    p.add_argument("--vlm-max-memory", default=None,
                   help="Split the VLM over the visible GPUs, filling them in order, e.g. '0:38GiB,1:24GiB' "
                        "(indices count within --gpus). Default: the whole VLM on the first GPU.")
    p.add_argument("--image-device", default="cuda",
                   help="Device of the diffusion pipeline, e.g. cuda:1 for the second GPU of --gpus "
                        "(default cuda, the first).")
    p.add_argument("--run-name", default=None,
                   help="Run directory under --out (default: the job start time, YYYYmmdd-HHMMSS). Pass the "
                        "same name to every parallel shard, or an existing one to finish an interrupted job.")
    p.add_argument("--no-run-dir", action="store_true",
                   help="Write the image_* dirs straight into --out, without a per-job run directory.")
    return p


def module_constants(path: Path, names) -> Dict[str, Any]:
    """Read literal module-level constants (e.g. create.DEFAULT_STEPS) without importing
    the module, which would pull in torch. `"..." .strip()` values are supported."""
    found: Dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
            continue
        value = node.value
        strip = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                 and value.func.attr == "strip" and not value.args)
        try:
            literal = ast.literal_eval(value.func.value if strip else value)
        except ValueError:
            continue
        found[node.targets[0].id] = literal.strip() if strip else literal
    return found


def git_info() -> Dict[str, Any]:
    def git(*cmd: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *cmd], cwd=str(rbl.RECREATE_DIR), capture_output=True,
                                  text=True, timeout=10, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=no")   # unstripped: "XY path" columns
    modified = [line[3:] for line in (status or "").splitlines() if line[3:].endswith((".py", ".sh"))]
    commit, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    return {"commit": commit and commit.strip(), "branch": branch and branch.strip(),
            "dirty": None if status is None else bool(status), "modified_code": modified}


def resolve_run_dir(args) -> None:
    """Point args.out at this job's run directory, <out>/<run-name>, named after the start time by default."""
    args.started_at = datetime.now().astimezone()
    args.out_root = args.out
    if args.no_run_dir:
        args.run_name = None
        return
    if not args.run_name:
        args.run_name = args.started_at.strftime(RUN_NAME_FORMAT)
    if Path(args.run_name).name != args.run_name:
        raise SystemExit(f"error: --run-name must be a plain directory name, got {args.run_name!r}")
    args.out = args.out / args.run_name


def build_config(args) -> Dict[str, Any]:
    """The experimental setup of this job. Everything under "experiment" must match across
    the shards and resumed jobs that write into one run directory."""
    describe = module_constants(rbl.RECREATE_DIR / "describe.py", DESCRIBE_CONSTANTS)
    experiment = {
        "dataset": args.dataset,
        "dataset_path": (args.data_dir / rbl.DATASETS[args.dataset]["dir"]).resolve().as_posix(),
        "selection": "sequential" if args.sequential else "random",
        "seed": args.seed,
        "start_index": args.start_index,
        "num_images": args.num_images,
        "min_source_size": args.min_source_size,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "vision_model": args.vision_model,
        "image_model": args.image_model,
        "width": args.width,
        "height": args.height,
        "render": {"diffusion_steps": args.render_steps,
                   "guidance_scale": args.guidance_scale,
                   "seed": None},
        "describe": {"backend": args.vlm_backend,
                     # vLLM turns off reasoning for chat templates that have it (e.g. Qwen3.8) so no
                     # <think> text reaches the prompt files; templates without the switch ignore it.
                     "thinking": "off" if args.vlm_backend == "vllm" else "model default",
                     "vlm_dtype": args.vlm_dtype or "float16",
                     "max_image_size": describe.get("DEFAULT_MAX_SIZE"),
                     "max_new_tokens": describe.get("DEFAULT_MAX_NEW_TOKENS"),
                     "prompt_request": describe.get("PROMPT_REQUEST")},
    }
    return {
        "run_name": args.run_name,
        "run_dir": args.out.resolve().as_posix(),
        "started_at": args.started_at.isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "mode": "subprocess per row" if args.subprocess else "in-process",
        # Shard split and GPU layout decide who renders which row, not what is rendered, so they
        # may change between the jobs of one run (e.g. a resume on fewer GPUs).
        "placement": {"num_shards": args.num_shards, "gpus": args.gpus, "vlm_max_memory": args.vlm_max_memory,
                      "vllm": ({"gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                                "max_model_len": args.vllm_max_model_len,
                                "tensor_parallel_size": args.vllm_tensor_parallel_size}
                               if args.vlm_backend == "vllm" else None),
                      "image_device": args.image_device},
        "command": [sys.executable, *sys.argv],
        "host": socket.gethostname(),
        "git": git_info(),
        "experiment": experiment,
    }


def write_config(args, config: Dict[str, Any]) -> Path:
    """Create <run dir>/config.json, or check an existing one (written by a parallel shard or an
    earlier job under the same --run-name) has the same "experiment" settings."""
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / CONFIG_NAME
    tmp = path.with_name(f".{CONFIG_NAME}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        os.link(tmp, path)   # atomic create-if-absent: a parallel shard never sees a half-written file
        return path
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")).get("experiment", {})
    ours = json.loads(json.dumps(config["experiment"]))
    diff = sorted(k for k in existing.keys() | ours.keys() if existing.get(k) != ours.get(k))
    if diff:
        raise SystemExit(f"error: {path} was written for a different setup ({', '.join(diff)} differ); "
                         "use another --run-name.")
    return path


def plan_runs(args) -> List[Planned]:
    """Selection + metadata + header-only size check, mirroring run_bbq_loop.prepare_runs
    but without decoding, wiping or seeding anything."""
    cfg = rbl.DATASETS[args.dataset]
    image_column, id_column, category_column = cfg["image_column"], cfg["id_column"], cfg["category_column"]
    split_ds, split_name = rbl.load_split(args.data_dir / cfg["dir"])
    meta_ds = split_ds.remove_columns([image_column])
    img_ds = split_ds.select_columns([image_column])
    if isinstance(split_ds.features[image_column], datasets.Image):
        img_ds = img_ds.cast_column(image_column, datasets.Image(decode=False))
    n = len(split_ds)
    picks = rbl.select_rows(n, args.start_index, args.num_images, args.seed, args.sequential,
                            args.shard, args.num_shards)
    mode = "sequential" if args.sequential else f"random (seed {args.seed})"
    if not picks:
        print(f"note: --start-index {args.start_index} selects nothing (split has {n} rows, {mode}).",
              file=sys.stderr)
        return []
    print(f"split: {split_name!r}  (row count: {n})  ->  {mode}, "
          f"selection positions {args.start_index}..{args.start_index + args.num_images - 1}, "
          f"shard {args.shard}/{args.num_shards}  ({len(picks)} image(s))")

    planned: List[Planned] = []
    for selection_pos, row_idx in picks:
        work_dir = args.out / f"image_{row_idx:05d}"
        meta = rbl.decode_metadata(split_ds, meta_ds[row_idx])
        raw_image = img_ds[row_idx][image_column]
        shown = raw_image if isinstance(raw_image, str) else ""
        complete = not args.redo_complete and rbl.run_is_complete(work_dir, args.steps)
        size = None
        if complete:
            try:  # keep the manifest record as before: reuse the size already recorded
                size = json.loads((work_dir / "source.json").read_text(encoding="utf-8")).get("source_size")
            except (OSError, ValueError):
                size = None
        else:
            try:
                with open_image_lazy(raw_image, args.data_dir) as im:
                    size = list(im.size)
            except Exception as exc:
                print(f"[row {row_idx:05d}] SKIPPED: cannot open source image {shown!r}: {exc}",
                      file=sys.stderr, flush=True)
                continue
            if min(size) < args.min_source_size:
                print(f"[row {row_idx:05d}] SKIPPED: source image {shown} is {size[0]}x{size[1]} "
                      f"(< --min-source-size {args.min_source_size})", file=sys.stderr, flush=True)
                continue
        if id_column is None:
            meta[ID_KEY] = Path(raw_image).stem
        elif id_column != ID_KEY:
            meta[ID_KEY] = meta[id_column]
        if isinstance(raw_image, str):
            meta["source_image"] = raw_image

        record: Dict[str, Any] = {
            "dir": work_dir.name,
            "row": row_idx,
            "selection_position": selection_pos,
            "selection": "sequential" if args.sequential else "random",
            "seed": args.seed,
            "dataset_name": args.dataset,
            "dataset": (args.data_dir / cfg["dir"]).as_posix(),
            "split": split_name,
            "num_rows": n,
            "image_column": image_column,
            "source_size": size,
            **meta,
        }
        if args.batch_size > 1:
            record["batch_size"] = args.batch_size
        head = f"[row {row_idx:05d}] id={record.get(ID_KEY)} category={record.get(category_column)} -> "
        size_str = f"{size[0]}x{size[1]}" if size else "?"
        if complete:
            print(f"{head}{work_dir} already has prompt_{args.steps}.txt, skipping")
        elif args.dry_run:
            if not work_dir.exists():
                action = "would seed"
            else:
                action = f"would wipe ({rbl.done_steps(work_dir)}/{args.steps} steps done) and reseed"
            print(f"{head}{action} {work_dir}/image_0.png  (source {size_str})")
        planned.append(Planned(row_idx, work_dir, record, raw_image, complete))
    return planned


def seed_run(p: Planned, args) -> bool:
    """Wipe any partial run and seed image_0.png + source.json just before the row runs.

    Partial runs are never resumed (even under --keep): a run is always produced in one go,
    so all of its steps share the same batch size and settings."""
    try:
        if p.work_dir.exists():
            shutil.rmtree(p.work_dir)
        with open_image_lazy(p.raw_image, args.data_dir) as im:
            im.load()
            rbl.save_seed(im, p.work_dir / "image_0.png")
    except Exception as exc:
        (p.work_dir / "image_0.png").unlink(missing_ok=True)
        print(f"[row {p.row:05d}] SKIPPED: cannot seed {p.work_dir}/image_0.png: {exc}",
              file=sys.stderr, flush=True)
        return False
    (p.work_dir / "source.json").write_text(json.dumps(p.record, indent=2) + "\n", encoding="utf-8")
    return True


class Runner:
    """Holds both models and advances a set of chains one loop step at a time."""

    def __init__(self, args):
        import torch
        import describe
        import create
        self.torch, self.describe, self.create = torch, describe, create
        self.args = args
        print("Loading vision model (describe)...", flush=True)
        vlm_device, vlm_dtype = describe.resolve_device_and_dtype("cuda")
        vlm_dtype = getattr(torch, args.vlm_dtype) if args.vlm_dtype else vlm_dtype
        if args.vlm_backend == "vllm":
            # Loaded before the diffusion model: vLLM sizes its KV cache from the memory free at startup.
            # CUDA is already initialized in this process, so vLLM's engine process must be spawned, not forked.
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            from vllm import LLM
            self.vlm = LLM(model=args.vision_model, dtype=str(vlm_dtype).removeprefix("torch."),
                           gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                           max_model_len=args.vllm_max_model_len,
                           tensor_parallel_size=args.vllm_tensor_parallel_size,
                           # Never more than batch_size requests at once; the default (256) makes hybrid
                           # models like Qwen3.8 reserve recurrent state for 256 sequences (~24 GiB).
                           max_num_seqs=args.batch_size,
                           limit_mm_per_prompt={"image": 1, "video": 0}, trust_remote_code=True)
            # Same sampling as the transformers pipeline: the model's generation_config.json
            # (Qwen3-VL samples: temperature 0.7, top_k 20, top_p 0.8), capped at describe's token limit.
            self.sampling = self.vlm.get_default_sampling_params()
            self.sampling.max_tokens = describe.DEFAULT_MAX_NEW_TOKENS
            vlm_device = f"vllm, tensor parallel {args.vllm_tensor_parallel_size}"
        elif args.vlm_max_memory:
            # Fill the GPUs in order up to each cap, so the split is predictable.
            self.vlm = describe.load_vlm_pipeline(model_id=args.vision_model, device=vlm_device, dtype=vlm_dtype,
                                                  device_map="sequential", max_memory=args.vlm_max_memory)
            vlm_device = f"cuda {args.vlm_max_memory}"
        else:
            self.vlm = describe.load_vlm_pipeline(model_id=args.vision_model, device=vlm_device, dtype=vlm_dtype)
        print(f"Vision model loaded: {args.vision_model} on {vlm_device}", flush=True)
        print("Loading image model (create)...", flush=True)
        # dtype comes from plain "cuda": resolve_device_and_dtype gives fp32 to any other device string.
        _, create_dtype = create.resolve_device_and_dtype("cuda", args.image_model)
        self.create_device = args.image_device
        self.pipe = create.load_text2image_pipeline(model_id=args.image_model, device=self.create_device,
                                                    dtype=create_dtype)
        print(f"Image model loaded: {args.image_model} on {self.create_device}", flush=True)
        if args.batch_size > 1 and args.vlm_backend == "transformers":
            tok = getattr(getattr(self.vlm, "processor", None), "tokenizer", None) or getattr(self.vlm, "tokenizer", None)
            if tok is not None:
                tok.padding_side = "left"   # decoder-only batched generation needs left padding

    def _oom(self, exc: Exception) -> bool:
        if isinstance(exc, self.torch.cuda.OutOfMemoryError):
            self.torch.cuda.empty_cache()
            return True
        return False

    def render(self, prompts: List[str]) -> List[Any]:
        with self.torch.no_grad():
            return self.create.generate_images(
                pipe=self.pipe, prompts=prompts, model_id=self.args.image_model, device=self.create_device,
                steps=self.args.render_steps, guidance_scale=self.args.guidance_scale,
                seed=None, width=self.args.width, height=self.args.height,
            )

    def describe_images(self, images: List[Any]) -> List[str]:
        d = self.describe
        if self.args.vlm_backend == "vllm":
            chats = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                                   {"type": "text", "text": d.PROMPT_REQUEST}]}]
                     for im in images]
            outputs = self.vlm.chat(chats, self.sampling, use_tqdm=False,
                                    chat_template_kwargs={"enable_thinking": False})
            prompts = [d.clean_prompt(out.outputs[0].text) for out in outputs]
            if len(prompts) != len(images) or not all(prompts):
                raise ValueError("vLLM returned a missing or empty prompt")
            return prompts
        with self.torch.no_grad():
            if len(images) == 1:
                return [d.generate_reconstruction_prompt(images[0], self.vlm)]
            chats = [[{"role": "user", "content": [{"type": "image", "image": im},
                                                   {"type": "text", "text": d.PROMPT_REQUEST}]}]
                     for im in images]
            outputs = self.vlm(text=chats, batch_size=len(chats), return_full_text=False,
                               clean_up_tokenization_spaces=False)
        prompts = [d.clean_prompt(d.extract_generated_text(out)) for out in outputs]
        if len(prompts) != len(images) or not all(prompts):
            raise ValueError("batched VLM call returned a missing or empty prompt")
        return prompts

    def _batched(self, fn, items: List[Any], chains: List[Chain], what: str):
        """Run fn on the whole batch; on failure retry one by one so a single bad chain
        is isolated. Returns (results, failed_chains) with results aligned to survivors."""
        try:
            return list(zip(chains, fn(items))), []
        except Exception as exc:
            self._oom(exc)
            if len(items) == 1:
                print(f"[row {chains[0].row:05d}] {what} failed: {exc!r}", file=sys.stderr, flush=True)
                return [], chains
            print(f"batched {what} failed ({exc!r}); retrying one by one", file=sys.stderr, flush=True)
        ok, failed = [], []
        for chain, item in zip(chains, items):
            try:
                ok.append((chain, fn([item])[0]))
            except Exception as exc:
                self._oom(exc)
                print(f"[row {chain.row:05d}] {what} failed: {exc!r}", file=sys.stderr, flush=True)
                failed.append(chain)
        return ok, failed

    def step(self, chains: List[Chain], stopping) -> List[Chain]:
        """One loop.py iteration for every chain. Returns the chains that failed."""
        failed: List[Chain] = []
        to_render = [c for c in chains if not (c.work_dir / f"image_{c.idx}.png").exists()]
        if to_render:
            missing = [c for c in to_render if not (c.work_dir / f"prompt_{c.idx}.txt").is_file()]
            for c in missing:
                print(f"[row {c.row:05d}] neither image_{c.idx}.png nor prompt_{c.idx}.txt exists",
                      file=sys.stderr, flush=True)
            failed += missing
            to_render = [c for c in to_render if c not in missing]
            prompts = [self.create.read_prompt(c.work_dir / f"prompt_{c.idx}.txt") for c in to_render]
            if to_render:
                print(f"  rendering {len(to_render)}: " + ", ".join(f"{c.row:05d}@{c.idx}" for c in to_render),
                      flush=True)
                done, bad = self._batched(self.render, prompts, to_render, "render")
                failed += bad
                for c, image in done:
                    image.save(c.work_dir / f"image_{c.idx}.png")
        if stopping():
            return failed   # like loop.py: stop after the render, the describe redoes it on resume
        to_describe = [c for c in chains if c not in failed]
        images = [self.describe.load_image(c.work_dir / f"image_{c.idx}.png", self.describe.DEFAULT_MAX_SIZE)
                  for c in to_describe]
        if to_describe:
            print(f"  describing {len(to_describe)}: " + ", ".join(f"{c.row:05d}@{c.idx}" for c in to_describe),
                  flush=True)
        done, bad = self._batched(self.describe_images, images, to_describe, "describe")
        failed += bad
        for c, prompt in done:
            (c.work_dir / f"prompt_{c.idx + 1}.txt").write_text(prompt + "\n", encoding="utf-8")
            c.idx += 1
            c.remaining -= 1
        return failed


def main() -> int:
    args = build_parser().parse_args()
    if args.count_rows:
        print(len(rbl.load_split(args.data_dir / rbl.DATASETS[args.dataset]["dir"])[0]))
        return 0
    if args.out is None:
        args.out = args.data_dir / rbl.DATASETS[args.dataset]["default_out"]
    args.batch_size = max(1, args.batch_size)
    render_defaults = module_constants(rbl.RECREATE_DIR / "create.py", CREATE_CONSTANTS)
    if args.render_steps is None:
        args.render_steps = render_defaults["DEFAULT_STEPS"]
    if args.guidance_scale is None:
        args.guidance_scale = render_defaults["DEFAULT_GUIDANCE_SCALE"]
    args.gpus = args.gpus or str(args.gpu)
    if args.vlm_max_memory:
        try:
            args.vlm_max_memory = {int(i): mem.strip() for i, mem in
                                   (part.split(":") for part in args.vlm_max_memory.split(","))}
        except ValueError:
            raise SystemExit(f"error: --vlm-max-memory wants 'GPU:SIZE,...', got {args.vlm_max_memory!r}")
    if args.vlm_backend == "vllm" and args.vlm_max_memory:
        raise SystemExit("error: with --vlm-backend vllm, split the VLM with --vllm-tensor-parallel-size.")
    if args.subprocess and (args.gpus != str(args.gpu) or args.vlm_max_memory or args.image_device != "cuda"
                            or args.vlm_dtype or args.vlm_backend != "transformers"
                            or args.render_steps != render_defaults["DEFAULT_STEPS"]
                            or args.guidance_scale != render_defaults["DEFAULT_GUIDANCE_SCALE"]):
        raise SystemExit("error: --subprocess runs loop.py, which takes none of --gpus, --vlm-max-memory, "
                         "--image-device, --vlm-dtype, --vlm-backend, --render-steps, --guidance-scale.")
    if not (args.dry_run or args.subprocess) and os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpus:
        # Imports above (datasets) can pull in torch, and CUDA reads CUDA_VISIBLE_DEVICES only
        # once, so setting it later is unreliable: restart the process with it pinned instead.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

    resolve_run_dir(args)
    config = build_config(args)
    print(f"run dir: {args.out}")
    if not args.dry_run:
        print(f"config: {write_config(args, config)}")

    planned = plan_runs(args)
    pending = [p for p in planned if not p.complete]
    skipped = len(planned) - len(pending)
    if args.dry_run:
        mode = "subprocess per row" if args.subprocess else f"in-process, batch size {args.batch_size}"
        print(f"\ndry-run: {len(pending)} to run ({mode}), {skipped} already complete; "
              f"CUDA_VISIBLE_DEVICES={args.gpus}, {args.vision_model} + {args.image_model}, "
              f"{args.width}x{args.height}, {args.steps} steps, {args.render_steps} diffusion steps at "
              f"guidance {args.guidance_scale}. No GPU, no models loaded.")
        print(f"would write {args.out / CONFIG_NAME}:\n{json.dumps(config, indent=2)}")
        return 0
    if planned:
        print(f"manifest: {rbl.write_manifest(args, [p.record for p in planned])}")

    stop = False

    def handle_signal(signum, frame):
        nonlocal stop
        if not stop:
            print("\nReceived signal, stopping after the current step...", file=sys.stderr, flush=True)
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    max_fail = max(1, args.max_consecutive_failures)
    failures: List[int] = []
    consecutive_fail = 0
    ok = 0
    aborted = False

    def note_failure(row: int, why: str) -> bool:
        nonlocal consecutive_fail
        failures.append(row)
        consecutive_fail += 1
        print(f"[row {row:05d}] {why} ({consecutive_fail}/{max_fail} consecutive failure(s))",
              file=sys.stderr, flush=True)
        return consecutive_fail >= max_fail

    queue = list(pending)

    def next_chain() -> Optional[Chain]:
        while queue and not stop:
            p = queue.pop(0)
            if not seed_run(p, args):
                continue
            print(f"[row {p.row:05d}] id={p.record.get(ID_KEY)} seeded {p.work_dir}", flush=True)
            return Chain(p.row, p.work_dir, p.record, idx=0, remaining=args.steps)
        return None

    if args.subprocess:
        kwargs = dict(steps=args.steps, vision_model=args.vision_model, image_model=args.image_model,
                      width=args.width, height=args.height)
        while (chain := next_chain()) is not None:
            rc = rbl.run_loop(chain.work_dir, args.gpu, **kwargs)
            if rc == 0:
                ok += 1
                consecutive_fail = 0
            elif note_failure(chain.row, f"loop exited rc={rc}"):
                aborted = True
                break
    elif pending:
        sys.path.insert(0, str(rbl.RECREATE_DIR))
        runner = Runner(args)
        active: List[Chain] = []
        rounds = 0
        while not stop:
            while len(active) < args.batch_size and (chain := next_chain()) is not None:
                active.append(chain)
            if not active:
                break
            t0 = time.time()
            for c in [c for c in active if c.remaining <= 0]:   # --steps 0: nothing to do
                active.remove(c)
                ok += 1
            failed = runner.step([c for c in active], lambda: stop) if active else []
            rounds += 1
            print(f"  step {rounds}: {len(active)} chain(s) in {time.time() - t0:.1f}s", flush=True)
            for c in failed:
                active.remove(c)
                if note_failure(c.row, "chain failed"):
                    aborted = True
            if aborted:
                print("aborting: too many consecutive failures (likely a systemic problem).", file=sys.stderr)
                break
            for c in [c for c in active if c.remaining <= 0]:
                active.remove(c)
                ok += 1
                consecutive_fail = 0
                print(f"[row {c.row:05d}] done ({args.steps} steps, {time.time() - c.started:.0f}s)", flush=True)
        if stop and active:
            print(f"stopped with {len(active)} partial run(s): "
                  f"{[c.row for c in active]} (rerun to redo them from scratch)", file=sys.stderr)

    print("\n" + "-" * 72, flush=True)
    print(f"summary: {len(planned)} scheduled, {skipped} already complete (skipped), "
          f"{ok} OK, {len(failures)} failed  (under {args.out})")
    if failures:
        print(f"FAILED rows: {failures} did not complete cleanly.", file=sys.stderr)
        return 1
    if stop:
        print("stopped before every selected row ran.", file=sys.stderr)
        return 1
    print("OK: all loops completed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
