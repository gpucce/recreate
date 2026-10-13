#!/usr/bin/env python
"""Speed test: run_bbq_loop.py (old) vs run_loop_optimized.py (new) on the same rows.

Every configuration processes the same --num-images rows (same --seed and
--start-index) into its own output dir under --out, so the runs never touch
the real results or each other:

    <out>/old/          run_bbq_loop.py            (one loop.py process per row)
    <out>/new_b1/       run_loop_optimized.py --batch-size 1
    <out>/new_b4/       run_loop_optimized.py --batch-size 4   (one dir per --batch-sizes entry)
    <out>/<name>.log    stdout/stderr of each configuration
    <out>/summary.json  the numbers printed at the end

Configurations are dealt round-robin over --gpus: configurations on different
GPUs run in parallel, those on the same GPU run one after the other. Compare
configurations only on identical, otherwise idle GPUs (the A40s are). Each
GPU is checked for --min-free-mb free VRAM before anything starts.

Reported per configuration:
    wall_s           start to exit of the driver (includes model loads and seeding)
    done             rows with prompt_<steps>.txt
    s_per_image      wall_s / done                      <- the headline number
    step_s_median    median time between consecutive prompt_k files of one row
                     (one render + one describe; at batch B this is the time of a
                     batched step, shared by B rows)
    overhead_s       wall_s minus the time chains spent stepping (B=1 only: model
                     reloads, seeding, process start-up)
    peak_vram_mb     peak memory.used on the GPU during the run, minus what was
                     in use before it started

Outputs are not compared image by image: the renders use seed=None, so two runs
never produce the same images, even with the same script.

Usage:
    ./conda_venv/bin/python content_convergence/speed_test.py --dry-run
    ./conda_venv/bin/python content_convergence/speed_test.py --gpus 0,1 --batch-sizes 1,4
    ./conda_venv/bin/python content_convergence/speed_test.py --summarize data_bkp/speed_test/20261009-120000
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
RECREATE_DIR = HERE.parent
OLD_SCRIPT = HERE / "run_bbq_loop.py"
NEW_SCRIPT = HERE / "run_loop_optimized.py"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Time run_bbq_loop.py against run_loop_optimized.py.")
    p.add_argument("--num-images", type=int, default=64, help="Rows per configuration (default 64).")
    p.add_argument("--steps", type=int, default=16, help="Loop steps per row (default 16).")
    p.add_argument("--gpus", default="0", help="Comma-separated GPU ids to use (default 0).")
    p.add_argument("--batch-sizes", default="1",
                   help="Comma-separated --batch-size values for the new script, one configuration each "
                        "(default 1).")
    p.add_argument("--no-old", action="store_true", help="Skip the old run_bbq_loop.py configuration.")
    p.add_argument("--dataset", default="bbq-v", help="Passed to both scripts (default bbq-v).")
    p.add_argument("--data-dir", type=Path, default=RECREATE_DIR / "data_bkp",
                   help="Passed to both scripts (default <repo>/data_bkp, as the launchers use).")
    p.add_argument("--seed", type=int, default=0, help="Row permutation seed (default 0).")
    p.add_argument("--start-index", type=int, default=0, help="Selection offset (default 0).")
    p.add_argument("--out", type=Path, default=None,
                   help="Root for this test (default <data-dir>/speed_test/<timestamp>). Must not exist yet.")
    p.add_argument("--min-free-mb", type=int, default=35000,
                   help="Refuse to start unless every GPU has this much free VRAM (default 35000).")
    p.add_argument("--py", default=str(RECREATE_DIR / "conda_venv" / "bin" / "python"),
                   help="Python used to run the scripts (default ./conda_venv/bin/python).")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    p.add_argument("--summarize", type=Path, default=None, metavar="ROOT",
                   help="Only recompute and print the summary of an earlier test root.")
    return p


def gpu_memory(gpu: int) -> Optional[tuple[int, int]]:
    """(used_mb, free_mb) of one GPU from nvidia-smi, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        used, free = (int(x) for x in out.split(","))
        return used, free
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class VramPoller(threading.Thread):
    """Track the peak memory.used of one GPU while a configuration runs."""

    def __init__(self, gpu: int, interval: float = 5.0):
        super().__init__(daemon=True)
        self.gpu, self.interval = gpu, interval
        self.peak = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            mem = gpu_memory(self.gpu)
            if mem:
                self.peak = max(self.peak, mem[0])
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join()


def make_configs(args) -> List[Dict[str, Any]]:
    common = ["--dataset", args.dataset, "--data-dir", str(args.data_dir), "--seed", str(args.seed),
              "--start-index", str(args.start_index), "--num-images", str(args.num_images),
              "--steps", str(args.steps), "--max-consecutive-failures", "3"]
    configs = []
    if not args.no_old:
        configs.append(dict(name="old", batch_size=1, cmd=[args.py, str(OLD_SCRIPT), *common, "--fresh"]))
    for b in (int(x) for x in args.batch_sizes.split(",") if x.strip()):
        configs.append(dict(name=f"new_b{b}", batch_size=b,
                            cmd=[args.py, str(NEW_SCRIPT), *common, "--batch-size", str(b),
                                 "--no-run-dir"]))   # analyze() reads <out>/<name>/image_* directly
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    for i, cfg in enumerate(configs):
        cfg["gpu"] = gpus[i % len(gpus)]
        cfg["out"] = args.out / cfg["name"]
        cfg["cmd"] += ["--gpu", str(cfg["gpu"]), "--out", str(cfg["out"])]
    return configs


def run_gpu_queue(configs: List[Dict[str, Any]], root: Path, procs: Dict[str, subprocess.Popen],
                  stopping: threading.Event) -> None:
    """Run the configurations assigned to one GPU, one after the other."""
    for cfg in configs:
        if stopping.is_set():
            return
        baseline = (gpu_memory(cfg["gpu"]) or (0, 0))[0]
        poller = VramPoller(cfg["gpu"])
        poller.start()
        log = (root / f"{cfg['name']}.log").open("w", encoding="utf-8")
        print(f"[{cfg['name']}] start on GPU {cfg['gpu']}: {' '.join(cfg['cmd'])}", flush=True)
        t0 = time.time()
        proc = subprocess.Popen(cfg["cmd"], cwd=str(RECREATE_DIR), stdout=log, stderr=subprocess.STDOUT)
        procs[cfg["name"]] = proc
        rc = proc.wait()
        wall = time.time() - t0
        poller.stop()
        log.close()
        timing = dict(name=cfg["name"], gpu=cfg["gpu"], batch_size=cfg["batch_size"], rc=rc,
                      started=t0, wall_s=round(wall, 1),
                      peak_vram_mb=max(0, poller.peak - baseline) if poller.peak else None)
        (root / f"{cfg['name']}.timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"[{cfg['name']}] exited rc={rc} after {wall / 60:.1f} min", flush=True)


def analyze(root: Path, steps: int) -> List[Dict[str, Any]]:
    """Per-configuration numbers from the .timing.json files and the run dirs' file mtimes."""
    rows = []
    for timing_path in sorted(root.glob("*.timing.json")):
        t = json.loads(timing_path.read_text())
        out = root / t["name"]
        gaps, spans = [], []
        done = 0
        for run_dir in sorted(out.glob("image_*")):
            stamps = []
            k = 1
            while (run_dir / f"prompt_{k}.txt").is_file():
                stamps.append((run_dir / f"prompt_{k}.txt").stat().st_mtime)
                k += 1
            if len(stamps) >= steps:
                done += 1
            gaps += [b - a for a, b in zip(stamps, stamps[1:])]
            if len(stamps) >= 2:
                # stepping time of this row: its steps 2..N plus one median-ish step for step 1
                spans.append(stamps[-1] - stamps[0])
        step_median = statistics.median(gaps) if gaps else None
        overhead = None
        if t["batch_size"] == 1 and step_median and spans:
            stepping = sum(spans) + step_median * len(spans)
            overhead = round(t["wall_s"] - stepping, 1)
        rows.append(dict(
            **t, done=done,
            s_per_image=round(t["wall_s"] / done, 1) if done else None,
            images_per_hour=round(3600 * done / t["wall_s"], 2) if done and t["wall_s"] else None,
            step_s_median=round(step_median, 1) if step_median else None,
            overhead_s=overhead,
        ))
    return rows


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("no finished configurations to summarize.")
        return
    cols = ["name", "gpu", "rc", "done", "wall_s", "s_per_image", "images_per_hour", "step_s_median",
            "overhead_s", "peak_vram_mb"]
    widths = {c: max(len(c), *(len(str(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c)).ljust(widths[c]) for c in cols))
    ref = next((r for r in rows if r["name"] == "old" and r.get("s_per_image")), None)
    if ref:
        for r in rows:
            if r is not ref and r.get("s_per_image"):
                print(f"{r['name']}: {ref['s_per_image'] / r['s_per_image']:.2f}x the throughput of old")


def main() -> int:
    args = build_parser().parse_args()
    if args.summarize:
        meta = json.loads((args.summarize / "test.json").read_text())
        rows = analyze(args.summarize, meta["steps"])
        print_summary(rows)
        return 0

    if args.out is None:
        args.out = args.data_dir / "speed_test" / time.strftime("%Y%m%d-%H%M%S")
    configs = make_configs(args)
    by_gpu: Dict[int, List[Dict[str, Any]]] = {}
    for cfg in configs:
        by_gpu.setdefault(cfg["gpu"], []).append(cfg)

    print(f"speed test: {len(configs)} configuration(s), {args.num_images} images x {args.steps} steps each, "
          f"output under {args.out}")
    for gpu, queue in by_gpu.items():
        print(f"  GPU {gpu}: " + " -> ".join(c["name"] for c in queue))
    for cfg in configs:
        print(f"  [{cfg['name']}] {' '.join(cfg['cmd'])}")
    if args.dry_run:
        print("dry-run: nothing started.")
        return 0

    if args.out.exists():
        print(f"error: {args.out} already exists; pick a new --out.", file=sys.stderr)
        return 2
    for gpu in by_gpu:
        mem = gpu_memory(gpu)
        if mem is None:
            print(f"error: cannot read the memory of GPU {gpu} with nvidia-smi.", file=sys.stderr)
            return 2
        if mem[1] < args.min_free_mb:
            print(f"error: GPU {gpu} has only {mem[1]} MiB free (< --min-free-mb {args.min_free_mb}); "
                  f"is something else running on it?", file=sys.stderr)
            return 2
    args.out.mkdir(parents=True)
    (args.out / "test.json").write_text(json.dumps(
        dict(vars(args), out=str(args.out), data_dir=str(args.data_dir),
             configs=[dict(c, out=str(c["out"])) for c in configs]), indent=2, default=str) + "\n")

    procs: Dict[str, subprocess.Popen] = {}
    stopping = threading.Event()

    def handle_signal(signum, frame):
        if not stopping.is_set():
            print("\nstopping: signalling the running configuration(s)...", file=sys.stderr, flush=True)
        stopping.set()
        for proc in procs.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    threads = [threading.Thread(target=run_gpu_queue, args=(queue, args.out, procs, stopping))
               for queue in by_gpu.values()]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(timeout=1.0)   # short joins keep the main thread responsive to Ctrl-C

    rows = analyze(args.out, args.steps)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\n" + "-" * 72)
    print_summary(rows)
    print(f"summary: {args.out / 'summary.json'}")
    return 0 if rows and all(r["rc"] == 0 for r in rows) and not stopping.is_set() else 1


if __name__ == "__main__":
    sys.exit(main())
