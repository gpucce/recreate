"""Cross-run similarity between two runs of the same experiments.

For every experiment dir (image_*) present in both --runs-a and --runs-b it saves,
in <out>/experiments/<experiment>/:
    images_cross_similarity.csv/.npy    Na x Nb cosine similarity, A image i vs B image j (CLIP)
    prompts_cross_similarity.csv/.npy   Ma x Mb cosine similarity, A prompt i vs B prompt j (Qwen3)
    images_paired.csv / prompts_paired.csv
                                        diagonal: sim(A step t, B step t), t < min(N)
Averages over experiments are saved in <out> as average_<name>.csv/.npy
(only experiments with the modal shape; paired curves are averaged per step).
Both runs are embedded with the same models, so values are comparable.
Existing tables are skipped unless --recompute is given.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similarity_matrices import (MODEL, TEXT_MODEL, embed_images, embed_texts,  # noqa: E402
                                 idx, save)

EXP_DIR = "experiments"
KINDS = ("images", "prompts")


def save_paired(out: Path, name: str, vals: np.ndarray):
    np.save(out / f"{name}.npy", vals)
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "similarity"])
        for t, v in enumerate(vals):
            w.writerow([t, f"{v:.6f}"])


def save_cross(out: Path, name: str, mat: np.ndarray, rows, cols):
    np.save(out / f"{name}.npy", mat)
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + cols)
        for lab, row in zip(rows, mat):
            w.writerow([lab] + [f"{v:.6f}" for v in row])


def average(out: Path, exps):
    for kind in KINDS:
        for suffix in ("cross_similarity", "paired"):
            name = f"{kind}_{suffix}"
            mats = [np.load(out / EXP_DIR / e / f"{name}.npy") for e in exps
                    if (out / EXP_DIR / e / f"{name}.npy").exists()]
            if not mats:
                continue
            if suffix == "paired":
                n = min(len(m) for m in mats)
                avg = np.mean([m[:n] for m in mats], axis=0)
                save_paired(out, f"average_{name}", avg)
            else:
                shapes = [m.shape for m in mats]
                modal = max(set(shapes), key=shapes.count)
                used = [m for m in mats if m.shape == modal]
                if len(used) < len(mats):
                    print(f"{name}: averaging {len(used)}/{len(mats)} experiments (shape {modal})")
                avg = np.mean(used, axis=0)
                labels = [f"{i}" for i in range(modal[0])], [f"{j}" for j in range(modal[1])]
                save_cross(out, f"average_{name}", avg, *labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-a", required=True, help="first run dir containing image_* experiments")
    ap.add_argument("--runs-b", required=True, help="second run dir containing image_* experiments")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    out_root = Path(a.out)

    model = proc = tmodel = tok = None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    exps = []
    for ea in sorted(Path(a.runs_a).glob("image_*")):
        eb = Path(a.runs_b) / ea.name
        if not (ea.is_dir() and eb.is_dir()):
            continue
        files = {}
        for kind, pat in (("images", "image_*.png"), ("prompts", "prompt_*.txt")):
            fa, fb = (sorted(d.glob(pat), key=idx) for d in (ea, eb))
            if fa and fb:
                files[kind] = (fa, fb)
        out = out_root / EXP_DIR / ea.name
        todo = [k for k in files
                if a.recompute or not (out / f"{k}_cross_similarity.npy").exists()]
        exps.append(ea.name)
        if not todo:
            print("skip", ea.name)
            continue
        out.mkdir(parents=True, exist_ok=True)

        for kind in todo:
            fa, fb = files[kind]
            if kind == "images":
                if model is None:
                    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
                    proc = CLIPProcessor.from_pretrained(MODEL)
                ea_, eb_ = (embed_images(model, proc, f, dev) for f in (fa, fb))
            else:
                if tmodel is None:
                    tok = AutoTokenizer.from_pretrained(TEXT_MODEL, padding_side="left")
                    tmodel = AutoModel.from_pretrained(
                        TEXT_MODEL, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32
                    ).to(dev).eval()
                ea_, eb_ = (embed_texts(tmodel, tok, f, dev) for f in (fa, fb))
            mat = ea_ @ eb_.T
            save_cross(out, f"{kind}_cross_similarity", mat,
                       [p.stem for p in fa], [p.stem for p in fb])
            save_paired(out, f"{kind}_paired", np.diag(mat).copy())
        print("done", ea.name)

    average(out_root, exps)


if __name__ == "__main__":
    main()
