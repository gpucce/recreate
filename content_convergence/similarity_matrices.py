"""Per-experiment similarity matrices for image_*/prompt_* outputs.

For each experiment dir under --runs it saves, in <out>/experiments/<experiment>/:
    images_similarity.csv    N x N cosine similarity between image embeddings
    prompts_similarity.csv   M x M cosine similarity between prompt embeddings
plus matching .npy files. Images are embedded with CLIP; prompts with
Qwen3-Embedding-8B (last-token pooling, up to 32k tokens, no truncation in practice).
Averages over all experiments are saved as <out>/average_<name>.csv/.npy.
Existing tables are skipped unless --recompute is given.
Heatmap PDFs are produced separately by similarity_plots.py.
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-large-patch14"
TEXT_MODEL = "Qwen/Qwen3-Embedding-8B"
TEXT_MAX_LEN = 32768


def idx(p: Path) -> int:
    return int(re.search(r"_(\d+)\.", p.name).group(1))


def save(out: Path, name: str, mat: np.ndarray, labels):
    np.save(out / f"{name}.npy", mat)
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for lab, row in zip(labels, mat):
            w.writerow([lab] + [f"{v:.6f}" for v in row])


@torch.no_grad()
def embed_images(model, proc, paths, dev, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        x = proc(images=ims, return_tensors="pt").to(dev)
        out.append(model.visual_projection(model.vision_model(pixel_values=x["pixel_values"]).pooler_output))
    return norm(torch.cat(out))


@torch.no_grad()
def embed_texts(model, tok, paths, dev, bs=4):
    texts = [p.read_text(errors="replace").strip() for p in paths]
    out = []
    for i in range(0, len(texts), bs):
        x = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                truncation=True, max_length=TEXT_MAX_LEN).to(dev)
        h = model(**x).last_hidden_state[:, -1]  # left padding: last token is the EOS summary
        out.append(h)
    return norm(torch.cat(out))


def norm(t):
    return torch.nn.functional.normalize(t, dim=-1).float().cpu().numpy()


EXP_DIR = "experiments"  # per-experiment tables; aggregates stay in the root
NAMES = ("images_similarity", "prompts_similarity")


def average_tables(out: Path, exps):
    for name in NAMES:
        mats = [(e, np.load(out / EXP_DIR / e / f"{name}.npy")) for e in exps
                if (out / EXP_DIR / e / f"{name}.npy").exists()]
        if not mats:
            continue
        shapes = [m.shape for _, m in mats]
        modal = max(set(shapes), key=shapes.count)
        used = [e for e, m in mats if m.shape == modal]
        if len(used) < len(mats):
            print(f"{name}: averaging {len(used)}/{len(mats)} experiments (shape {modal})")
        avg = np.mean([m for _, m in mats if m.shape == modal], axis=0)
        ref = next(e for e in used)
        with open(out / EXP_DIR / ref / f"{name}.csv") as f:
            labels = next(csv.reader(f))[1:]
        save(out, f"average_{name}", avg, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./data/content_convergence/vlstereoset_runs")
    ap.add_argument("--out", default="./data/similarities/vlstereoset_runs")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute tables that already exist (default: only missing ones)")
    a = ap.parse_args()
    out_root = Path(a.out)

    model = proc = dev = tmodel = tok = None
    exps = []
    for exp in sorted(Path(a.runs).glob("image_*")):
        if not exp.is_dir():
            continue
        imgs = sorted(exp.glob("image_*.png"), key=idx)
        prompts = sorted(exp.glob("prompt_*.txt"), key=idx)
        if not imgs:
            continue
        out = out_root / EXP_DIR / exp.name
        exps.append(exp.name)
        todo = {n for n in NAMES if a.recompute or not (out / f"{n}.npy").exists()}
        if not prompts:
            todo -= {"prompts_similarity"}
        if not todo:
            print("skip", exp.name)
            continue
        out.mkdir(parents=True, exist_ok=True)

        if model is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
            proc = CLIPProcessor.from_pretrained(MODEL)

        ie = embed_images(model, proc, imgs, dev) if "images_similarity" in todo else None
        pe = None
        if prompts and "prompts_similarity" in todo:
            if tmodel is None:
                tok = AutoTokenizer.from_pretrained(TEXT_MODEL, padding_side="left")
                tmodel = AutoModel.from_pretrained(
                    TEXT_MODEL, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32
                ).to(dev).eval()
            pe = embed_texts(tmodel, tok, prompts, dev)
        if "images_similarity" in todo:
            save(out, "images_similarity", ie @ ie.T, [p.stem for p in imgs])
        if "prompts_similarity" in todo:
            save(out, "prompts_similarity", pe @ pe.T, [p.stem for p in prompts])
        print("done", exp.name)

    average_tables(out_root, exps)


if __name__ == "__main__":
    main()
