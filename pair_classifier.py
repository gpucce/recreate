"""Pairwise classifier: is text B a *direct* AI rewrite of text A, or the result of multiple rewrites?

Each row of a generation CSV is a chain: original -> iteration_00 -> iteration_01 -> ...
For ACL the original column (the paper) is dropped and the chain starts at iteration_00 (the first research plan):
the paper is converted into a plan once, after which plans and reviews alternate.
Pairs (chain[t], chain[t + gap]) are labelled:
    1 (direct)   if gap == pos_gap
    0 (multiple) if gap in {pos_gap + 2, pos_gap + 4, ..., max_gap}
The generation loop alternates formats (e.g. abstract -> proposal -> abstract for ERC grants), so with the
default even gaps (pos_gap=2, negatives 4, 6, ..., max_gap) both texts in a pair are always of the same type:
a "direct copy" is the next text of the same type (one round trip through the other format), while a
"multiple reuses" pair is two or more round trips apart. Negative gaps always keep the same parity as pos_gap,
so the type combination is identical in both classes and cannot be used as a shortcut
(an odd pos_gap, e.g. 1, instead compares texts of different types).
--text-type restricts pairs to one type: "original" (positions 0, 2, 4, ...: ERC abstracts / ACL research plans,
the default), "alternate" (positions 1, 3, 5, ...: ERC full proposals / ACL reviews) or "both".

For every document and every negative gap we pick a random source position t (where t + gap fits in the chain)
and add one negative pair (t, t + gap) and one positive pair (t, t + pos_gap) sharing the same source text.
All negative gaps are thus equally represented, the labels are balanced, and the source text carries no
label information.
With --reverse, every pair is also added with its two texts swapped, so that text A comes later in the chain than
text B, and the task becomes 4-way:
    1 (direct)            B = chain[t + pos_gap]          0 (multiple)          B = chain[t + gap], gap > pos_gap
    2 (reversed direct)   A = chain[t + pos_gap], B = chain[t]
    3 (reversed multiple) A = chain[t + gap],     B = chain[t], gap > pos_gap
Reversed pairs are exact swaps of the forward ones, so the same texts appear in both directions and only their
order tells forward from reversed. Reversed pairs are stored with a negative gap (e.g. -2, -4).
Train/val/test splits are done by document (row) to avoid leakage across chains.

--lora fine-tunes a large decoder LM (e.g. mistralai/Ministral-3-14B-Instruct-2512-BF16) instead of a BERT-style
encoder: the frozen model is loaded in bf16 with gradient checkpointing and only LoRA adapters plus the
classification head are trained (the prediction is read from the last token). Each text gets a "Text A:" /
"Text B:" header instead of BERT's [CLS] / [SEP]. For multimodal checkpoints (config with a text_config) only the
language model is loaded. The best checkpoint is saved as a PEFT adapter in --output-dir.
Models without a sequence-classification class in transformers (e.g. Qwen/Qwen3-VL-Embedding-2B, google/gemma-4-E2B)
get a LastTokenClassifier: the LoRA-adapted text model plus a linear head on the hidden state of the last
non-padding token (the pooling Qwen3-VL-Embedding is trained with); the head is saved as score.pt next to the adapter.
--smoke-test N trains N steps and evaluates N batches, then prints the projected duration of the full run and exits.
"""
import os
import json
import random
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedTokenizerFast, get_linear_schedule_with_warmup)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES

LABEL_NAMES = {0: "multiple reuses", 1: "direct copy", 2: "reversed direct copy", 3: "reversed multiple reuses"}

DEFAULT_DATA = (
    "/raid/datasets/content_convergence/da_mettere_in_homeprogetto/outputs/textonly-generations/"
    "erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv"
)


def chain_columns(df, drop_original):
    """Columns forming the chain, in order. Position 0 is 'original', or 'iteration_00' if the original is dropped."""
    return ([] if drop_original else ["original"]) + sorted(c for c in df.columns if c.startswith("iteration_"))


def build_pairs(df, cols, doc_ids, pos_gap, max_gap, text_type, rng, reverse=False):
    neg_gaps = list(range(pos_gap + 2, max_gap + 1, 2))
    assert neg_gaps, "max_gap must be >= pos_gap + 2"
    # chain positions 0, 2, 4, ... have the type of the first text (ERC: abstracts, ACL: research plans);
    # positions 1, 3, 5, ... have the other type (ERC: full proposals, ACL: reviews)
    parities = {"original": [0], "alternate": [1], "both": [0, 1]}[text_type]
    assert text_type == "both" or pos_gap % 2 == 0, "--text-type original/alternate needs an even --pos-gap"
    pairs = []
    for d in doc_ids:
        chain = df.loc[d, cols].astype(str).tolist()
        sources = [t for t in range(len(chain)) if t % 2 in parities]
        # one negative per gap and document, so all gaps are equally represented. The source is drawn uniformly
        # among the valid ones for each gap independently, so no gap is pushed towards the end of the chain.
        # If two gaps draw the same source, its positive pair is added twice to keep sources matched across labels.
        for g in neg_gaps:
            valid = [t for t in sources if t + g < len(chain)]
            if not valid:
                continue
            t = rng.choice(valid)
            for gap, label in ((pos_gap, 1), (g, 0)):
                pairs.append(dict(doc=int(d), src=t, gap=gap, label=label, a_col=cols[t], b_col=cols[t + gap],
                                  a=chain[t], b=chain[t + gap]))
                if reverse:  # same two texts swapped: A is later in the chain than B
                    pairs.append(dict(doc=int(d), src=t + gap, gap=-gap, label=3 - label, a_col=cols[t + gap],
                                      b_col=cols[t], a=chain[t + gap], b=chain[t]))
    return pairs


def truncate_text(tokenizer, text, n_tokens):
    """Cut text to (roughly) its first n_tokens tokens, using the character offsets of the tokenizer."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    return text if len(offsets) <= n_tokens else text[:offsets[n_tokens - 1][1]]


# headers of the two texts for decoder LMs (--lora), which have no [CLS] / [SEP] structure
DECODER_HEADERS = ("Text A:\n", "\n\nText B:\n")


def encode_pairs(tokenizer, batch, max_length, **kwargs):
    """Encode pairs as [CLS] A [SEP] B [SEP], giving each text at most half of max_length.
    With tokenizer.pair_headers (decoder LMs) the texts become "Text A:\n<A>" and "\n\nText B:\n<B>"."""
    heads = getattr(tokenizer, "pair_headers", ("", ""))
    n_head = max(len(tokenizer(h, add_special_tokens=False)["input_ids"]) for h in heads)
    half = (max_length - tokenizer.num_special_tokens_to_add(pair=True)) // 2 - n_head
    a = [heads[0] + truncate_text(tokenizer, p["a"], half) for p in batch]
    b = [heads[1] + truncate_text(tokenizer, p["b"], half) for p in batch]
    # longest_first only acts as a safety net in case re-tokenizing a cut text adds a token
    return tokenizer(a, b, truncation="longest_first", max_length=max_length, **kwargs)


def make_loader(pairs, tokenizer, max_length, batch_size, shuffle):
    def collate(batch):
        enc = encode_pairs(tokenizer, batch, max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([p["label"] for p in batch])
        return enc
    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=2)


def show_examples(pairs, tokenizer, max_length, n):
    """Print the first n pairs exactly as the model will see them, i.e. after truncation."""
    for p in pairs[:n]:
        enc = encode_pairs(tokenizer, [p], max_length)
        ids = np.array(enc["input_ids"][0])
        seq = np.array([-1 if s is None else s for s in enc.sequence_ids(0)])
        print(f"\n{'=' * 100}\ndoc {p['doc']} | A = {p['a_col']}, B = {p['b_col']} | gap {p['gap']} | label {p['label']} ({LABEL_NAMES[p['label']]})")
        for i, key in enumerate(("a", "b")):
            n_full = len(tokenizer(p[key], add_special_tokens=False)["input_ids"])
            kept = ids[seq == i]
            print(f"{'-' * 100}\ntext {key.upper()}: {len(kept)}/{n_full} tokens kept\n{'-' * 100}")
            print(tokenizer.decode(kept))
    print("=" * 100)


class LastTokenClassifier(torch.nn.Module):
    """A decoder text model with a linear classification head on the last non-padding token (right padding)."""

    def __init__(self, backbone, num_labels):
        super().__init__()
        self.backbone = backbone
        self.score = torch.nn.Linear(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = attention_mask.sum(1) - 1
        logits = self.score(hidden[torch.arange(len(hidden), device=hidden.device), last])
        loss = None if labels is None else torch.nn.functional.cross_entropy(logits.float(), labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def has_seq_cls_class(config):
    return config.model_type in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES


def load_last_token_classifier(args, config, num_labels, device):
    """Only the text model of a (multimodal) checkpoint, with LoRA adapters, plus a trainable linear head."""
    from peft import LoraConfig, get_peft_model
    backbone = AutoModel.from_pretrained(args.model, config=config, dtype=torch.bfloat16,
                                         key_mapping={r"^model\.language_model\.": ""})
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules="all-linear")
    backbone = get_peft_model(backbone, lora)
    backbone.print_trainable_parameters()
    model = LastTokenClassifier(backbone, num_labels).to(device)
    model.train()
    return model


def load_model(args, num_labels, device):
    """The classifier: a BERT-style encoder fine-tuned in full, or with --lora a decoder LM with LoRA adapters."""
    if not args.lora:
        return AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).to(device)
    from peft import LoraConfig, get_peft_model
    config = AutoConfig.from_pretrained(args.model)
    kwargs = {}
    if hasattr(config, "text_config"):   # multimodal checkpoint (e.g. Ministral 3): keep only the language model
        config = config.text_config
        kwargs["key_mapping"] = {r"^language_model\.model\.": "model."}
    if not has_seq_cls_class(config):
        return load_last_token_classifier(args, config, num_labels, device)
    config.num_labels = num_labels
    config.pad_token_id = args.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config, dtype=torch.bfloat16,
                                                               **kwargs).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(task_type="SEQ_CLS", r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules="all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()   # from_pretrained returns eval mode, where gradient checkpointing is off
    return model


def save_model(model, tokenizer, args):
    if isinstance(model, LastTokenClassifier):
        model.backbone.save_pretrained(args.output_dir)
        torch.save(model.score.state_dict(), os.path.join(args.output_dir, "score.pt"))
    else:
        model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def load_best(model, args, num_labels, device):
    """Reload the best validation checkpoint (for --lora: its adapter weights, into the model already in memory)."""
    if not args.lora:
        return AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    from peft import load_peft_weights, set_peft_model_state_dict
    if isinstance(model, LastTokenClassifier):
        set_peft_model_state_dict(model.backbone, load_peft_weights(args.output_dir, device=str(device)))
        model.score.load_state_dict(torch.load(os.path.join(args.output_dir, "score.pt"), map_location=device))
    else:
        set_peft_model_state_dict(model, load_peft_weights(args.output_dir, device=str(device)))
    return model


def smoke_test(model, optimizer, scheduler, loaders, args, device):
    """Time args.smoke_test training steps and eval batches (the first of each is a warm-up and is not counted)
    and print the projected duration of the full run."""
    n = args.smoke_test
    torch.cuda.reset_peak_memory_stats(device)
    times = {}
    for name, loader in (("train", loaders["train"]), ("eval", loaders["val"])):
        it, ts = iter(loader), []
        for i in range(n + 1):
            batch = {k: v.to(device) for k, v in next(it).items()}
            torch.cuda.synchronize(device)
            t0 = time.time()
            if name == "train":
                train_step(model, batch, optimizer, scheduler, i + 1, args)
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.eval()
                    model(**batch)
                    model.train()
            torch.cuda.synchronize(device)
            if i:
                ts.append(time.time() - t0)
        times[name] = float(np.mean(ts))
    n_train, n_val, n_test = (len(loaders[k]) for k in ("train", "val", "test"))
    total = args.epochs * (n_train * times["train"] + n_val * times["eval"]) + n_test * times["eval"]
    res = {"train_s_per_batch": times["train"], "eval_s_per_batch": times["eval"], "train_batches": n_train,
           "val_batches": n_val, "test_batches": n_test, "epochs": args.epochs, "projected_hours": total / 3600,
           "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9}
    print("smoke test:", json.dumps(res, indent=2))
    with open(os.path.join(args.output_dir, "smoke_test.json"), "w") as f:
        json.dump({"args": vars(args), **res}, f, indent=2)


def train_step(model, batch, optimizer, scheduler, step, args):
    """Forward / backward on one batch; the optimizer steps every args.grad_accum batches."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch["input_ids"].is_cuda):
        loss = model(**batch).loss
    (loss / args.grad_accum).backward()
    if step % args.grad_accum == 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return loss


@torch.no_grad()
def evaluate(model, loader, pairs, device, num_labels=2):
    model.eval()
    preds = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        preds += logits.argmax(-1).tolist()
    preds = np.array(preds)
    labels = np.array([p["label"] for p in pairs])
    gaps = np.array([p["gap"] for p in pairs])
    res = {"accuracy": float((preds == labels).mean())}
    # confusion[i][j] = number of pairs with true label i predicted as j
    res["confusion"] = np.bincount(labels * num_labels + preds, minlength=num_labels ** 2).reshape(num_labels, -1).tolist()
    if num_labels == 4:
        # direction only (forward = labels 0/1, reversed = 2/3), and copy type only (direct = 1/2, multiple = 0/3)
        res["direction_accuracy"] = float(((preds >= 2) == (labels >= 2)).mean())
        res["copy_type_accuracy"] = float((np.isin(preds, [1, 2]) == np.isin(labels, [1, 2])).mean())
    # accuracy broken down by gap: how hard is each number of reuses to tell apart from a direct copy
    for g in sorted(set(gaps)):
        res[f"acc_gap{g}"] = float((preds[gaps == g] == labels[gaps == g]).mean())
    # same metrics restricted to pairs whose first text is the original (human) one, when it is in the set,
    # and for comparison to pairs whose first text is itself a generation
    is_original = np.array([p["a_col"] == "original" for p in pairs])
    if is_original.any():
        for name, mask in (("original", is_original), ("generated", ~is_original)):
            res[f"{name}_src_accuracy"] = float((preds[mask] == labels[mask]).mean())
            for g in sorted(set(gaps[mask])):
                m = mask & (gaps == g)
                res[f"{name}_src_acc_gap{g}"] = float((preds[m] == labels[m]).mean())
    model.train()
    return res


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    # ACL: the original column is the paper, which is only converted once into the first research plan
    drop_original = args.drop_original or "acl" in os.path.basename(args.data).split(".")
    cols = chain_columns(df, drop_original)
    print(f"chain: {cols[0]} -> ... -> {cols[-1]} ({len(cols)} texts){' (original column dropped)' if drop_original else ''}")
    docs = list(df.index)
    rng.shuffle(docs)
    n_test = n_val = int(0.1 * len(docs))
    splits = {"test": docs[:n_test], "val": docs[n_test:n_test + n_val], "train": docs[n_test + n_val:]}
    pairs = {k: build_pairs(df, cols, v, args.pos_gap, args.max_gap, args.text_type, rng, args.reverse)
             for k, v in splits.items()}
    num_labels = 4 if args.reverse else 2
    print({k: f"{len(v)} pairs from {len(splits[k])} docs" for k, v in pairs.items()})

    # save pair indices: `doc` is the row of the source CSV, `a_col` / `b_col` the columns of the two texts,
    # `src` the chain position of text A (0 = first column of the chain). To get the texts back for a pair:
    #   gen = pd.read_csv("<generation csv>")
    #   pairs = pd.read_csv("outputs/pair_classifier/pairs.csv")
    #   r = pairs.iloc[0]
    #   text_a, text_b = gen.loc[r.doc, r.a_col], gen.loc[r.doc, r.b_col]
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame([{"split": k, **{c: p[c] for c in ("doc", "src", "gap", "label", "a_col", "b_col")}}
                  for k, v in pairs.items() for p in v]).to_csv(os.path.join(args.output_dir, "pairs.csv"), index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if type(tokenizer).__name__ == "MistralCommonBackend":
        # the mistral-common tokenizer cannot encode text pairs or return character offsets: use tokenizer.json
        # (with the corrected pre-tokenization regex of Mistral's recent tokenizers)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model, fix_mistral_regex=True)
    if args.lora:
        tokenizer.pair_headers = DECODER_HEADERS
        tokenizer.padding_side = "right"   # LastTokenClassifier reads the last non-padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        args.pad_token_id = tokenizer.pad_token_id
    if args.lr is None:
        args.lr = 1e-4 if args.lora else 3e-5
    print(f"max_length: {args.max_length} tokens (at most {args.max_length // 2} per text)")
    # save the arguments of the run (with max_length resolved), before training so they are there even if it fails
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    show_examples(pairs["train"], tokenizer, args.max_length, args.show_examples)
    if args.dry_run:
        return
    model = load_model(args, num_labels, device)

    train_loader = make_loader(pairs["train"], tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(pairs["val"], tokenizer, args.max_length, args.batch_size * 2, shuffle=False)
    test_loader = make_loader(pairs["test"], tokenizer, args.max_length, args.batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    if args.smoke_test:
        smoke_test(model, optimizer, scheduler, {"train": train_loader, "val": val_loader, "test": test_loader},
                   args, device)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    best_val, history = -1.0, []
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, scheduler, step, args)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val = evaluate(model, val_loader, pairs["val"], device, num_labels)
        history.append({"epoch": epoch + 1, **val})
        print(f"epoch {epoch + 1} val: {val}")
        if val["accuracy"] > best_val:
            best_val = val["accuracy"]
            save_model(model, tokenizer, args)

    model = load_best(model, args, num_labels, device)
    test = evaluate(model, test_loader, pairs["test"], device, num_labels)
    print(f"test (best val checkpoint): {test}")
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "history": history, "test": test}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="CSV with 'original' and 'iteration_XX' columns.")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-base", help="HF encoder model.")
    parser.add_argument("--text-type", choices=["original", "alternate", "both"], default="original",
                        help="Which texts to pair: same type as the first text of the chain (ERC: abstracts, "
                             "ACL: research plans), the other type (ERC: full proposals, ACL: reviews), or both.")
    parser.add_argument("--drop-original", action="store_true",
                        help="Ignore the original column (done automatically for ACL files).")
    parser.add_argument("--pos-gap", type=int, default=2, help="Gap defining a direct copy (2 = next text of the same type).")
    parser.add_argument("--max-gap", type=int, default=10, help="Largest gap used for 'multiple reuses' negatives.")
    parser.add_argument("--reverse", action="store_true",
                        help="Also add every pair with its texts swapped (A later than B): 4 classes, "
                             "direct / multiple / reversed direct / reversed multiple.")
    parser.add_argument("--max-length", type=int, default=8192, help="Max tokens for the pair, split equally between the two texts (default: 8192, ModernBERT's maximum context).")
    parser.add_argument("--batch-size", type=int, default=8, help="Use <= 8 for long texts (e.g. ERC proposals at 8192 tokens) to fit in 80GB.")
    parser.add_argument("--lr", type=float, default=None, help="Default: 3e-5, or 1e-4 with --lora.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Batches per optimizer step (effective batch = batch size x this).")
    parser.add_argument("--lora", action="store_true", help="Fine-tune a decoder LM with LoRA adapters (bf16, gradient checkpointing).")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (alpha = 2 x rank).")
    parser.add_argument("--smoke-test", type=int, default=0, metavar="N",
                        help="Time N training steps and N eval batches, print the projected run time and exit.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-examples", type=int, default=2, help="Print this many training pairs as seen by the model.")
    parser.add_argument("--dry-run", action="store_true", help="Build the dataset, print the examples and exit without training.")
    parser.add_argument("--output-dir", type=str, default="outputs/pair_classifier")
    main(parser.parse_args())
