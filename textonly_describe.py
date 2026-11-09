from src.utils import parse_describe as parse_json

import pandas as pd
import json
import os
import re
import yaml
from collections import defaultdict


HEADER_RE = re.compile(r"(?m)^[ \t]{0,3}(?=#{1,6}[ \t])")


def split_into_sections(text):
    """Split a markdown-like document at '#' headers, keeping each header with its body.

    Documents without any header line are returned unchanged as a single section.
    """
    text = str(text)
    parts = [p.strip() for p in HEADER_RE.split(text) if p and p.strip()]
    return parts or [text.strip()]


def merge_sections(sections):
    """Merge per-section NER results into one document-level result.

    All entity occurrences are kept (no deduplication). Document-level fields
    (e.g. dates/location) take the first non-empty, non-"missing" value seen.
    """
    entities = []
    doc_fields = {}
    seen_fields = set()
    saw_named_entities = False
    for parsed in sections:
        if "named_entities" in parsed:
            saw_named_entities = True
            entities.extend(parsed["named_entities"])
        for key, value in parsed.items():
            if key in ("named_entities", "error", "raw_output"):
                continue
            seen_fields.add(key)
            if value in ("", "missing", None):
                continue
            doc_fields.setdefault(key, value)

    if not saw_named_entities:
        return sections[0]
    for key in seen_fields:
        doc_fields.setdefault(key, "missing")
    return {"named_entities": entities, **doc_fields}


def get_prompt(dataset_name):
    if dataset_name == "bbc-news":
        prompt = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_describe"]
    elif dataset_name == "erc-grants":
        prompt = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_describe_erc"]
    else:
        raise NotImplementedError(f"Prompts for dataset {dataset_name} are not implemented yet.")

    return prompt


def main(args):
    data = pd.read_csv(args.input)
    max_iterations = args.max_iterations
    _, _, generator_model_name, _ = args.input.split("/")

    # store the descriptions in a JSON file (one for each target iteration)
    outdir = os.path.join(args.outdir, args.dataset, generator_model_name)
    os.makedirs(outdir, exist_ok=True)

    print(f"{'#'*50}\nInput data path: {args.input}\nInferred Generator Model Name: {generator_model_name}\nDataset: {args.dataset}\nMax iterations: {max_iterations}\nOutput directory: {outdir}\n{'#'*50}")
    
    if args.debug:
        data = data[:10]
        max_iterations = 10
        print("Debug mode: using only the first 10 rows and 10 iterations.")
    
    prompt = get_prompt(args.dataset)

    # Determine which iterations to process based on the existing files in the output directory and the columns in the input data
    iterations_cols = ["original"] + [f"iteration_{i:02d}" for i in range(0, max_iterations)]
    if args.skip_first_iterations:
        already_processed = sorted(os.listdir(outdir))
        available_cols = data.columns.tolist()
        iterations_cols = [col for col in iterations_cols if f"{col}.json" not in already_processed and col in available_cols]

    print("Iterations to process:")
    for col in iterations_cols:
        print(f"- {col}")

    model_name = "Qwen/Qwen3.5-27B"
    print(f"Data size: {len(data)}\nModel: {model_name}\nPrompt: {prompt}")
    llm = LLM(model_name,
              dtype="bfloat16",
              tensor_parallel_size=1,
              max_model_len=8192*6,
              gpu_memory_utilization=0.85,
              max_num_seqs=64
    )

    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    # sampling_params = SamplingParams(max_tokens=2048, temperature=1.0, top_p=0.95) # preferred by config

    descriptions = defaultdict(list)

    for target_iter in iterations_cols:
        iter_data = data[target_iter].tolist()

        if args.split_sections:
            doc_sections = [split_into_sections(text) for text in iter_data]
            flat = [(d, sec) for d, secs in enumerate(doc_sections) for sec in secs]
            messages = [[{"role": "user", "content": prompt.format(data=sec)}] for _, sec in flat]
        else:
            messages = [[{"role": "user", "content": prompt.format(data=text)}] for text in iter_data]

        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )

        outputs = [o.outputs[0].text.strip() for o in outputs]
        if args.split_sections:
            per_doc = defaultdict(list)
            for (d, _), text in zip(flat, outputs):
                per_doc[d].append(parse_json(text))
            for i in range(len(iter_data)):
                descriptions[target_iter].append(merge_sections(per_doc[i]))
        else:
            for text in outputs:
                descriptions[target_iter].append(parse_json(text))

        fname = target_iter
        if args.debug:
            fname = target_iter + ".debug"
        outpath = os.path.join(outdir, f"{fname}.json")

        print(f"Saving descriptions for iteration {target_iter} to {outpath}")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(descriptions[target_iter], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["bbc-news", "erc-grants", "acl"], default="bbc-news", help="Dataset to use for text rewriting.")
    parser.add_argument("--input", type=str, default="textonly-generations/bbc-news/bbc-news-with_generations.csv", help="Path to the input CSV file containing the data to be rewritten.")
    parser.add_argument("--skip-first-iterations", action="store_true", help="Skip pre-computed iterations when resuming extraction.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of new tokens to generate per iteration.")
    parser.add_argument("--max-iterations", type=int, default=30, help="Maximum number of iterations to process.")
    parser.add_argument("--outdir", type=str, default="outputs/textonly_describe", help="Directory to save the output JSON files.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer iterations and data.")
    parser.add_argument("--split-sections", action="store_true", help="Split each document into '#' sections and run NER per section, merging the results into one entry per document.")
    args = parser.parse_args()
    main(args)
