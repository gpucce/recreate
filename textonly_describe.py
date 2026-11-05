from src.utils import parse_describe as parse_json

import pandas as pd
import json
import os
import yaml
from collections import defaultdict


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

    if args.debug:
        data = data[:100]
        max_iterations = 10
        print("Debug mode: using only the first 100 rows and 10 iterations.")
    
    prompt = get_prompt(args.dataset)

    # Determine which iterations to process based on the existing files in the output directory and the columns in the input data
    iterations_cols = ["original"] + [f"iteration_{i:02d}" for i in range(0, max_iterations)]
    if args.skip_first_iterations:
        already_processed = sorted(os.listdir(args.outdir))
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
              max_model_len=8417*6,
              gpu_memory_utilization=0.85,
              # disable_custom_all_reduce=True,
            #   trust_remote_code=True,
              max_num_seqs=64
    )

    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)
    # sampling_params = SamplingParams(max_tokens=2048, temperature=1.0, top_p=0.95) # preferred by config

    descriptions = defaultdict(list)

    for target_iter in iterations_cols:
        iter_data = data[target_iter].tolist()
        messages = [[{"role": "user", "content": prompt.format(data=text)}] for text in iter_data]
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )

        outputs = [o.outputs[0].text.strip() for o in outputs]
        for i, text in enumerate(outputs):
            parsed = parse_json(text)
            descriptions[target_iter].append(parsed)

        # store the descriptions in a JSON file (one for each target iteration)
        outdir = os.path.join(args.outdir, args.dataset)
        os.makedirs(outdir, exist_ok=True)


        fname = target_iter
        if args.debug:
            fname = target_iter + ".debug"
        outpath = os.path.join(outdir, f"{fname}.json")

        print(f"Saving descriptions for iteration {target_iter} to {outpath}")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(descriptions[target_iter], f, ensure_ascii=False, indent=2)

        # for target_iter in iterations_cols:
        #     if args.debug:
        #         target_iter = target_iter + ".debug"
        #     outpath = os.path.join(outdir, f"{target_iter}.json")
        #     with open(outpath, "w", encoding="utf-8") as f:
        #         json.dump(descriptions[target_iter], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["bbc-news", "erc-grants", "acl"], default="bbc-news", help="Dataset to use for text rewriting.")
    parser.add_argument("--input", type=str, default="textonly-generations/bbc-news/bbc-news-with_generations.csv", help="Path to the input CSV file containing the data to be rewritten.")
    parser.add_argument("--skip-first-iterations", action="store_true", help="Skip pre-computed iterations when resuming extraction.")
    parser.add_argument("--max-iterations", type=int, default=30, help="Maximum number of iterations to process.")
    parser.add_argument("--outdir", type=str, default="outputs/textonly_describe", help="Directory to save the output JSON files.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer iterations and data.")
    args = parser.parse_args()
    main(args)
