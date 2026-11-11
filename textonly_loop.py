import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

import yaml


def get_data(dataset_name):
    if dataset_name == "bbc-news":
        ds = load_dataset("SetFit/bbc-news")
        # merge train and test "text" columns into a single list
        data = [elem for elem in ds["train"]["text"]] + [elem for elem in ds["test"]["text"]]
    elif dataset_name == "erc-grants":
        df = pd.read_csv("data/content_convergence/original_datasets/ERC_data_filtered.csv")
        data = df["Abstract"].tolist()
    elif dataset_name == "acl":
        MARKDOWN_DIR = "data/content_convergence/original_datasets/2026_acl_markdown"
        df_ids = pd.read_csv("data/content_convergence/original_datasets/2026.acl.csv")[["id"]]
        data = []
        for idx, row in df_ids.iterrows():
            markdown_file = os.path.join(MARKDOWN_DIR, f"10.18653_v1_2026.acl-long.{row['id']}", "paper.md")
            try:
                with open(markdown_file, "r") as f:
                    markdown_content = f.read()
            except FileNotFoundError:
                print(f"Markdown file not found for document ID {row['id']}. Skipping.")
                continue
            lookfor = '<span id="page-9-'
            markdown_content = markdown_content.split(lookfor)[0]  # keep only the content before the 9th page
            data.append(markdown_content)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented yet.")

    return data

def get_prompts(dataset_name):
    if dataset_name == "bbc-news":
        prompt_to_social_media = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_socialmedia"]
        prompt_to_news_outlet = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_news"]
        return prompt_to_social_media, prompt_to_news_outlet, None
    elif dataset_name == "erc-grants":
        prompt_to_project =  yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_erc_project"]
        prompt_to_abstract = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_erc_abstract"]
        return prompt_to_project, prompt_to_abstract, None
    elif dataset_name == "acl":
        prompt_to_research_plan = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_research_plan"]
        prompt_to_review = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_review"]
        prompt_to_update_research_plan = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_update_research_plan"]
        return prompt_to_update_research_plan, prompt_to_review, prompt_to_research_plan
    else:
        raise NotImplementedError(f"Prompts for dataset {dataset_name} are not implemented yet.")


def main(args):
    num_iterations = args.num_iterations
    model_name = args.model_name

    data = get_data(args.dataset)
    if args.debug:
        print("Debug mode: using only the first 10 articles and 10 iterations.")
        data = data[:10]
        num_iterations = 10

    first_gen_prompt, second_gen_prompt, convert_initial_prompt = get_prompts(args.dataset)

    if args.dataset == "bbc-news":
        outdir = os.path.join(args.outdir, "bbc-news")
        fname = "bbc-news.with_generations.csv"
    elif args.dataset == "erc-grants":
        outdir = os.path.join(args.outdir, "erc_grants")
        fname = "erc_grants.with_generations.csv"
    elif args.dataset == "acl":
        outdir = os.path.join(args.outdir, "acl")
        fname = "acl.with_generations.csv"
    outdir = os.path.join(args.outdir, args.dataset, model_name.lower().replace("/", "_"))
    os.makedirs(outdir, exist_ok=True)

    if args.resume is not None:
        fname = fname.replace(".csv", ".resumed.csv")

    if args.debug:
        fname = "debug." + fname
    
    print(f"{'#' * 50}\nDataset: {args.dataset}\nGenerator Model: {model_name}\nNumber of iterations: {num_iterations}\nOutput directory: {outdir}\n{'#' * 50}")

    llm = LLM(model_name,
              dtype="bfloat16",
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs,
              disable_custom_all_reduce=args.disable_custom_all_reduce,
              language_model_only=True,
            )

    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    df = pd.DataFrame({"original": data})
    iterations = np.arange(num_iterations)

    if args.resume is not None:
        print("Resuming from previous run...")
        _resume_df = pd.read_csv(args.resume)
        df = _resume_df
        last_iteration = max([int(col.split("_")[1]) for col in df.columns if col.startswith("iteration_")], default=-1)
        iterations = np.arange(last_iteration + 1, last_iteration + num_iterations + 1)

    for iteration in iterations:
        if args.dataset == "acl" and iteration == 0:
            # first iteration for ACL dataset: convert the original abstract into a research plan
            prompt = convert_initial_prompt
        else:
            if iteration % 2 == 0:
                prompt = first_gen_prompt
            else:
                prompt = second_gen_prompt

        # first iteration reads the original column, later ones read the previous iteration
        if args.dataset == "acl" and iteration % 2 == 0 and iteration > 0:
            # hit iteration to update research plan based on reviews. We need both the previous research plan and the reviews to generate the updated research plan.
            prev_reviews = df[f"iteration_{iteration - 1:02d}"].tolist()
            prev_plans = df[f"iteration_{iteration - 2:02d}"].tolist()

            messages = [
                [{"role": "user", "content": prompt.format(reviews=reviews, plan=plan)}]
                 for reviews, plan in zip(prev_reviews, prev_plans)
            ]

        else:
            src_col = "original" if iteration == 0 else f"iteration_{iteration - 1:02d}"
            current = df[src_col].tolist()
    
            # one conversation per article: a list of lists of message dicts
            messages = [
                [{"role": "user", "content": prompt.format(data=text)}]
                for text in current
            ]
    
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )
    
        # # outputs come back in the same order as the input messages
        df[f"iteration_{iteration:02d}"] = [o.outputs[0].text.strip() for o in outputs]

    print(f"Saving results to {os.path.join(outdir, fname)}")
    df.to_csv(os.path.join(outdir, fname), index=False)


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.8-27B", help="Model name to use for text rewriting.")
    parser.add_argument("--dataset", type=str, choices=["bbc-news", "erc-grants", "acl"], default="bbc-news", help="Dataset to use for text rewriting.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer iterations and data.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a previous run by reading the existing CSV file.")
    parser.add_argument("--num-iterations", type=int, default=15, help="Number of iterations to perform.")
    parser.add_argument("--outdir", type=str, default="textonly-generations", help="Directory to save the output CSV files.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of new tokens to generate per iteration.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for distributed training.")
    parser.add_argument("--disable-custom-all-reduce", action="store_true", help="Disable custom all-reduce for distributed training.")
    parser.add_argument("--max-model-len", type=int, default=8192*4, help="KV cache size.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum number of sequences to process in parallel.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory utilization for the model.")
    args = parser.parse_args()

    main(args)
