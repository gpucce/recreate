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
        df = pd.read_csv("data/content_convergence/original_datasets/2026.acl.csv")
        data = df["abstract"].tolist()
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

    data = get_data(args.dataset)
    if args.debug:
        print("Debug mode: using only the first 10 articles and 10 iterations.")
        data = data[:10]
        num_iterations = 10

    first_gen_prompt, second_gen_prompt, convert_initial_prompt = get_prompts(args.dataset)
    
    model_name = "Qwen/Qwen3.8-27B"
    model_name = "google/gemma-4-31B-it"
    print(f"Data size: {len(data)}\nNumber of iterations: {num_iterations}\nModel: {model_name}\n{'#' * 50}\nFirst iteration prompt: {first_gen_prompt}\n{'#' * 50}\nSecond iteration prompt: {second_gen_prompt}")

    llm = LLM(model_name,
              dtype="bfloat16",
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=0.95,
              max_num_seqs=64,
              disable_custom_all_reduce=args.disable_custom_all_reduce,
              language_model_only=True,
            )

    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    df = pd.DataFrame({"original": data})
    iterations = np.arange(num_iterations)

    if args.resume:
        print("Resuming from previous run...")
        _resume_df = pd.read_csv(args.resume)
        df = _resume_df
        last_iteration = max([int(col.split("_")[1]) for col in df.columns if col.startswith("iteration_")], default=-1)

        iterations = np.arange(last_iteration + 1, last_iteration + num_iterations)

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

    if args.dataset == "bbc-news":
        outdir = os.path.join(args.outdir, "bbc-news")
        fname = "bbc-news.with_generations.csv"
    elif args.dataset == "erc-grants":
        outdir = os.path.join(args.outdir, "erc_grants")
        fname = "erc_grants.with_generations.csv"
    elif args.dataset == "acl":
        outdir = os.path.join(args.outdir, "acl")
        fname = "acl.with_generations.csv"
    outdir = os.path.join(args.outdir, model_name.lower().replace("/", "_"))
    os.makedirs(outdir, exist_ok=True)

    if args.resume:
        fname = fname.replace(".csv", ".resumed.csv")

    if args.debug:
        fname = "debug." + fname

    print(f"Saving results to {os.path.join(outdir, fname)}")
    df.to_csv(os.path.join(outdir, fname), index=False)


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["bbc-news", "erc-grants", "acl"], default="bbc-news", help="Dataset to use for text rewriting.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer iterations and data.")
    parser.add_argument("--resume", action="store_true", help="Resume from a previous run by reading the existing CSV file.")
    parser.add_argument("--num-iterations", type=int, default=15, help="Number of iterations to perform.")
    parser.add_argument("--outdir", type=str, default="textonly-generations", help="Directory to save the output CSV files.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of new tokens to generate per iteration.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for distributed training.")
    parser.add_argument("--disable-custom-all-reduce", action="store_true", help="Disable custom all-reduce for distributed training.")
    parser.add_argument("--max-model-len", type=int, default=8192*4, help="KV cache size.")
    args = parser.parse_args()

    main(args)

