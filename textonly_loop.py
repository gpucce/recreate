import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

import yaml


def main(args):
    if args.debug:
        ITERATIONS = 15
    else:
        ITERATIONS = 15

    ds = load_dataset("SetFit/bbc-news")

    # merge train and test "text" columns into a single list
    data = [elem for elem in ds["train"]["text"]] + [elem for elem in ds["test"]["text"]]

    if args.debug:
        data = data[:100]

    prompt_to_social_media = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_socialmedia"]
    prompt_to_news_outlet = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_rewrite_as_news"]

    # model_name = "Qwen/Qwen3.5-4B"
    model_name = "Qwen/Qwen3.8-27B"

    print(f"Data size: {len(data)}\nNumber of iterations: {ITERATIONS}\nModel: {model_name}\nPrompt to social media: {prompt_to_social_media}\nPrompt to news outlet: {prompt_to_news_outlet}")

    llm = LLM(model_name,
              dtype="bfloat16",
            #   device="cuda",
              tensor_parallel_size=1,
              max_model_len=8417*4,
              gpu_memory_utilization=0.85,
              max_num_seqs=64
            )

    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)

    df = pd.DataFrame({"original": data})
    iterations = np.arange(ITERATIONS)

    if args.resume:
        _resume_df = pd.read_csv(args.resume)
        df = _resume_df
        last_iteration = max([int(col.split("_")[1]) for col in df.columns if col.startswith("iteration_")], default=-1)

        iterations = np.arange(last_iteration + 1, last_iteration + ITERATIONS)

    for iteration in iterations:
        if iteration % 2 == 0:
            prompt = prompt_to_social_media
        else:
            prompt = prompt_to_news_outlet

        # first iteration reads the original column, later ones read the previous iteration
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
    
        # outputs come back in the same order as the input messages
        df[f"iteration_{iteration:02d}"] = [o.outputs[0].text.strip() for o in outputs]

    if args.resume:
        df.to_csv("resumed.back_and_forth_platforms.rewritten_articles.csv", index=False)
    else:
        df.to_csv(f"{'debug.' if args.debug else ''}back_and_forth_platforms.rewritten_articles.csv", index=False)

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer iterations and data.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--skip-first-iterations", type=bool, default=False, help="Skip initial iterations when resuming extraction.")
    args = parser.parse_args()

    main(args)

