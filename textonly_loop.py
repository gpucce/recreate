import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

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
              tensor_parallel_size=2,
              max_model_len=8417*6
            )

    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)

    df = pd.DataFrame({"original": data})

    for iteration in range(ITERATIONS):
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

    df.to_csv(f"{'debug.' if args.debug else ''}back_and_forth_platforms.rewritten_articles.csv", index=False)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer iterations and data.")
    args = parser.parse_args()

    main(args)