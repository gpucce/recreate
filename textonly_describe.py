import pandas as pd
from vllm import LLM, SamplingParams
from src.utils import parse_describe as parse_json

import yaml


def main(args):
    data = pd.read_csv(args.input)

    if args.debug:
        data = data[:100]
    

    prompt = yaml.safe_load(open("src/prompts.yaml", "r"))["textonly_describe"]
    model_name = "Qwen/Qwen3.5-4B"

    print(f"Data size: {len(data)}\nModel: {model_name}\nPrompt: {prompt}")
    llm = LLM(model_name,
              dtype="bfloat16",
              tensor_parallel_size=1,
              max_model_len=8417*6,
              disable_custom_all_reduce=True,
              # enforce_eager=True,  # uncomment only if the hang persists

    )
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)

    MAX_ITERATION = 15
    iterations_cols = ["original"] + [f"iteration_{i:02d}" for i in range(0, MAX_ITERATION)]
    print(iterations_cols)

    from collections import defaultdict
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

        ouputs = [o.outputs[0].text.strip() for o in outputs]
        for i, text in enumerate(ouputs):
            parsed = parse_json(text)
            descriptions[target_iter].append(parsed)

    # store the descriptions in a JSON file (one for each target iteration)
    outdir = "outputs/textonly_describe"
    import os
    os.makedirs(outdir, exist_ok=True)
    for target_iter in iterations_cols:
        outpath = os.path.join(outdir, f"{target_iter}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            import json
            json.dump(descriptions[target_iter], f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="debug.rewritten_articles.csv", help="Path to the input CSV file containing the data to be rewritten.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer iterations and data.")
    args = parser.parse_args()
    main(args)