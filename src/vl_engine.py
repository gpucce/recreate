import functools

from transformers import pipeline
from tqdm.auto import tqdm


def main():
    print("debugging module vl_engine.py")
    generate(
        model_name="Qwen/Qwen3.5-4B",
        backend="hf",
        messages=[
            {"role": "user", "content": [
                {"type": "text",
                "text": "The quick brown fox jumps over the"}]} for _ in range(8)
        ],
        batch_size=8,
        max_new_tokens=64
    )


def load_model_hf(model_name, hf_task, dtype, device):
    pipe = pipeline(
        hf_task,
        model=model_name,
        dtype=dtype,
        device_map="auto",
        max_length=None,
    )
    return pipe

def load_model_vllm(model_name, dtype="bfloat16", device="cuda", gpu_memory_utilization=0.85, max_model_len=8192):
    from vllm import LLM
    return LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )


def to_vllm_messages(messages):
    """Rewrite content parts for vLLM: {"type":"image"} -> {"type":"image_pil"}."""
    out = []
    for message in messages:
        content = []
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                part = {"type": "image_pil", "image_pil": part.get("image")}
            content.append(part)
        out.append({"role": message.get("role"), "content": content})
    return out


def parse_responses(responses: list, backend: str):
    if backend == "hf":
        parsed = []
        for r in responses:
            parsed.append(r[0]['generated_text'])
        return parsed
    elif backend == "vllm":
        return [r.outputs[0].text for r in responses]
    pass


def generate(
        model_name: str = "Qwen/Qwen3.5-4B", backend: str = "hf", messages: list = None,
        hf_task: str = "image-text-to-text", device="cuda", dtype="bfloat16", batch_size: int = 2, max_new_tokens: int = 64,
        loaded_llm=None):
    if backend == "hf":
        llm = load_model_hf(model_name, hf_task, dtype, device)
        llm.processor.apply_chat_template = functools.partial(llm.processor.apply_chat_template, enable_thinking=False)
        llm.processor.tokenizer.padding_side = "left"
    elif backend == "vllm":
        if loaded_llm is not None:
            llm = loaded_llm
        else:
            llm = load_model_vllm(model_name, dtype=dtype, device=device)

    if backend == "hf":
        outputs = []
        for i in tqdm(range(0, len(messages), batch_size), desc="Generating"):
            batch = messages[i:i + batch_size]
            outputs.extend(
                llm(
                    text=batch,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    return_full_text=False,
                )
            )

    elif backend == "vllm":
        from vllm import SamplingParams
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        outputs = []
        for i in tqdm(range(0, len(messages), batch_size), desc="Generating"):
            batch = messages[i:i + batch_size]
            outputs.extend(
                llm.chat(
                    [to_vllm_messages(m) for m in batch],
                    sampling_params,
                    chat_template_kwargs={"enable_thinking": False},
                    use_tqdm=False,
                )
            )

    responses = parse_responses(responses=outputs, backend=backend)
    return responses, llm


if __name__ == "__main__":
    main()