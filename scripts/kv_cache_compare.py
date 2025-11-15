import torch
import argparse
import tiktoken
from mini_transformer.benchmark.benchmark_utils import timer
from mini_transformer.models.gpt2 import GPT2Config
from mini_transformer.scratch import (
    gpt2_with_dynamic_cache,
    gpt2_with_sliding_window_cache,
    gpt2_without_kv_cache,
)


def gpt2_inference_without_kv_cache():
    from mini_transformer.scratch.gpt2_without_kv_cache import (
        GPT2ModelForCausalLM,
        generate_text_simple,
    )

    config = GPT2Config()
    model = GPT2ModelForCausalLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    prompt = 'Hello, I am'
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    @timer
    def inference_without_kv_cache():
        output_ids = generate_text_simple(model, input_ids, 200, config.context_length)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return output_ids

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output_ids = inference_without_kv_cache()
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(output_text)


def gpt2_inference_with_dynamic_cache():
    from mini_transformer.scratch.gpt2_with_dynamic_cache import (
        GPT2ModelForCausalLM,
        generate_text_with_kv_cache,
    )

    config = GPT2Config()
    model = GPT2ModelForCausalLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    prompt = 'Hello, I am'
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    @timer
    def inference_with_dynamic_cache():
        output_ids = generate_text_with_kv_cache(model, input_ids, 200, config.context_length)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return output_ids

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output_ids = inference_with_dynamic_cache()
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(output_text)


def gpt_inference_with_sliding_window_cache():
    from mini_transformer.scratch.gpt2_with_sliding_window_cache import (
        GPT2ModelForCausalLM,
        generate_text_with_kv_cache,
    )

    config = GPT2Config()
    model = GPT2ModelForCausalLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    prompt = 'Hello, I am'
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    @timer
    def inference_with_sliding_window_cache():
        output_ids = generate_text_with_kv_cache(model, input_ids, 200, config.context_length)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return output_ids

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output_ids = inference_with_sliding_window_cache()
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(output_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to compare kv cache in GPT2 model.')
    parser.add_argument(
        '--kv-version', choices=['dynamic', 'sliding-window', 'none'], default='none'
    )
    args = parser.parse_args()

    match args.kv_version:
        case 'dynamic':
            gpt2_inference_with_dynamic_cache()
        case 'sliding-window':
            gpt_inference_with_sliding_window_cache()
        case 'none':
            gpt2_inference_without_kv_cache()
        case _:
            raise ValueError('kv-version must be "dynamic", "sliding-window" or "none" (default)')
