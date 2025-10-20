import torch
import tiktoken
from road2dl.benchmark.benchmark_utils import timer
from road2dl.models.gpt2 import GPT2Config


def gpt2_inference_without_kv_cache():
    
    from road2dl.scratch.gpt2_without_kv_cache import (
        GPT2ModelForCausalLM,
        generate_text_simple
    )

    config = GPT2Config()
    model = GPT2ModelForCausalLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    prompt = 'Hello, I am'
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    @timer
    def inference_without_kv_cache():
        output_ids = generate_text_simple(
            model, input_ids, 200, config.context_length
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return output_ids
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output_ids = inference_without_kv_cache()
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(output_text)
    

def gpt2_inference_with_kv_cache():
    from road2dl.scratch.gpt2_with_kv_cache import (
        GPT2ModelForCausalLM,
        generate_text_with_kv_cache
    )

    config = GPT2Config()
    model = GPT2ModelForCausalLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    prompt = 'Hello, I am'
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    @timer
    def inference_with_kv_cache():
        output_ids = generate_text_with_kv_cache(
            model, input_ids, 200, config.context_length
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return output_ids
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output_ids = inference_with_kv_cache()
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(output_text)
    
    
def gpt_inference_with_kv_cache_optimized():
    from road2dl.scratch.gpt2_with_kv_cache_optimized import (
        GPT2ModelForCausalLM,
        generate_text_with_kv_cache
    )

    config = GPT2Config()
    model = GPT2ModelForCausalLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    prompt = 'Hello, I am'
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    @timer
    def inference_with_kv_cache_optimized():
        output_ids = generate_text_with_kv_cache(
            model, input_ids, 200, config.context_length
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return output_ids
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output_ids = inference_with_kv_cache_optimized()
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(output_text)
    

if __name__ == '__main__':
    # gpt2_inference_without_kv_cache()
    # gpt2_inference_with_kv_cache()
    gpt_inference_with_kv_cache_optimized()
