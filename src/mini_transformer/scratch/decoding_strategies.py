"""
This module shows how to implement different decoding strategies for text generation.

References:
- https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html
- https://huggingface.co/blog/zh/how-to-generate
"""

from typing import Literal
import torch


def greedy_sampling(logits, beams):
    assert len(logits.shape) == 2, (
        f'Input shape should be (batch_size, vocab_size), got {logits.shape}'
    )
    return torch.topk(logits, k=beams).indices


def top_k_sampling(logits, top_k, temperature, beams):
    assert len(logits.shape) == 2, (
        f'Input shape should be (batch_size, vocab_size), got {logits.shape}'
    )
    assert top_k > 0
    assert beams <= top_k

    top_k_logits, _ = torch.topk(logits, k=top_k)
    masked_logits = torch.where(logits < top_k_logits[:, -1].reshape((-1, 1)), -torch.inf, logits)
    topk_probs = torch.softmax(masked_logits / temperature, dim=-1)
    next_tokens_id = torch.multinomial(topk_probs, num_samples=beams)

    return next_tokens_id


def top_p_sampling(logits, top_p, temperature, beams):
    assert len(logits.shape) == 2, (
        f'Input shape should be (batch_size, vocab_size), got {logits.shape}'
    )
    assert top_p > 0 and top_p <= 1

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits / temperature, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Obtain the nucleus sampling mask based on cumulative probability, and ensure that at least the first token is retained.
    mask_sorted = cumulative_probs <= top_p
    mask_sorted[:, 0] = True

    # If the number of 1 in a row exceeds beams, the excess 1 should be replaced with 0.
    positions = (
        torch.arange(logits.size(1), device=logits.device).unsqueeze(0).expand_as(mask_sorted)
    )
    mask_sorted = mask_sorted & (positions < beams)

    # Restore the mask from the sorted index back to the original index order
    mask = torch.zeros_like(mask_sorted, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, mask_sorted)

    # Filter out unreserved tokens (set to -inf)
    masked_logits = torch.where(mask, logits, -torch.inf)
    topp_probs = torch.softmax(masked_logits / temperature, dim=-1)
    next_tokens_id = torch.multinomial(topp_probs, num_samples=beams)

    return next_tokens_id


@torch.no_grad()
def greedy_search(model, input_ids, max_new_tokens):
    """
    Generate text using greedy search.

    Args:
        model (torch.nn.Module): The model to use for generation.
        input_ids (torch.Tensor): The input token IDs.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        torch.Tensor: The generated token IDs.
    """
    assert len(input_ids.shape) == 2, (
        f'Input shape should be (batch_size, sequence_length), got {input_ids.shape}'
    )
    model.eval()
    context_length = model.config.context_length
    for _ in range(max_new_tokens):
        truncated_input_ids = input_ids[:, -context_length:]
        logits: torch.Tensor = model(truncated_input_ids)[:, -1]
        next_token_id = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    return input_ids


def get_log_probs(logits, token_id):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs)
    return log_probs.gather(1, token_id)


@torch.no_grad()
def beam_search_simple(
    model,
    input_ids,
    max_new_tokens,
    sampling: Literal['greedy', 'topk', 'topp'],
    num_beams=1,
    temperature=1.0,
    top_k=20,
    top_p=0.5,
):
    """
    Generate text using beam search with simple sampling strategies. This is a simple implementation
    which only samples one token at each time step in each beam.

    Args:
        model (torch.nn.Module): The model to use for text generation.
        input_ids (torch.Tensor): The input token IDs.
        max_new_tokens (int): The maximum number of new tokens to generate.
        sampling (Literal['greedy', 'topk', 'topp']): The sampling strategy to use.
        num_beams (int): The number of beams to use.
        temperature (float): The temperature to use for sampling.
        top_k (int): The top-k value to use for top-k sampling.
        top_p (float): The top-p value to use for top-p sampling.

    Returns:
        torch.Tensor: The generated token IDs.
    """
    assert len(input_ids.shape) == 2, (
        f'Input shape should be (batch_size, sequence_length), got {input_ids.shape}'
    )
    model.eval()
    context_length = model.config.context_length
    truncated_input_ids = input_ids[:, -context_length:]
    logits: torch.Tensor = model(truncated_input_ids)[:, -1]

    assert sampling in ['greedy', 'topk', 'topp'], f'Invalid sampling method: {sampling}'
    match sampling:
        case 'greedy':
            next_tokens_id = greedy_sampling(logits, num_beams)
        case 'topk':
            next_tokens_id = top_k_sampling(logits, top_k, temperature, num_beams)
        case 'topp':
            next_tokens_id = top_p_sampling(logits, top_p, temperature, num_beams)

    beams = [
        torch.cat([input_ids, next_token_id], dim=-1)
        for next_token_id in torch.split(next_tokens_id, 1, dim=1)
    ]
    beams_avg_probs = [
        get_log_probs(logits, next_token_id)
        for next_token_id in torch.split(next_tokens_id, 1, dim=1)
    ]
    for step in range(max_new_tokens - 1):
        for i in range(num_beams):
            input_ids = beams[i]
            logits = model(input_ids)[:, -1]
            match sampling:
                case 'greedy':
                    next_token_id = greedy_sampling(logits, 1)
                case 'topk':
                    next_token_id = top_k_sampling(logits, top_k, temperature, 1)
                case 'topp':
                    next_token_id = top_p_sampling(logits, top_p, temperature, 1)

            beams[i] = torch.cat([input_ids, next_token_id], dim=-1)
            beams_avg_probs[i] += get_log_probs(logits, next_token_id) / (step + 2)

    beams_avg_probs = torch.stack(beams_avg_probs, dim=0).squeeze(-1)
    beams = torch.stack(beams, dim=0)
    max_indices = torch.argmax(beams_avg_probs, dim=0)
    input_ids = beams[max_indices, torch.arange(beams.size(1))]

    return input_ids


@torch.no_grad()
def beam_search_standard(): ...


# TODO
