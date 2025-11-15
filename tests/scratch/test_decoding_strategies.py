import pytest
import torch
import tiktoken
from pathlib import Path
from mini_transformer.scratch.decoding_strategies import (
    greedy_sampling,
    top_k_sampling,
    top_p_sampling,
    get_log_probs,
    greedy_search,
    beam_search_simple,
)
from mini_transformer.models.gpt2 import (
    GPT2ModelForCausalLM,
)


@pytest.fixture
def logits():
    batch_size = 2
    vocab_size = 8
    return torch.randn(batch_size, vocab_size)


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding('gpt2')


@pytest.fixture
def input_ids(tokenizer):
    batch = []
    batch.append(torch.tensor(tokenizer.encode('Every effort moves you')))
    batch.append(torch.tensor(tokenizer.encode('Every day holds a')))
    input_ids = torch.stack(batch, dim=0)
    return input_ids


def test_greedy_sampling(logits):
    beams = 3
    next_tokens_id = greedy_sampling(logits, beams)
    assert next_tokens_id.shape == torch.Size([2, beams])


def test_top_k_sampling(logits):
    beams = 3
    next_tokens_id = top_k_sampling(logits, 4, 1, beams)
    assert next_tokens_id.shape == torch.Size([2, beams])


def test_top_p_sampling(logits):
    beams = 3
    next_tokens_id = top_p_sampling(logits, 0.65, 1, beams)
    assert next_tokens_id.shape == torch.Size([2, beams])


def test_get_log_probs(logits):
    token_id = torch.tensor([[1], [2]])
    log_probs = get_log_probs(logits, token_id)
    assert log_probs.shape == torch.Size([2, 1])


@pytest.mark.skipif(
    condition=not (
        (Path(__file__).parents[2] / 'models/gpt2_124M/model.pth').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.pt').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.bin').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.safetensors').exists()
    ),
    reason='Model not found',
)
def test_greedy_search(input_ids, tokenizer):
    model_directory = Path(__file__).parents[2] / 'models/gpt2_124M'
    model = GPT2ModelForCausalLM.from_pretrained(model_directory)
    output_ids = greedy_search(model, input_ids, 20)
    assert output_ids.shape == torch.Size([2, 24])
    print(tokenizer.decode(output_ids[0].tolist()))
    print(tokenizer.decode(output_ids[1].tolist()))

    output_ids2 = greedy_search(model, input_ids, 20)
    assert torch.allclose(output_ids, output_ids2)
    print(tokenizer.decode(output_ids2[0].tolist()))
    print(tokenizer.decode(output_ids2[1].tolist()))


@pytest.mark.skipif(
    condition=not (
        (Path(__file__).parents[2] / 'models/gpt2_124M/model.pth').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.pt').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.bin').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.safetensors').exists()
    ),
    reason='Model not found',
)
def test_beam_search_simple(input_ids, tokenizer):
    model_directory = Path(__file__).parents[2] / 'models/gpt2_124M'
    model = GPT2ModelForCausalLM.from_pretrained(model_directory)
    output_ids = beam_search_simple(model, input_ids, 20, 'greedy', 2)
    assert output_ids.shape == torch.Size([2, 24])
    print(tokenizer.decode(output_ids[0].tolist()))
    print(tokenizer.decode(output_ids[1].tolist()))

    output_ids = beam_search_simple(model, input_ids, 20, 'topk', 2)
    assert output_ids.shape == torch.Size([2, 24])
    print(tokenizer.decode(output_ids[0].tolist()))
    print(tokenizer.decode(output_ids[1].tolist()))

    output_ids = beam_search_simple(model, input_ids, 20, 'topp', 2)
    assert output_ids.shape == torch.Size([2, 24])
    print(tokenizer.decode(output_ids[0].tolist()))
    print(tokenizer.decode(output_ids[1].tolist()))


@pytest.mark.skip(reason='Not implemented')
def test_beam_search_standard(input_ids, tokenizer):
    pass
