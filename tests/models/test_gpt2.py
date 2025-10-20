import torch
import pytest
import tiktoken
from road2dl.models.gpt2 import (
    GPT2Config,
    GPT2LayerNorm,
    GPT2TransformerBlock,
    GPT2Model,
    GPT2ModelForCausalLM,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding('gpt2')
    

@pytest.fixture
def config():
    return GPT2Config()
    

def test_gpt2_layer_norm():
    batch_example = torch.randn(2, 5)
    ln = GPT2LayerNorm(GPT2Config(embd_dim=5))
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_transformer_block(config):
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = GPT2TransformerBlock(config)
    output = block(x)
    assert output.shape == x.shape
    

def test_gpt2_model(tokenizer, config):
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    assert tuple(batch.shape) == (2, 4)
    
    model = GPT2Model(config)
    model_outputs = model(batch)
    assert tuple(model_outputs.shape) == (2, 4, config.n_embd)


def test_gpt2_124m_memory(config):
    model = GPT2ModelForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 163_009_536
    assert model.model.tok_embd.weight.shape == torch.Size((config.vocab_size, config.n_embd))
    assert model.lm_head.weight.shape == torch.Size((config.vocab_size, config.n_embd))
    real_params = total_params - sum(p.numel() for p in model.lm_head.parameters())
    assert real_params == 124_412_160
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 ** 2)
    assert abs(total_size_mb - 621.83) < 1e-2


def test_generate_text_simple(tokenizer, config):
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    assert encoded_tensor.shape == torch.Size((1, 4))
    
    model = GPT2ModelForCausalLM(config)
    output = generate_text_simple(
        model=model,
        input_ids=encoded_tensor,
        max_new_tokens=6,
        context_length=config.context_length
    )
    print("Output:", output)
    assert len(output[0]) == 10
    decoded_text = tokenizer.decode(output.squeeze(0).tolist())
    print(decoded_text)


def test_text_token_ids_transformation(tokenizer, config):
    model = GPT2ModelForCausalLM(config)
    start_context = 'Every effort moves you'
    token_ids = generate_text_simple(
        model=model,
        input_ids=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_length=config.context_length
    )
    output_text = token_ids_to_text(token_ids, tokenizer)
    assert isinstance(output_text, str)
