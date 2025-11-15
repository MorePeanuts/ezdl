import torch
import pytest
import tiktoken
from torch.utils.data import DataLoader
from mini_transformer.models.gpt2 import (
    GPT2Config,
    GPT2LayerNorm,
    GPT2TransformerBlock,
    GPT2Model,
    GPT2ModelForCausalLM,
    GPT2ModelForClassification,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
    calc_accuracy_dataloader,
    calc_loss_batch,
    calc_loss_dataloader,
)
from mini_transformer.data.sms_spam_collection import SMSSpamCollection
from mini_transformer.data.the_verdict import TheVerdictDataset


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding('gpt2')


@pytest.fixture
def batch_input_ids(tokenizer):
    txt1 = 'Every effort moves you'
    txt2 = 'Every day holds a'
    batch = torch.tensor([tokenizer.encode(txt1), tokenizer.encode(txt2)])
    return batch


@pytest.fixture
def sms_input_ids(tokenizer):
    text = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def test_gpt2_layer_norm():
    batch_example = torch.randn(2, 5)
    ln = GPT2LayerNorm(GPT2Config(embd_dim=5))
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_transformer_block():
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = GPT2TransformerBlock(GPT2Config())
    output = block(x)
    assert output.shape == x.shape


def test_gpt2_model(tokenizer, batch_input_ids):
    assert tuple(batch_input_ids.shape) == (2, 4)

    model = GPT2Model.from_default_config()
    model_outputs = model(batch_input_ids)
    assert tuple(model_outputs.shape) == (2, 4, model.config.embd_dim)


def test_gpt2_124m_memory():
    model = GPT2ModelForCausalLM.from_default_config()
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 163_009_536
    assert model.model.tok_embd.weight.shape == torch.Size((config.vocab_size, config.embd_dim))
    assert model.lm_head.weight.shape == torch.Size((config.vocab_size, config.embd_dim))
    real_params = total_params - sum(p.numel() for p in model.lm_head.parameters())
    assert real_params == 124_412_160
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024**2)
    assert abs(total_size_mb - 621.83) < 1e-2


def test_generate_text_simple(tokenizer):
    start_context = 'Hello, I am'
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    assert encoded_tensor.shape == torch.Size((1, 4))

    model = GPT2ModelForCausalLM.from_default_config()
    config = model.config
    output = generate_text_simple(
        model=model,
        input_ids=encoded_tensor,
        max_new_tokens=6,
        context_length=config.context_length,
    )
    print('Output:', output)
    assert tuple(output.shape) == (1, 10)
    decoded_text = tokenizer.decode(output.squeeze(0).tolist())
    print(decoded_text)


def test_classificate_text(tokenizer, sms_input_ids):
    model = GPT2ModelForClassification.from_default_config()
    model.eval()
    context_length = model.config.context_length
    num_labels = model.config.num_labels
    output = model(sms_input_ids[:, -context_length:])
    output = output[:, -1]
    assert output.shape == torch.Size([1, num_labels])
    print(output)


def test_text_token_ids_transformation(tokenizer):
    model = GPT2ModelForCausalLM.from_default_config()
    config = model.config
    start_context = 'Every effort moves you'
    token_ids = generate_text_simple(
        model=model,
        input_ids=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_length=config.context_length,
    )
    output_text = token_ids_to_text(token_ids, tokenizer)
    assert isinstance(output_text, str)


def test_calc_acc_dataloader(tokenizer):
    model = GPT2ModelForClassification.from_default_config()
    model.eval()
    dataset = SMSSpamCollection(tokenizer)
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    acc = calc_accuracy_dataloader(dataloader, model, torch.device('cpu'), 10)
    assert acc >= 0 and acc <= 1.0
    print(f'Acc: {acc * 100:.2f}%')


def test_calc_loss_generation(tokenizer):
    # Test model for causal language modeling
    model = GPT2ModelForCausalLM.from_default_config()
    model.eval()
    dataset = TheVerdictDataset(tokenizer, max_length=256, stride=256)
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    input_batch, target_batch = next(iter(dataloader))
    loss = calc_loss_batch(input_batch, target_batch, model, torch.device('cpu'))
    assert loss.shape == torch.Size([])

    avg_loss = calc_loss_dataloader(dataloader, model, torch.device('cpu'), 10)
    assert isinstance(avg_loss, float)


def test_calc_loss_classifacation(tokenizer):
    # Test model for classification
    model = GPT2ModelForClassification.from_default_config()
    model.eval()
    dataset = SMSSpamCollection(tokenizer)
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    input_batch, target_batch = next(iter(dataloader))
    loss = calc_loss_batch(input_batch, target_batch, model, torch.device('cpu'), True)
    assert loss.shape == torch.Size([])

    avg_loss = calc_loss_dataloader(dataloader, model, torch.device('cpu'), 10, True)
    assert isinstance(avg_loss, float)
