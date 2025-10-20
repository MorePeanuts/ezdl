import pytest
import torch
import tiktoken
from road2dl.device_utils import get_single_device
from road2dl.models.gpt2 import GPT2Config, GPT2ModelForCausalLM


@pytest.fixture
def model():
    config = GPT2Config()
    return GPT2ModelForCausalLM(config)


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def input_ids(tokenizer):
    prompt = "Hello, I am"
    return torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)


@pytest.mark.xfail(
    condition=(
        torch.backends.mps.is_available() and
        get_single_device('gpu') == torch.device('mps')
    ),
    reason='MPS backend cannot pass this test.'
)
def test_forward_consistency(model, input_ids):
    cpu = get_single_device("cpu")
    model_cpu = model.to(cpu)
    with torch.no_grad():
        cpu_output = model_cpu(input_ids.to(cpu))

    gpu = get_single_device("gpu")
    model_gpu = model.to(gpu)
    with torch.no_grad():
        gpu_output = model_gpu(input_ids.to(gpu))

    diff = torch.abs(cpu_output - gpu_output.cpu()).max().item()
    assert diff < 1e-3, (
        f"Forward consistency failed with max diff {diff}"
    )


@pytest.mark.xfail(
    condition=(
        torch.backends.mps.is_available() and
        get_single_device('gpu') == torch.device('mps')
    ),
    reason='MPS backend cannot pass this test.'
)
def test_gradient_consistency(model, input_ids):
    model.train()

    cpu = get_single_device("cpu")
    model_cpu = model.to(cpu)
    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=4e-4)
    optimizer_cpu.zero_grad()
    output_cpu = model_cpu(input_ids[:, :-1].to(cpu))
    loss_cpu = torch.nn.functional.cross_entropy(output_cpu.flatten(0, 1), input_ids[:, 1:].flatten().to(cpu))
    loss_cpu.backward()
    optimizer_cpu.step()
    grad_cpu = [p.grad.clone() for p in model_cpu.parameters() if p.grad is not None]

    gpu = get_single_device("gpu")
    model_gpu = model.to(gpu)
    optimizer_gpu = torch.optim.AdamW(model_gpu.parameters(), lr=4e-4)
    optimizer_gpu.zero_grad()
    output_gpu = model_gpu(input_ids[:, :-1].to(gpu))
    loss_gpu = torch.nn.functional.cross_entropy(output_gpu.flatten(0, 1), input_ids[:, 1:].flatten().to(gpu))
    loss_gpu.backward()
    optimizer_gpu.step()
    grad_gpu = [
        p.grad.cpu().clone() for p in model_gpu.parameters() if p.grad is not None
    ]

    max_grad_diff = 0
    for c_grad, g_grad in zip(grad_cpu, grad_gpu):
        diff = torch.abs(c_grad - g_grad).max().item()
        max_grad_diff = max(max_grad_diff, diff)

    assert max_grad_diff < 1e-3, (
        f"Gradient consistency failed with max diff {max_grad_diff}"
    )
