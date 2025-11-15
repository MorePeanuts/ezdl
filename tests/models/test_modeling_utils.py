import torch
import pytest
import tiktoken
import tempfile
from pathlib import Path
from mini_transformer.models.gpt2 import GPT2ModelForCausalLM


@pytest.mark.skipif(
    condition=not (
        (Path(__file__).parents[2] / 'models/gpt2_124M/model.pth').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.pt').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.bin').exists()
        or (Path(__file__).parents[2] / 'models/gpt2_124M/model.safetensors').exists()
    ),
    reason='Model not found',
)
def test_model_from_pretrained():
    model_directory = Path(__file__).parents[2] / 'models/gpt2_124M'
    GPT2ModelForCausalLM.from_pretrained(model_directory)


def test_model_save_pretrained_and_load():
    model = GPT2ModelForCausalLM.from_default_config()
    model.eval()
    tokenizer = tiktoken.get_encoding('gpt2')
    tokenized_text = tokenizer.encode('Hello, world!')
    input_ids = torch.tensor(tokenized_text).unsqueeze(0)
    output = model(input_ids)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded_model = GPT2ModelForCausalLM.from_pretrained(tmpdir)
        loaded_model.eval()
        for module_1, module_2 in zip(model.modules(), loaded_model.modules()):
            if hasattr(module_1, 'weight') and hasattr(module_2, 'weight'):
                assert module_1.weight.shape == module_2.weight.shape
                assert torch.allclose(module_1.weight, module_2.weight)  # type: ignore
        assert torch.allclose(output, loaded_model(input_ids))
