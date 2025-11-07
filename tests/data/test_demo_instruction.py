import pytest
import tiktoken
import torch
from torch.utils.data import DataLoader
from mini_transformer.data.demo_instruction import DemoInstructionDataset


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding('gpt2')


def test_demo_instruction_dataset(tokenizer):
    dataset = DemoInstructionDataset(tokenizer)
    assert len(dataset) == 1100
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=dataset.instruction_collate,
        shuffle=True,
        drop_last=True,
    )
    inputs, targets = next(iter(dataloader))
    assert len(inputs.shape) == 2 and len(targets.shape) == 2
    assert isinstance(inputs, torch.Tensor)
    
    print(tokenizer.decode([
        item for item in inputs[0].tolist() if item >= 0
    ]))
    print(tokenizer.decode([
        item for item in targets[0].tolist() if item >= 0
    ]))
    