import tiktoken
from road2dl.data.the_verdict import TheVerdictDataset
from torch.utils.data import DataLoader


def test_the_verdict_dataset():
    tokenizer = tiktoken.get_encoding('gpt2')
    max_length, batch_size = 4, 8
    dataset = TheVerdictDataset(tokenizer, max_length, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        drop_last=True,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    assert tuple(inputs.shape) == (batch_size, max_length)
    assert tuple(targets.shape) == (batch_size, max_length)
