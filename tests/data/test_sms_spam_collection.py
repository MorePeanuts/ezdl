import pytest
import tiktoken
import torch
from mini_transformer.data.sms_spam_collection import SMSSpamCollection
from torch.utils.data import DataLoader


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding('gpt2')


def test_sms_spam_collection(tokenizer):
    dataset = SMSSpamCollection(tokenizer, 1024)
    assert len(dataset) == 1494
    raw_df = dataset.get_raw_dataframe()
    raw_counts = raw_df['label'].value_counts()
    assert raw_counts['ham'] == 4825
    assert raw_counts['spam'] == 747
    balanced_df = dataset.get_balanced_dataframe()
    balanced_counts = balanced_df['label'].value_counts() # type: ignore
    assert balanced_counts[0] == 747
    assert balanced_counts[1] == 747
    item = dataset[0]
    print(item[0], item[1])
    
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    input_batch, target_batch = next(iter(dataloader))
    assert tuple(input_batch.shape) == (batch_size, 1024)
    assert tuple(target_batch.shape) == (batch_size,)


def test_sms_spam_collection_label_digit_transform():
    indices = torch.Tensor([1, 0, 0, 1, 1])
    labels = [SMSSpamCollection.digit2label(int(idx)) for idx in indices]
    assert all(label in ['spam', 'ham'] for label in labels)
    
    indices_back = [SMSSpamCollection.label2digit(label) for label in labels]
    assert all(idx in [0, 1] for idx in indices_back)
