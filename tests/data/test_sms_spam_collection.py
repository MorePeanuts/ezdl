import pytest
import tiktoken
from ezdl.data.sms_spam_collection import SMSSpamCollection


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
