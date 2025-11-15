import requests
import torch
import pandas as pd
import zipfile
from pathlib import Path
from torch.utils.data import Dataset
from ..tokenizer.tokenizer_utils import Tokenizer


class SMSSpamCollection(Dataset):
    dataset_path = (
        Path(__file__).parents[3] / 'datasets/sms_spam_collection/sms_spam_collection.tsv'
    )
    zipfile_path = (
        Path(__file__).parents[3] / 'datasets/sms_spam_collection/sms_spam_collection.zip'
    )
    dataset_url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
    label_map = {'ham': 0, 'spam': 1}
    label_map_reverse = {0: 'ham', 1: 'spam'}

    def __init__(self, tokenizer: Tokenizer, max_length: int | None = None, pad_token_id=50256):
        self.data = SMSSpamCollection.get_balanced_dataframe()

        # pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data['text']]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        # pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]['label']
        return torch.tensor(encoded), torch.tensor(label)

    def _longest_encoded_length(self):
        return len(max(self.encoded_texts, key=len))

    @staticmethod
    def download_and_unzip_raw_data():
        # downloading the file
        response = requests.get(SMSSpamCollection.dataset_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(SMSSpamCollection.zipfile_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # unzipping the file
        with zipfile.ZipFile(SMSSpamCollection.zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(SMSSpamCollection.dataset_path.parent)

        original_file_path = SMSSpamCollection.dataset_path.parent / 'SMSSpamCollection'
        original_file_path.rename(SMSSpamCollection.dataset_path)

    @staticmethod
    def get_raw_dataframe():
        if not SMSSpamCollection.dataset_path.exists():
            SMSSpamCollection.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            SMSSpamCollection.download_and_unzip_raw_data()
        return pd.read_csv(
            SMSSpamCollection.dataset_path, sep='\t', header=None, names=['label', 'text']
        )

    @staticmethod
    def get_balanced_dataframe():
        raw_df = SMSSpamCollection.get_raw_dataframe()

        # count the instances of 'spam'
        num_spam = raw_df[raw_df['label'] == 'spam'].shape[0]

        # randomly sample 'ham' instances to match the number of 'spam'
        ham_subset = raw_df[raw_df['label'] == 'ham'].sample(n=num_spam, random_state=42)

        # combine ham with spam
        balanced_df = pd.concat([ham_subset, raw_df[raw_df['label'] == 'spam']])
        balanced_df['label'] = balanced_df['label'].map(SMSSpamCollection.label_map)  # type: ignore

        return balanced_df

    @classmethod
    def digit2label(cls, digit):
        return cls.label_map_reverse[digit]

    @classmethod
    def label2digit(cls, label):
        return cls.label_map[label]
