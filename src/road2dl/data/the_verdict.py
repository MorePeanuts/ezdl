import torch
from pathlib import Path
from torch.utils.data import Dataset
from ..protocol import Tokenizer


class TheVerdictDataset(Dataset):
    
    dataset_path = Path(__file__).parents[3] / 'datasets/the_verdict/the-verdict.txt'
    
    def __init__(self, tokenizer: Tokenizer, max_length: int, stride: int) -> None:
        """
        Initialize 'The Verdict' dataset with the given tokenizer, maximum length, and stride.
        
        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding the text.
            max_length (int): The maximum length of each input sequence.
            stride (int): The stride to use when sliding the window over the text.
        """
        self.input_ids = []
        self.target_ids = []
        
        raw_text = self.get_raw_text()
        token_ids = tokenizer.encode(raw_text)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
        
    @staticmethod
    def get_raw_text():
        assert TheVerdictDataset.dataset_path.exists(), f"Dataset file {TheVerdictDataset.dataset_path} does not exist."
        with open(TheVerdictDataset.dataset_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        return raw_text
