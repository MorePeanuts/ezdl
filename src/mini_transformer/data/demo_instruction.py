import json
import torch
import requests
from pathlib import Path
from torch.utils.data import Dataset
from .data_utils import format_instruction


class DemoInstructionDataset(Dataset):
    
    dataset_path = Path(__file__).parents[3] / 'datasets/demo_instruction/demo_instruction.json'
    dataset_url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    
    def __init__(self, tokenizer):
        self.data = DemoInstructionDataset.load_raw_json_data()
        self.encoded_instructions = []
        self.encoded_responses = []
        for entry in self.data:
            instruction, response = format_instruction(entry)
            self.encoded_instructions.append(tokenizer.encode(instruction))
            self.encoded_responses.append(tokenizer.encode(response))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.encoded_instructions[idx], self.encoded_responses[idx]
        
    @staticmethod
    def download_raw_data():
        DemoInstructionDataset.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(DemoInstructionDataset.dataset_url, timeout=30)
        response.raise_for_status()
        text = response.text
        with DemoInstructionDataset.dataset_path.open('w', encoding='utf-8') as f:
            f.write(text)
    
    @staticmethod
    def load_raw_json_data():
        if not DemoInstructionDataset.dataset_path.exists():
            DemoInstructionDataset.download_raw_data()
        with DemoInstructionDataset.dataset_path.open('r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def instruction_collate(
        batch, 
        pad_token_id=50256, 
        ignore_index=-100,
        allowed_max_length=None
    ):
        batch = [instruction + response for instruction, response in batch]
        batch_max_length = max(len(item) + 1 for item in batch)
        inputs_list, targets_list = [], []
        
        for item in batch:
            pad_item = item.copy()
            pad_item += [pad_token_id]
            pad = pad_item + [pad_token_id] * (batch_max_length - len(pad_item))
            inputs = torch.tensor(pad[:-1])
            targets = torch.tensor(pad[1:])
            
            mask = targets == pad_token_id
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                targets[indices[1:]] = ignore_index
            if allowed_max_length is not None:
                inputs = inputs[:allowed_max_length]
                targets = targets[:allowed_max_length]
                
            inputs_list.append(inputs)
            targets_list.append(targets)
            
        inputs_tensor = torch.stack(inputs_list)
        targets_tensor = torch.stack(targets_list)
        
        return inputs_tensor, targets_tensor
