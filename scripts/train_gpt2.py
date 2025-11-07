import typer
import torch
import matplotlib.pyplot as plt
from typing import Literal, Annotated
from torch.utils.data import (
    DataLoader,
    random_split,
)
from torch.optim import AdamW
from mini_transformer.trainer_seq2seq import (
    train_gpt2_simple,
    train_gpt2_classifier_simple
)
from mini_transformer.models.gpt2 import (
    GPT2Config,
    GPT2ModelForCausalLM,
    GPT2ModelForClassification,
    calc_accuracy_dataloader
)
from mini_transformer.device_utils import get_single_device
from mini_transformer.data.the_verdict import TheVerdictDataset
from mini_transformer.data.sms_spam_collection import SMSSpamCollection
from mini_transformer.data.demo_instruction import DemoInstructionDataset
from mini_transformer.tokenizer.bpe import BPETokenizerTiktoken
from mini_transformer.plot_utils import plot_loss, plot_loss_and_acc


def train_gpt2_for_generation():
    config = GPT2Config(context_length=256)
    model = GPT2ModelForCausalLM(config)
    device = get_single_device('gpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    tokenizer = BPETokenizerTiktoken('gpt2')
    num_epochs = 10
    dataset = TheVerdictDataset(tokenizer, config.context_length, config.context_length)
    train_size = int(0.9*len(dataset))
    eval_size = int(0.1*len(dataset))
    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False
    )
    
    train_losses, eval_losses, tokens_seen = train_gpt2_simple(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        device,
        num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context='Every effort moves you',
        tokenizer=tokenizer
    )
    
    plot_loss(num_epochs, train_losses, eval_losses, tokens_seen)
    
    
def train_gpt2_for_classification():
    from pathlib import Path
    
    model_directory = Path(__file__).parents[1] / 'models/gpt2_124M'
    model = GPT2ModelForClassification.from_pretrained(model_directory)
    device = get_single_device('gpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    tokenizer = BPETokenizerTiktoken('gpt2')
    num_epochs = 5
    dataset = SMSSpamCollection(tokenizer)
    train_size = int(0.7*len(dataset))
    eval_size = int(0.1*len(dataset))
    test_size = len(dataset) - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = random_split(
        dataset, [train_size, eval_size, test_size]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False
    )
    
    train_losses, eval_losses, train_accs, eval_accs, samples_seen = train_gpt2_classifier_simple(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        device,
        num_epochs,
        eval_freq=50,
        eval_iter=5
    )
    
    plot_loss_and_acc(num_epochs, train_losses, eval_losses, train_accs, eval_accs, samples_seen)
    
    test_acc = calc_accuracy_dataloader(
        test_dataloader,
        model,
        device,
    )
    print(f'Test accuracy: {test_acc*100:.2f}%')
    
    model.eval()
    context_length = model.config.context_length
    max_length = dataset.max_length
    
    encoded = [
        tokenizer.encode(
            "You are a winner you have been specially"
            " selected to receive $1000 cash or a $2000 award."
        )[:min(max_length, context_length)],
        tokenizer.encode(
            "Hey, just wanted to check if we're still on"
            " for dinner tonight? Let me know!"
        )[:min(max_length, context_length)],
    ]
    seq_len = len(max(encoded, key=len))
    pad_token_id = 50256
    encoded = [seq + [pad_token_id] * (seq_len - len(seq)) for seq in encoded]
    input_ids = torch.tensor(encoded, device=device)
    with torch.no_grad():
        logits = model(input_ids)[:, -1]
    predictions = torch.argmax(logits, dim=-1)
    labels = [dataset.digit2label(int(idx)) for idx in predictions]
    for prediction, label in zip(predictions, labels):
        print(f'Prediction: {prediction}, Label: {label}')
        
        
def train_gpt2_instruction_fine_tuning():
    from pathlib import Path
    
    model_directory = Path(__file__).parents[1] / 'models/gpt2_124M'
    model = GPT2ModelForCausalLM.from_pretrained(model_directory)
    device = get_single_device('gpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    tokenizer = BPETokenizerTiktoken('gpt2')
    num_epochs = 2 
    dataset = DemoInstructionDataset(tokenizer)
    train_size = int(0.85*len(dataset))
    eval_size = int(0.1*len(dataset))
    test_size = len(dataset) - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = random_split(
        dataset, [train_size, eval_size, test_size]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=dataset.instruction_collate,
        shuffle=True,
        drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=8,
        collate_fn=dataset.instruction_collate,
        shuffle=False,
        drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=dataset.instruction_collate,
        shuffle=False,
        drop_last=False
    )
    
    train_losses, eval_losses, tokens_seen = train_gpt2_simple(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        device,
        num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=tokenizer.decode(test_dataset[0][0]),
        tokenizer=tokenizer
    )
    
    plot_loss(num_epochs, train_losses, eval_losses, tokens_seen)
    
    
def main(
    task: Annotated[Literal['generation', 'classification', 'instruction'], typer.Argument(
        help='The task to train the model for.'
    )], 
    option: Annotated[int, typer.Argument(
        help='The option to use for training.'
    )] = 42,
):
    """
    Train a GPT-2 model.
    """
    match task:
        case 'generation':
            train_gpt2_for_generation()
        case 'classification':
            train_gpt2_for_classification()
        case 'instruction':
            train_gpt2_instruction_fine_tuning()
        case _:
            raise ValueError(f'Unknown task: {task}')
    
    
if __name__ == '__main__':
    typer.run(main)
