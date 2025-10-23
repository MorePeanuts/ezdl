import typer
import torch
import matplotlib.pyplot as plt
from typing import Literal, Annotated
from torch.utils.data import (
    DataLoader,
    random_split,
)
from torch.optim import AdamW
from ezdl.trainer_seq2seq import (
    train_gpt2_simple,
    train_gpt2_classifier_simple
)
from ezdl.models.gpt2 import (
    GPT2Config,
    GPT2ModelForCausalLM,
    GPT2ModelForClassification,
    calc_accuracy_dataloader
)
from ezdl.device_utils import get_single_device
from ezdl.data.the_verdict import TheVerdictDataset
from ezdl.data.sms_spam_collection import SMSSpamCollection
from ezdl.tokenizer.bpe import BPETokenizerTiktoken


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
    
    epochs_seen = torch.linspace(0, num_epochs, len(train_losses))
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label='Training loss')
    ax1.plot(epochs_seen, eval_losses, linestyle='-.', label='Evaluation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel('Tokens seen')
    fig.tight_layout()
    plt.show()
    
    
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
    
    epochs_seen = torch.linspace(0, num_epochs, len(train_losses))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(epochs_seen, train_losses, label='Training loss')
    axes[0].plot(epochs_seen, eval_losses, linestyle='-.', label='Evaluation loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    ax0_t = axes[0].twiny()
    ax0_t.plot(torch.linspace(0, samples_seen, len(train_losses)), train_losses, alpha=0)
    ax0_t.set_xlabel('Samples seen')
    
    epochs_seen = torch.linspace(0, num_epochs, len(train_accs))
    axes[1].plot(epochs_seen, train_accs, label='Training acc')
    axes[1].plot(epochs_seen, eval_accs, linestyle='-.', label='Evaluation acc')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Acc')
    ax1_t = axes[1].twiny()
    ax1_t.plot(torch.linspace(0, samples_seen, len(train_accs)), train_accs, alpha=0)
    ax1_t.set_xlabel('Samples seen')
    
    fig.tight_layout()
    plt.show()
    
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
    
    
def main(
    task: Annotated[Literal['generation', 'classification'], typer.Argument(
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
        case _:
            raise ValueError(f'Unknown task: {task}')
    
    
if __name__ == '__main__':
    typer.run(main)
