import torch
import matplotlib.pyplot as plt
from torch.utils.data import (
    DataLoader,
    random_split,
)
from torch.optim import AdamW
from ezdl.trainer_seq2seq import train_gpt2_simple
from ezdl.models.gpt2 import (
    GPT2Config,
    GPT2ModelForCausalLM,
    GPT2ModelForClassification,
)
from ezdl.device_utils import get_single_device
from ezdl.data.the_verdict import TheVerdictDataset
from ezdl.tokenizer.bpe import BPETokenizerTiktoken


if __name__ == '__main__':
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
    num_epochs = 10
    
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
