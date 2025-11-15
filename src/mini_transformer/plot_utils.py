import numpy as np
import matplotlib.pyplot as plt


def plot_loss(
    num_epochs,
    train_losses,
    eval_losses,
    tokens_seen=None,
    figsize=(5, 3),
):
    epochs_seen = np.linspace(0, num_epochs, len(train_losses))
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(epochs_seen, train_losses, label='Training loss')
    ax1.plot(epochs_seen, eval_losses, linestyle='-.', label='Evaluation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    if tokens_seen:
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha=0)
        ax2.set_xlabel('Tokens seen')
    fig.tight_layout()
    plt.show()


def plot_loss_and_acc(
    num_epochs,
    train_losses,
    eval_losses,
    train_accs,
    eval_accs,
    samples_seen=None,
    figsize=(12, 6),
):
    epochs_seen = np.linspace(0, num_epochs, len(train_losses))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(epochs_seen, train_losses, label='Training loss')
    axes[0].plot(epochs_seen, eval_losses, linestyle='-.', label='Evaluation loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    if samples_seen:
        ax0_t = axes[0].twiny()
        ax0_t.plot(np.linspace(0, samples_seen, len(train_losses)), train_losses, alpha=0)
        ax0_t.set_xlabel('Samples seen')

    epochs_seen = np.linspace(0, num_epochs, len(train_accs))
    axes[1].plot(epochs_seen, train_accs, label='Training acc')
    axes[1].plot(epochs_seen, eval_accs, linestyle='-.', label='Evaluation acc')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Acc')
    if samples_seen:
        ax1_t = axes[1].twiny()
        ax1_t.plot(np.linspace(0, samples_seen, len(train_accs)), train_accs, alpha=0)
        ax1_t.set_xlabel('Samples seen')

    fig.tight_layout()
    plt.show()
