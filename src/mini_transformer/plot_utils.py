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


def plot_data_points(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale='linear',
    yscale='linear',
    fmts=('-', 'm--', 'g-.', 'r:'),
    figsize=(10.5, 7.5),
    axes=None,
):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (
            hasattr(X, 'ndim')
            and X.ndim == 1
            or isinstance(X, list)
            and not hasattr(X[0], '__len__')
        )

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    plt.rcParams['figure.figsize'] = figsize
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    axes.set_xlabel(xlabel or 'x')
    axes.set_ylabel(ylabel or 'y')
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    plt.show()
