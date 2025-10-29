import torch
import typer
import torch.nn as nn
from typing import Literal, Annotated
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ezdl.trainer import (
    train_regression_model_simple,
    train_classification_model_simple
)
from ezdl.data.data_utils import SyntheticRegressionData
from ezdl.data.fashion_mnist import FashionMNIST
from ezdl.scratch.linear_model import (
    LinearRegression,
    MultiLinearRegression,
    NaiveSoftmaxRegression,
    MLPForClassification
)
from ezdl.optimizer.gredient_descent import NaiveSGD
from ezdl.models.loss_utils import MSELoss, CrossEntropyLoss
from ezdl.device_utils import get_single_device
from ezdl.plot_utils import plot_loss, plot_loss_and_acc


def train_linear_regression_on_synthetic_data(naive_optim=False):
    lr = 0.03
    num_epochs = 3
    device = get_single_device('cpu')
    data = SyntheticRegressionData(w=[2, -3.4], b=4.2, total_samples=2048)
    model = LinearRegression(in_features=2)
    model.to(device)
    loss = MSELoss(reduction='mean')
    if naive_optim:
        optimizer = NaiveSGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_data, eval_data = train_test_split(
        data, test_size=0.1, train_size=0.9, random_state=42
    )
    train_dataloader = DataLoader(
        train_data, 
        batch_size=32, 
        shuffle=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=32,
        shuffle=False,
        drop_last=False
    )
    
    train_losses, eval_losses = train_regression_model_simple(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        loss,
        device, 
        num_epochs,
        eval_freq=8
    )
    
    plot_loss(num_epochs, train_losses, eval_losses)
    
    
def train_naive_softmax_regression_on_fashion_mnist():
    device = get_single_device('gpu')
    train_data = FashionMNIST()
    eval_data = FashionMNIST(train=False)
    model = nn.Sequential(
        nn.Flatten(),
        NaiveSoftmaxRegression(in_features=28*28, num_classes=10)
    )
    model.to(device)
    loss = model[1].cross_entropy_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 10
    train_dataloader = DataLoader(
        train_data, 
        batch_size=256, 
        shuffle=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=256,
        shuffle=False,
        drop_last=False
    )
    train_losses, eval_losses, train_accs, eval_accs = train_classification_model_simple(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        loss,
        device, 
        num_epochs,
        eval_freq=128
    )
    
    plot_loss_and_acc(num_epochs, train_losses, eval_losses, train_accs, eval_accs)
    
    
def train_softmax_regression_on_fashion_mnist():
    device = get_single_device('gpu')
    train_data = FashionMNIST()
    eval_data = FashionMNIST(train=False)
    model = nn.Sequential(
        nn.Flatten(),
        MultiLinearRegression(in_features=28*28, out_features=10)
    )
    model.to(device)
    loss = CrossEntropyLoss()
    # loss.forward = loss.scratch_forward
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 10
    train_dataloader = DataLoader(
        train_data, 
        batch_size=256, 
        shuffle=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=256,
        shuffle=False,
        drop_last=False
    )
    train_losses, eval_losses, train_accs, eval_accs = train_classification_model_simple(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        loss,
        device, 
        num_epochs,
        eval_freq=128
    )
    
    plot_loss_and_acc(num_epochs, train_losses, eval_losses, train_accs, eval_accs)
    
    
def train_mlp_classifier_on_fashion_mnist():
    device = get_single_device('gpu')
    train_data = FashionMNIST()
    eval_data = FashionMNIST(train=False)
    model = nn.Sequential(
        nn.Flatten(),
        MLPForClassification(
            in_features=28*28,
            num_classes=10,
            hidden_dim=256,
        )
    )
    model.to(device)
    loss = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 10
    train_dataloader = DataLoader(
        train_data, 
        batch_size=256, 
        shuffle=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=256,
        shuffle=False,
        drop_last=False
    )
    train_losses, eval_losses, train_accs, eval_accs = train_classification_model_simple(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        loss,
        device, 
        num_epochs,
        eval_freq=128
    )
    
    plot_loss_and_acc(num_epochs, train_losses, eval_losses, train_accs, eval_accs)
    
    
def main(
    task: Annotated[Literal['linear-regression', 'softmax-classifier', 'mlp-classifier'],
        typer.Argument(
            help="The task to train the model on"
        )],
    naive: Annotated[bool, typer.Option(help="Use naive implementation")] = False
):
    """
    Train linear model (linear-regression, softmax-classifier, mlp-classifier).
    """
    match (task, naive):
        case ('linear-regression', _):
            train_linear_regression_on_synthetic_data(naive)
        case ('softmax-classifier', False):
            train_softmax_regression_on_fashion_mnist()
        case ('softmax-classifier', True):
            train_naive_softmax_regression_on_fashion_mnist()
        case ('mlp-classifier', _):
            train_mlp_classifier_on_fashion_mnist()
        case _:
            raise ValueError(f"Invalid task: {task}")
    
    
if __name__ == '__main__':
    typer.run(main)
