import typer
import torch
import torch.nn as nn
from typing import Literal, Annotated
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from mini_transformer.data.fashion_mnist import FashionMNIST
from mini_transformer.scratch.linear_model import (
    MultiLinearRegression,
    MLPForClassification
)
from mini_transformer.models.lenet import LeNetModelForClassification
from mini_transformer.models.alexnet import AlexNetModelForClassification, AlexNetConfig
from mini_transformer.models.vgg import VGGModelForClassification, VGGConfig
from mini_transformer.models.nin import NiNModelForClassification, NiNConfig
from mini_transformer.models.googlenet import GoogLeNetModelForClassification, GoogLeNetConfig
from mini_transformer.models.loss_utils import CrossEntropyLoss
from mini_transformer.trainer import train_classification_model_simple
from mini_transformer.device_utils import get_single_device
from mini_transformer.plot_utils import plot_loss_and_acc


def train_classifier(
    model,
    lr=0.1,
    num_epochs=10,
    resize=(28, 28),
    batch_size=256,
    eval_freq=128
):
    device = get_single_device('gpu')
    train_data = FashionMNIST(resize)
    eval_data = FashionMNIST(resize, train=False)
    model.to(device)
    loss = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=batch_size,
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
        eval_freq=eval_freq
    )
    
    plot_loss_and_acc(num_epochs, train_losses, eval_losses, train_accs, eval_accs)


def train_softmax_classifier():
    model = nn.Sequential(
        nn.Flatten(),
        MultiLinearRegression(
            in_features=28*28,
            out_features=10,
        )
    )
    train_classifier(
        model, 
        lr=0.1,
        num_epochs=10,
        batch_size=256,
        eval_freq=128
    )


def train_mlp_classifier():
    model = nn.Sequential(
        nn.Flatten(),
        MLPForClassification(
            in_features=28*28,
            num_classes=10,
            hidden_dim=256,
        )
    )
    train_classifier(
        model, 
        lr=0.1,
        num_epochs=10,
        batch_size=256,
        eval_freq=128,
    )
    

def train_lenet_classifier():
    model = LeNetModelForClassification.from_default_config()
    train_classifier(
        model,
        lr=0.1,
        num_epochs=10,
        batch_size=128,
        eval_freq=256,
    )
    
    
def train_alexnet_classifier():
    config = AlexNetConfig(
        num_classes=10,
        in_features=[1, 224, 224],
    )
    model = AlexNetModelForClassification(config)
    train_classifier(
        model,
        lr=0.01,
        num_epochs=10,
        resize=(224, 224),
        batch_size=128,
        eval_freq=256,
    )        
    
    
def train_vgg_classifier():
    config = VGGConfig(
        num_classes=10,
        in_features=[1, 224, 224],
    )
    model = VGGModelForClassification(config)
    train_classifier(
        model,
        lr=0.01,
        num_epochs=10,
        resize=(224, 224),
        batch_size=128,
        eval_freq=256,
    ) 


def train_nin_classifier():
    config = NiNConfig(
        num_classes=10,
        in_features=[1, 224, 224],
    )
    model = NiNModelForClassification(config)
    train_classifier(
        model,
        lr=0.05,
        num_epochs=10,
        resize=(224, 224),
        batch_size=128,
        eval_freq=256,
    )    
    
    
def train_googlenet_classifier():
    config = GoogLeNetConfig(
        num_classes=10,
        in_features=[1, 224, 224],
    )
    model = GoogLeNetModelForClassification(config)
    train_classifier(
        model,
        lr=0.01,
        num_epochs=10,
        resize=(224, 224),
        batch_size=128,
        eval_freq=256,
    )    
    
    
def main(
    task: Annotated[Literal["softmax", "mlp", "lenet", "alexnet", "vgg", "nin", "googlenet"], typer.Argument(help="The task to train.")],
):
    match task:
        case 'softmax':
            train_softmax_classifier()
        case 'mlp':
            train_mlp_classifier()
        case 'lenet':
            train_lenet_classifier()
        case 'alexnet':
            train_alexnet_classifier()
        case 'vgg':
            train_vgg_classifier()
        case 'nin':
            train_nin_classifier()
        case 'googlenet':
            train_googlenet_classifier()
        case _:
            raise ValueError(f"Invalid task: {task}")
            
            
if __name__ == "__main__":
    typer.run(main)
