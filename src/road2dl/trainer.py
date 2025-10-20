import torch
import torch.nn as nn


class Trainer:
    
    def __init__(
        self,
        model: nn.Module | None = None,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        model_init = None,
        compute_loss_func = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = None,
        optimizer_cls_and_kwargs = None,
    ):
        ...
        