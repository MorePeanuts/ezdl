import torch
import torch.nn as nn
from tqdm import tqdm


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
        
    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
    ):
        if resume_from_checkpoint is not None:
            raise NotImplementedError
            
        
        