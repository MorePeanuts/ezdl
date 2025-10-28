import torch
import torch.nn as nn
from deprecated import deprecated
from tqdm import tqdm


class Trainer:
    
    def __init__(
        self,
        model: nn.Module | None = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        model_init = None,
        compute_loss_func = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = None,
        optimizer_cls_and_kwargs = None,
        preprocess_logits_for_metrics = None
    ):
        ...
        
    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
    ):
        if resume_from_checkpoint is not None:
            raise NotImplementedError
            
        
@deprecated(
    version='1.0.0',
    reason='This function is a simple implementation of regression trainer, and will be replaced by `Trainer` class in the future.'
)
def train_regression_model_simple(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    loss_func,
    device,
    num_epochs,
    eval_freq=None,
    eval_iter=None,
):
    print(f'Training regression model {model} on {device}...')
    
    eval_freq = len(train_dataloader) if eval_freq is None else eval_freq
    eval_iter = len(eval_dataloader) if eval_iter is None else eval_iter
    train_losses, eval_losses = [], []
    total_steps = num_epochs * len(train_dataloader)
    train_loss, global_step = 0.0, 0
    pbar = tqdm(total=total_steps)
    
    # main training loop
    for epoch in range(num_epochs):
        model.train()
        pbar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
        
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            loss = loss_func(model(x), y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            global_step += 1
            pbar.update(1)
            
            if global_step % eval_freq == 0:
                model.eval()
                pbar.set_description('Evaluating...')
                
                train_losses.append(train_loss / eval_freq)
                train_loss = 0
                
                with torch.no_grad():
                    eval_iterator = iter(eval_dataloader)
                    eval_loss = 0.0
                    for eval_step in range(eval_iter):
                        try:
                            x, y = next(eval_iterator)
                            x, y = x.to(device), y.to(device)
                            loss = loss_func(model(x), y)
                            eval_loss += loss.item()
                        except StopIteration:
                            break
                
                eval_losses.append(eval_loss / eval_step)
                pbar.write(f'Ep {epoch+1} (Step {global_step:06d}): '
                    f'Train loss {train_losses[-1]:.3f}, Val loss {eval_losses[-1]:.3f}')
                model.train()
                pbar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
                
    return train_losses, eval_losses
    
    
@deprecated(
    version='1.0.0',
    reason='This function is a simple implementation of classification trainer, and will be replaced by `Trainer` class in the future.'
)
def train_classification_model_simple(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    loss_func,
    device,
    num_epochs,
    eval_freq=None,
    eval_iter=None,
):  
    print(f'Training classifier on {device}...')
    
    eval_freq = len(train_dataloader) if eval_freq is None else eval_freq
    eval_iter = len(eval_dataloader) if eval_iter is None else eval_iter
    train_losses, eval_losses = [], []
    train_accs, eval_accs = [], []
    total_steps = num_epochs * len(train_dataloader)
    train_loss, global_step = 0.0, 0
    acc_counter = [0, 0]
    pbar = tqdm(total=total_steps)
    
    # main training loop
    for epoch in range(num_epochs):
        model.train()
        pbar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
        
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            global_step += 1
            pbar.update(1)
            
            preds = torch.argmax(y_hat, dim=-1).type(torch.int64)
            acc_counter[0] += torch.sum(preds == y).item()
            acc_counter[1] += y.size(0)
            
            if global_step % eval_freq == 0:
                model.eval()
                pbar.set_description('Evaluating...')
                
                train_losses.append(train_loss / eval_freq)
                train_loss = 0
                train_accs.append(acc_counter[0] / acc_counter[1])
                acc_counter = [0, 0]
                
                with torch.no_grad():
                    eval_iterator = iter(eval_dataloader)
                    eval_loss = 0.0
                    for eval_step in range(eval_iter):
                        try:
                            x, y = next(eval_iterator)
                            x, y = x.to(device), y.to(device)
                            loss = loss_func(model(x), y)
                            eval_loss += loss.item()
                            
                            preds = torch.argmax(model(x), dim=-1).type(torch.int64)
                            acc_counter[0] += torch.sum(preds == y).item()
                            acc_counter[1] += y.size(0)
                            
                        except StopIteration:
                            break
                
                eval_losses.append(eval_loss / eval_step)
                eval_accs.append(acc_counter[0] / acc_counter[1])
                acc_counter = [0, 0]
                
                pbar.write(f'Ep {epoch+1} (Step {global_step:06d}): '
                    f'Train loss {train_losses[-1]:.3f}, Val loss {eval_losses[-1]:.3f}, '
                    f'Train accuracy {train_accs[-1]:.3f}, Eval accuracy {eval_accs[-1]:.3f}')
                model.train()
                pbar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
                
    return train_losses, eval_losses, train_accs, eval_accs
    