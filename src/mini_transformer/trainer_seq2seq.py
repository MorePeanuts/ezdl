import torch
from tqdm import tqdm
from deprecated import deprecated
from .trainer import Trainer


class Seq2SeqTrainer(Trainer):
    ...


@deprecated(
    version='1.0.0',
    reason='This function is a simple implementation of gpt2 trainer, and will be replaced by `Seq2SeqTrainer` class in the future.'
)
def train_gpt2_simple(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    from .models.gpt2 import (
        calc_loss_batch,
        calc_loss_dataloader,
        text_to_token_ids,
        token_ids_to_text,
        generate_text_simple,
    )
    print(f'Training GPT2 model on {device}...')
    
    train_losses, eval_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    total_steps = num_epochs * len(train_dataloader)
    pbar = tqdm(total=total_steps)
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        pbar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
        
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel() # max_length == stride only
            global_step += 1
            pbar.update(1)
            
            if global_step % eval_freq == 0 or global_step == total_steps:
                model.eval()
                pbar.set_description('Evaluating...')
                with torch.no_grad():
                    train_loss = calc_loss_dataloader(
                        train_dataloader,
                        model,
                        device,
                        num_batches=eval_iter
                    )
                    eval_loss = calc_loss_dataloader(
                        eval_dataloader,
                        model,
                        device,
                        num_batches=eval_iter
                    )
                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
                track_tokens_seen.append(tokens_seen)
                pbar.write(f'Ep {epoch+1} (Step {global_step:06d}): '
                    f'Train loss {train_loss:.3f}, Val loss {eval_loss:.3f}')
                model.train()
                pbar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
            
        model.eval()
        context_length = model.model.pos_embd.weight.shape[0]
        encoded_text = text_to_token_ids(
            start_context, tokenizer
        ).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(model, encoded_text, 50, context_length)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        pbar.write(decoded_text.replace('\n', ''))
    
    return train_losses, eval_losses, track_tokens_seen
                
                
@deprecated(
    version='1.0.0',
    reason='This function is a simple implementation of gpt2 trainer, and will be replaced by `Seq2SeqTrainer` class in the future.'
)
def train_gpt2_classifier_simple(
    model, 
    train_dataloader,
    eval_dataloader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter
):
    from .models.gpt2 import (
        calc_loss_batch,
        calc_loss_dataloader,
        calc_accuracy_dataloader
    )
    print(f'Training GPT2 classifier on {device}...')
    
    train_losses, eval_losses, train_accs, eval_accs = [], [], [], []
    samples_seen, global_step = 0, -1
    total_steps = num_epochs * len(train_dataloader)
    pbar = tqdm(total=total_steps)
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        pbar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
        
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device, True)
            loss.backward()
            optimizer.step()
            samples_seen += input_batch.shape[0]
            global_step += 1
            pbar.update(1)
            
            if global_step % eval_freq == 0 or global_step == total_steps:
                model.eval()
                pbar.set_description('Evaluating...')
                with torch.no_grad():
                    train_loss = calc_loss_dataloader(
                        train_dataloader,
                        model,
                        device,
                        eval_iter,
                        True
                    )
                    eval_loss = calc_loss_dataloader(
                        eval_dataloader,
                        model,
                        device,
                        eval_iter,
                        True
                    )
                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
                pbar.write(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {eval_loss:.3f}")
                model.train()
                pbar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
                
        model.eval()
        train_acc = calc_accuracy_dataloader(
            train_dataloader,
            model,
            device,
            eval_iter
        )
        eval_acc = calc_accuracy_dataloader(
            eval_dataloader,
            model,
            device,
            eval_iter
        )
        pbar.write(f"Training accuracy: {train_acc*100:.2f}% | "
            f"Validation accuracy: {eval_acc*100:.2f}%")
        train_accs.append(train_acc)
        eval_accs.append(eval_acc)
        
    return train_losses, eval_losses, train_accs, eval_accs, samples_seen
    