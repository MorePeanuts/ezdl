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
    train_bar = tqdm(total=total_steps)
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        train_bar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
        
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            train_bar.update(1)
            
            if global_step % eval_freq == 0:
                model.eval()
                train_bar.set_description('Evaluating...')
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
                train_bar.write(f'Ep {epoch+1} (Step {global_step:06d}): '
                    f'Train loss {train_loss:.3f}, Val loss {eval_loss:.3f}')
                train_bar.set_description(f'Training Epoch {epoch+1}/{num_epochs}')
            
        model.eval()
        context_length = model.model.pos_embd.weight.shape[0]
        encoded_text = text_to_token_ids(
            start_context, tokenizer
        ).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(model, encoded_text, 50, context_length)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        train_bar.write(decoded_text.replace('\n', ''))
    
    return train_losses, eval_losses, track_tokens_seen
                