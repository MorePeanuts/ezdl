"""
This script downloads the gpt2 model from HuggingFace and converts it to PyTorch format used in this repo.

Reference:
- https://github.com/rasbt/LLMs-from-scratch/blob/0adb5b8c6573e337ce01c69d6553ba031c23e405/ch05/01_main-chapter-code/ch05.ipynb
"""

import argparse
import tempfile
import urllib
import torch
import tiktoken
import torch.nn as nn
from pathlib import Path
from mini_transformer.models.gpt2 import (
    GPT2Config,
    GPT2ModelForCausalLM,
    generate_text_simple,
    token_ids_to_text,
)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, 'd_out must be divisible by n_heads'

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # type: ignore

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias'],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


CONFIG = {
    '124m': {'emb_dim': 768, 'n_layers': 12, 'n_heads': 12},
    '355m': {'emb_dim': 1024, 'n_layers': 24, 'n_heads': 16},
    '774m': {'emb_dim': 1280, 'n_layers': 36, 'n_heads': 20},
    '1558m': {'emb_dim': 1600, 'n_layers': 48, 'n_heads': 25},
}
CONFIG = {
    key: {
        **config,
        'vocab_size': 50257,
        'context_length': 1024,
        'qkv_bias': True,
        'drop_rate': 0.1,
    }
    for key, config in CONFIG.items()
}
TO_CONFIG = {
    '124m': GPT2Config(qkv_bias=True),
    '355m': GPT2Config(qkv_bias=True, embd_dim=1024, n_layer=24, n_head=16),
    '774m': GPT2Config(qkv_bias=True, embd_dim=1280, n_layer=36, n_head=20),
    '1558m': GPT2Config(qkv_bias=True, embd_dim=1600, n_layer=48, n_head=25),
}


def download_gpt2_and_load():
    print(f'downloading gpt2_{args.gpt2_version}')

    file_name_map = {
        '124m': 'gpt2-small-124M.pth',
        '355m': 'gpt2-medium-355M.pth',
        '774m': 'gpt2-large-774M.pth',
        '1558m': 'gpt2-xl-1558M.pth',
    }
    file_name = file_name_map[args.gpt2_version]
    url = f'https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}'

    with tempfile.NamedTemporaryFile(suffix='.pth', mode='wb') as tmp_file:
        with urllib.request.urlopen(url) as response:  # type: ignore
            tmp_file.write(response.read())

        model = GPTModel(CONFIG[args.gpt2_version])
        model.load_state_dict(torch.load(tmp_file.name))

    return model


def convert_ckpt(model: GPTModel) -> GPT2ModelForCausalLM:
    gpt2_model = GPT2ModelForCausalLM(TO_CONFIG[args.gpt2_version])

    # Get source and destination state dicts
    src_state_dict = model.state_dict()
    dst_state_dict = gpt2_model.state_dict()

    # Mapping from source to destination parameters
    param_mapping = {}

    # Embedding layers
    param_mapping['model.tok_embd.weight'] = 'tok_emb.weight'
    param_mapping['model.pos_embd.weight'] = 'pos_emb.weight'

    # Output head
    param_mapping['lm_head.weight'] = 'out_head.weight'

    # Final layer norm
    param_mapping['model.final_norm.scale'] = 'final_norm.scale'
    param_mapping['model.final_norm.shift'] = 'final_norm.shift'

    # Transformer blocks
    n_layers = CONFIG[args.gpt2_version]['n_layers']
    for i in range(n_layers):
        # Attention layers
        param_mapping[f'model.trf_blocks.{i}.attn.W_q.weight'] = (
            f'trf_blocks.{i}.att.W_query.weight'
        )
        param_mapping[f'model.trf_blocks.{i}.attn.W_q.bias'] = f'trf_blocks.{i}.att.W_query.bias'
        param_mapping[f'model.trf_blocks.{i}.attn.W_k.weight'] = f'trf_blocks.{i}.att.W_key.weight'
        param_mapping[f'model.trf_blocks.{i}.attn.W_k.bias'] = f'trf_blocks.{i}.att.W_key.bias'
        param_mapping[f'model.trf_blocks.{i}.attn.W_v.weight'] = (
            f'trf_blocks.{i}.att.W_value.weight'
        )
        param_mapping[f'model.trf_blocks.{i}.attn.W_v.bias'] = f'trf_blocks.{i}.att.W_value.bias'
        param_mapping[f'model.trf_blocks.{i}.attn.out_proj.weight'] = (
            f'trf_blocks.{i}.att.out_proj.weight'
        )
        param_mapping[f'model.trf_blocks.{i}.attn.out_proj.bias'] = (
            f'trf_blocks.{i}.att.out_proj.bias'
        )
        param_mapping[f'model.trf_blocks.{i}.attn.mask'] = f'trf_blocks.{i}.att.mask'

        # Feed forward layers
        param_mapping[f'model.trf_blocks.{i}.ffn.layers.0.weight'] = (
            f'trf_blocks.{i}.ff.layers.0.weight'
        )
        param_mapping[f'model.trf_blocks.{i}.ffn.layers.0.bias'] = (
            f'trf_blocks.{i}.ff.layers.0.bias'
        )
        param_mapping[f'model.trf_blocks.{i}.ffn.layers.2.weight'] = (
            f'trf_blocks.{i}.ff.layers.2.weight'
        )
        param_mapping[f'model.trf_blocks.{i}.ffn.layers.2.bias'] = (
            f'trf_blocks.{i}.ff.layers.2.bias'
        )

        # Layer norm layers
        param_mapping[f'model.trf_blocks.{i}.norm1.scale'] = f'trf_blocks.{i}.norm1.scale'
        param_mapping[f'model.trf_blocks.{i}.norm1.shift'] = f'trf_blocks.{i}.norm1.shift'
        param_mapping[f'model.trf_blocks.{i}.norm2.scale'] = f'trf_blocks.{i}.norm2.scale'
        param_mapping[f'model.trf_blocks.{i}.norm2.shift'] = f'trf_blocks.{i}.norm2.shift'

    # Copy parameters from source to destination
    for dst_key, src_key in param_mapping.items():
        if src_key in src_state_dict and dst_key in dst_state_dict:
            dst_state_dict[dst_key].copy_(src_state_dict[src_key])
        else:
            print(f'Warning: {src_key} -> {dst_key} mapping failed')

    return gpt2_model


def main():
    model_directory = Path(__file__).parents[1] / f'models/gpt2_{args.gpt2_version}'
    if not args.no_download:
        model = download_gpt2_and_load()
    else:
        model = GPTModel(CONFIG[args.gpt2_version])
        state_dict = torch.load(model_directory / 'model.pt')
        model.load_state_dict(state_dict)
        if args.print_state_dict:
            for key in state_dict.keys():
                print(key)

    if args.no_convert:
        print('Skipping conversion')
        return

    gpt2_model = convert_ckpt(model)
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor(tokenizer.encode('Hello, I am')).unsqueeze(0)
    context_length = gpt2_model.config.context_length  # type: ignore
    output = generate_text_simple(gpt2_model, input_ids, 20, context_length)
    print(token_ids_to_text(output, tokenizer))
    gpt2_model.save_pretrained(model_directory, safe_serialization=args.safetensors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('gpt2_version', type=str, choices=['124m', '355m', '774m', '1558m'])
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Convert local model instead of downloading. Rename the ckpt to `model.pth` first and then run this script to convert it.',
    )
    parser.add_argument(
        '--no-convert',
        action='store_true',
        help='Only download the model without converting it.',
    )
    parser.add_argument(
        '--safetensors', action='store_true', help='Convert into safetensors format.'
    )
    parser.add_argument(
        '--print-state-dict',
        action='store_true',
        help='Print the state dict of the raw model.',
    )

    args = parser.parse_args()
    main()
