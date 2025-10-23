import pytest
import torch
from ezdl.scratch.self_attention import (
    SelfAttention,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    MultiHeadAttentionCombinedQKV,
    MultiHeadAttentionEinsum,
    MultiHeadAttentionScaledDotProduct,
    MultiHeadAttentionPytorch,
    FlashAttentionScratch,
)


@pytest.fixture
def inputs():
    return torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your (x^1)
            [0.55, 0.87, 0.66],  # journey (x^2)
            [0.57, 0.85, 0.64],  # starts (x^3)
            [0.22, 0.58, 0.33],  # with (x^4)
            [0.77, 0.25, 0.10],  # one (x^5)
            [0.05, 0.80, 0.55],  # step (x^6)
        ]
    )


def test_self_attention(inputs): 
    d_in, d_out = inputs.shape[1], 2
    sa = SelfAttention(d_in, d_out)
    outputs = sa(inputs)
    assert tuple(outputs.shape) == (inputs.shape[0], d_out)


def test_causal_attention(inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    sequence_length = batch.shape[1]
    d_in, d_out = inputs.shape[1], 2
    
    for context_length in [
        sequence_length,
        sequence_length * 2,
        # sequence_length // 2,
    ]:
        ca = CausalAttention(d_in, d_out, context_length, 0.0)
        outputs = ca(batch)
        assert tuple(outputs.shape) == (2, sequence_length, d_out), f'Failed when context_length={context_length} and sequence_length={sequence_length}'

    
@pytest.mark.parametrize('MHA', [
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    MultiHeadAttentionCombinedQKV,
    MultiHeadAttentionEinsum,
])
def test_multi_head_attention(MHA, inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    sequence_length = batch.shape[1]
    d_in, d_out = inputs.shape[1], 4
    
    for context_length in [
        sequence_length,
        sequence_length * 2,
    ]:
        mha = MHA(d_in, d_out, context_length, 0.0, num_heads=2)
        outputs = mha(batch)
        assert tuple(outputs.shape) == (2, sequence_length, d_out), f'Failed when context_length={context_length} and sequence_length={sequence_length}'
        
        
@pytest.mark.parametrize('MHA', [
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    MultiHeadAttentionCombinedQKV,
    MultiHeadAttentionEinsum,
])
@pytest.mark.xfail(reason="The input's sequence length cannot exceed the context length.")
def test_multi_head_attention_error_case(MHA, inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    sequence_length = batch.shape[1]
    d_in, d_out = inputs.shape[1], 4
    context_length = sequence_length // 2
    mha = MHA(d_in, d_out, context_length, 0.0, num_heads=2)
    outputs = mha(batch)
    assert tuple(outputs.shape) == (2, sequence_length, d_out), f'Failed when context_length={context_length} and sequence_length={sequence_length}'
    

def test_multi_head_attention_scaled_dot_product(inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    sequence_length = batch.shape[1]
    d_in, d_out = inputs.shape[1], 4
    
    for context_length in [
        sequence_length,
        sequence_length * 2,
        sequence_length // 2,
    ]:
        mha = MultiHeadAttentionScaledDotProduct(d_in, d_out, 0.0, num_heads=2)
        outputs = mha(batch)
        assert tuple(outputs.shape) == (2, sequence_length, d_out), f'Failed when context_length={context_length} and sequence_length={sequence_length}'
    

def test_multi_head_attention_pytorch(inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    batch = torch.cat((batch, batch), dim=-1)
    sequence_length = batch.shape[1]
    embed_dim = batch.shape[-1]
    
    for context_length in [
        sequence_length,
        sequence_length * 2,
        # sequence_length // 2,
    ]:
        mha = MultiHeadAttentionPytorch(embed_dim, context_length, 0.0, num_heads=2)
        outputs = mha(batch)
        assert tuple(outputs.shape) == (2, sequence_length, embed_dim), f'Failed when context_length={context_length} and sequence_length={sequence_length}'
    
    
@pytest.mark.xfail(reason="The input's sequence length cannot exceed the context length.")
def test_multi_head_attention_pytorch_error_case(inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    batch = torch.cat((batch, batch), dim=-1)
    sequence_length = batch.shape[1]
    embed_dim = batch.shape[-1]
    
    context_length = sequence_length // 2,
    mha = MultiHeadAttentionPytorch(embed_dim, context_length, 0.0, num_heads=2)
    with pytest.raises(RuntimeError):
        mha(batch)
    
    
def test_flash_attention_scratch():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    batch_size = 2
    seq_length = 128
    d_in = 256
    d_out = 512
    num_heads = 8
    context_length = 512
    dropout = 0.1
    block_size = 32

    # Create input tensor
    x = torch.randn(batch_size, seq_length, d_in)

    # Initialize FlashAttentionScratch
    flash_attn = FlashAttentionScratch(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=False,
        block_size=block_size,
    )

    # Initialize standard MultiHeadAttention for comparison
    standard_attn = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=False,
    )

    # Set both models to evaluation mode
    flash_attn.eval()
    standard_attn.eval()

    # Forward pass
    with torch.no_grad():
        flash_output = flash_attn(x)
        standard_output = standard_attn(x)

    # Check output shapes
    assert flash_output.shape == standard_output.shape, (
        f"Output shapes don't match: {flash_output.shape} vs {standard_output.shape}"
    )
    assert flash_output.shape == (batch_size, seq_length, d_out), (
        f"Expected output shape {(batch_size, seq_length, d_out)}, got {flash_output.shape}"
    )

    print("✓ FlashAttentionScratch implementation test passed!")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {flash_output.shape}")
    print(f"✓ Block size: {block_size}")
    print(f"✓ Number of heads: {num_heads}")
    print(f"✓ Head dimension: {d_out // num_heads}")

    # Test with different block sizes
    for test_block_size in [16, 64, 128]:
        flash_attn_test = FlashAttentionScratch(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=False,
            block_size=test_block_size,
        )
        flash_attn_test.eval()

        with torch.no_grad():
            test_output = flash_attn_test(x)

        assert test_output.shape == (batch_size, seq_length, d_out), (
            f"Block size {test_block_size} test failed"
        )
        print(f"✓ Block size {test_block_size} test passed")

    print("\n🎉 All FlashAttentionScratch tests completed successfully!")