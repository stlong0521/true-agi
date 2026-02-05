import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import SingleHeadAttention, MultiHeadAttention, FeedForward

def test_single_head_attention():
    d_model = 8
    d_head = 4
    batch_size = 2
    seq_len = 5
    
    single_head_attention = SingleHeadAttention(d_model, d_head)
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    attention_output = single_head_attention(x)
    
    print(f"SingleHead - Input shape: {x.shape}")
    print(f"SingleHead - Output shape: {attention_output.shape}")
    
    assert attention_output.shape == (batch_size, seq_len, d_head), f"Expected shape {(batch_size, seq_len, d_head)}, but got {attention_output.shape}"
    print("SingleHead Shape verification passed!")


def test_multi_head_attention():
    d_model = 8
    n_heads = 2
    batch_size = 2
    seq_len = 5
    
    multi_head_attention = MultiHeadAttention(d_model, n_heads)
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    attention_output = multi_head_attention(x)
    
    print(f"MultiHead - Input shape: {x.shape}")
    print(f"MultiHead - Output shape: {attention_output.shape}")
    
    assert attention_output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, but got {attention_output.shape}"
    print("MultiHead Shape verification passed!")


def test_feed_forward_shape():
    d_model = 8
    d_ff = 32
    batch_size = 2
    seq_len = 5
    
    ff = FeedForward(d_model, d_ff)
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = ff(x)
    
    print(f"FeedForward - Input shape: {x.shape}")
    print(f"FeedForward - Output shape: {output.shape}")
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, but got {output.shape}"
    print("FeedForward Shape verification passed!")


def test_feed_forward_logic():
    d_model = 8
    d_ff = 16
    ff = FeedForward(d_model, d_ff)
    
    x = torch.randn(1, 1, d_model)
    
    # Manual calculation
    with torch.no_grad():
        expected = ff.linear2(torch.relu(ff.linear1(x)))
        actual = ff(x)
    
    assert torch.allclose(actual, expected), "Forward pass logic mismatch"
    print("FeedForward Logic verification passed!")


if __name__ == "__main__":
    print("Running SingleHeadAttention tests...")
    test_single_head_attention()

    print("Running MultiHeadAttention tests...")
    test_multi_head_attention()

    print("Running FeedForward tests...")
    test_feed_forward_shape()
    test_feed_forward_logic()
    
    print("All tests passed!")
