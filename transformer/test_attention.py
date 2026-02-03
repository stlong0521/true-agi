import torch
from transformer import SingleHeadAttention, MultiHeadAttention

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

if __name__ == "__main__":
    print("Running SingleHeadAttention test...")
    test_single_head_attention()

    print("Running MultiHeadAttention test...")
    test_multi_head_attention()
