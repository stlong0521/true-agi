import numpy as np
from transformer import SingleHeadAttention

def test_single_head_attention():
    d_model = 8
    d_head = 4
    batch_size = 2
    seq_len = 5
    
    single_head_attention = SingleHeadAttention(d_model, d_head)
    
    # Dummy input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    attention_output = single_head_attention.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {attention_output.shape}")
    
    assert attention_output.shape == (batch_size, seq_len, d_head), f"Expected shape {(batch_size, seq_len, d_head)}, but got {attention_output.shape}"
    print("Shape verification passed!")

if __name__ == "__main__":
    test_single_head_attention()
