import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):

    def __init__(self, d_model, d_head):
        """
        Initialize the SingleHeadAttention.
        :param d_model: The dimension of the model (= embedding dim).
        :param d_head: The dimension of this single head (= d_model / n_heads = d_k in the orginal transformer paper).
        """
        super().__init__()
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x):
        """
        Compute the scaled dot-product attention.
        :param x: The input (batch_size x seq_len x d_model).
        :return: Attention output.
        """
        Q = self.W_q(x)  # batch_size x seq_len x d_head
        K = self.W_k(x)  # batch_size x seq_len x d_head
        V = self.W_v(x)  # batch_size x seq_len x d_head

        # Scaled dot-product attention
        attention_scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)  # batch_size x seq_len x seq_len

        # In a decoder model, apply mask to prevent attending to future tokens
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax on the last dimension; F.softmax is essentially doing the following:
        # attention_scores = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        # attention_scores /= np.sum(attention_scores, axis=-1, keepdims=True)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = attention_weights @ V  # batch_size x seq_len x d_head
        return attention_output


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        """
        Initialize the MultiHeadAttention.
        :param d_model: The dimension of the model.
        :param n_heads: The number of attention heads.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Creating multiple SingleHeadAttention instances
        self.heads = nn.ModuleList([SingleHeadAttention(d_model, self.d_head) for _ in range(n_heads)])

        # Final projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        Compute the multi-head attention.
        :param x: The input (batch_size x seq_len x d_model).
        :return: Attention output.
        """
        # Collect outputs from each head
        head_outputs = [head(x) for head in self.heads]  # List of n_heads * (batch_size x seq_len x d_head)
        
        # Concatenate outputs along the last dimension
        attention_output = torch.cat(head_outputs, dim=-1)  # batch_size x seq_len x d_model

        # Final projection
        attention_output = self.W_o(attention_output)  # batch_size x seq_len x d_model

        return attention_output


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        pass
