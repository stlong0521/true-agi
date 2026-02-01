import numpy as np


class SingleHeadAttention(object):

    def __init__(self, d_model, d_head):
        """
        Initialize the SingleHeadAttention.
        :param d_model: The dimension of the model (= embedding dim).
        :param d_head: The dimension of this single head (= d_model / n_heads = d_k in the orginal transformer paper).
        """
        self.d_head = d_head
        self.W_q = np.random.randn(d_model, d_head)
        self.W_k = np.random.randn(d_model, d_head)
        self.W_v = np.random.randn(d_model, d_head)

    def forward(self, x):
        """
        Compute the scaled dot-product attention.
        :param x: The input (batch_size x seq_len x d_model).
        :return: Attention output.
        """
        Q = x @ self.W_q  # batch_size x seq_len x d_head
        K = x @ self.W_k  # batch_size x seq_len x d_head
        V = x @ self.W_v  # batch_size x seq_len x d_head

        # Scaled dot-product attention
        attention_scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_head)  # batch_size x seq_len x seq_len

        # In a decoder model, apply mask to prevent attending to future tokens
        seq_len = x.shape[1]
        mask = np.tril(np.ones(seq_len, seq_len))
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax; np.max and np.sum are applied along the last axis (i.e. along each row of the seq_len x seq_len submatrix)
        attention_scores = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        attention_scores /= np.sum(attention_scores, axis=-1, keepdims=True)

        attention_output = attention_scores @ V  # batch_size x seq_len x d_head
        return attention_output


class Transformer(object):

    def __init__(self):
        pass

