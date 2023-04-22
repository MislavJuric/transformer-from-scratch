"""
    Note:   This implementation was written after reading only the original paper (Attention Is All You Need) and no other additional material.
            The main mistake here was in the signature of the forward method; it should accept an embedding, not Queries, Keys and Values matrices.
            After reading The Illustrated Transformer I re-implemented this layer.

    Scaled Dot-Product Attention implementation (from Attention Is All You Need, section 3.2.1)

    d_k ... dimension of queries and keys
    d_v ... dimension of values
    Q ... queries matrix (of dimensions number_of_queries x d_k)
    K ... keys matrix (of dimensions number_of_keys x d_k)
    V ... values matrix (of dimensions number_of_values x d_v)
"""

import torch
import numpy as np
import math

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k, d_v):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v

    def forward(self, Q, K, V):
        # asserts
        d_k = Q.shape[1]
        assert self.d_k == d_k, "d_k specified in the constructor must match the second dimension (number of columns) of the Q matrix"
        d_k = K.shape[1]
        assert self.d_k == d_k, "d_k specified in the constructor must match the second dimension (number of columns) of the K matrix"
        d_v = V.shape[1]
        assert self.d_v == d_v, "d_v specified in the constructor must match the second dimension (number of columns) of the V matrix"

        # type casting (if needed)
        if isinstance(Q, np.ndarray):
            Q = torch.from_numpy(Q)
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        if isinstance(V, np.ndarray):
            V = torch.from_numpy(V)

        softmax_fn = torch.nn.Softmax(dim=1)

        return torch.matmul(softmax_fn(torch.div(torch.matmul(Q, torch.transpose(K, 0, 1)), math.sqrt(d_k))), V)
