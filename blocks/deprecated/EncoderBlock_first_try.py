"""
    Note:   This was my first try at implementing the Encoder block from Attention Is All You Need paper. My mistake here,
            as I became aware of it after reading The Illustrated Transformer, was that I thought there were only three
            weight matrices that transformed embeddings to queries, keys and values, while in fact, every Scaled Dot-Product
            Attention layer has its own three weight matrices that transformed the embeddings into queries, keys and values.
            This code relies on the "_first_try.py" versions of Scaled Dot-Product Attention and Multi-Head Attention and
            as such isn't updated.

    Encoder block implementation (from Attention Is All You Need, section 3.1)

    embedding_dim ... dimension of the embedding
    d_k ... dimension of queries and keys
    d_v ... dimension of values
    h ... number of parallel attention layers
    d_ff ... dimension of inner feedforward network layer
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, ".")

import torch
import numpy as np

from layers.MultiHeadAttention import MultiHeadAttention
from layers.FeedForward import FeedForward

class EncoderBlock(torch.nn.Module):

    # static variable for FeedForwardLayer, since The Illustrated Transformer says that it is shared between Encoder blocks
    d_model_for_static_FeedForward_layer = 512
    d_ff_for_static_FeedForward_layer = 2048
    FeedForwardLayer = FeedForward(d_model=d_model_for_static_FeedForward_layer, d_ff=d_ff_for_static_FeedForward_layer)

    def __init__(self, embedding_dim=512, d_k=512, d_v=512, h=8, d_ff=2048):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.h = h

        self.d_k = d_k
        self.d_v = d_v

        self.EmbeddingToQueryLayer = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.d_k)
        self.EmbeddingToKeyLayer = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.d_k)
        self.EmbeddingToValueLayer = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.d_v)

        self.MultiHeadAttentionLayer = MultiHeadAttention(h=self.h, d_k=self.d_k, d_v=self.d_v)
        self.AddAndNormLayer = torch.nn.LayerNorm(normalized_shape=self.embedding_dim)

    def forward(self, embeddings):
        # type casting (if needed)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
            embeddings = embeddings.float()

        Q = self.EmbeddingToQueryLayer(embeddings)
        K = self.EmbeddingToKeyLayer(embeddings)
        V = self.EmbeddingToValueLayer(embeddings)

        first_sublock_output = self.AddAndNormLayer(embeddings + self.MultiHeadAttentionLayer(Q=Q, K=K, V=V))
        final_result = self.AddAndNormLayer(first_sublock_output + EncoderBlock.FeedForwardLayer(first_sublock_output))

        return final_result
