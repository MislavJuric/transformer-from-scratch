"""
    Encoder block implementation (from Attention Is All You Need, section 3.1)

    d_model ... dimension of the embedding
    h ... number of parallel attention layers
    d_k ... dimension of queries and keys
    d_v ... dimension of values
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

    def __init__(self, d_model=512, h=8, d_k=64, d_v=64, d_ff=2048):
        super().__init__()

        self.d_model = d_model

        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

        self.multi_head_attention_layer = MultiHeadAttention(d_model=self.d_model, h=self.h, d_k=self.d_k, d_v=self.d_v, masking=False)
        self.add_and_norm_layer = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.feed_forward_layer = FeedForward(d_model=self.d_model, d_ff=self.d_ff)

    def forward(self, embeddings):
        first_subblock_output = self.add_and_norm_layer(embeddings + self.multi_head_attention_layer(embeddings))
        final_result = self.add_and_norm_layer(first_subblock_output + self.feed_forward_layer(first_subblock_output))
        return final_result
