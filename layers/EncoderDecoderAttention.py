"""
    Encoder-Decoder Attention implementation (from The Illustrated Transformer)
    Encoder-Decoder Attention is supposed to take in Keys and Values matrices from the Encoder, but it has its own Queries matrix.

    d_model ... embedding dimensions
    h ... number of parallel attention layers
    d_k ... dimension of queries and keys
    d_v ... dimension of values
    masking ... a boolean indicating if look-ahead masking is to be applied (implemented based on https://medium.com/mlearning-ai/how-do-self-attention-masks-work-72ed9382510f)

    K ... keys matrix (of dimensions number_of_samples x d_k)
    V ... values matrix (of dimensions number_of_samples x d_v)
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, ".")

import torch
import numpy as np

from layers.ScaledDotProductAttentionDecoder import ScaledDotProductAttentionDecoder

class EncoderDecoderAttention(torch.nn.Module):
    def __init__(self, d_model=512, h=8, d_k=64, d_v=64, masking=False):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.masking = masking

        self.scaled_dot_product_attention_layers = torch.nn.ModuleList()

        for i in range(0, self.h):
            new_layer_attention = ScaledDotProductAttentionDecoder(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, masking=self.masking)
            self.scaled_dot_product_attention_layers.append(new_layer_attention)

        self.final_linear_layer = torch.nn.Linear(in_features=self.h*self.d_v, out_features=self.d_model)

    # TODO: maybe this function should have a better name since it returns True for all 2D shapes and beyond
    def is_matrix(self, arg):
        shape_of_arg = arg.shape
        shape_of_arg = list(shape_of_arg)
        if (len(shape_of_arg) > 1):
            return True
        else:
            return False

    def forward(self, embeddings, K, V):
        # TODO: assert if embeddings.shape[1] == self.d_model
        # TODO: an assert here that V.shape[1] == self.d_v

        results = []
        for attention_layer_index in range(0, self.h):
            result_from_one_attention_layer = self.scaled_dot_product_attention_layers[attention_layer_index](embeddings, K, V)
            results.append(result_from_one_attention_layer)

        if (self.is_matrix(results[0])):
            concat_result = torch.cat(results, dim=1)
        else:
            concat_result = torch.cat(results, dim=0)

        return self.final_linear_layer(concat_result)
