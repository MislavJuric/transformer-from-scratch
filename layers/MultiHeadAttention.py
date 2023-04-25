"""
    Multi-Head Attention implementation (from Attention Is All You Need, section 3.2.2)

    d_model ... embedding dimensions
    h ... number of parallel attention layers
    d_k ... dimension of queries and keys
    d_v ... dimension of values
    masking ... a boolean indicating if look-ahead masking is to be applied (implemented based on https://medium.com/mlearning-ai/how-do-self-attention-masks-work-72ed9382510f)
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, ".")

import torch
import numpy as np

from layers.ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model=512, h=8, d_k=64, d_v=64, masking=False):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.masking = masking

        self.scaled_dot_product_attention_layers = torch.nn.ModuleList()

        for i in range(0, self.h):
            new_layer_attention = ScaledDotProductAttention(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, masking=self.masking)
            self.scaled_dot_product_attention_layers.append(new_layer_attention)

        self.final_linear_layer = torch.nn.Linear(in_features=self.h*self.d_v, out_features=self.d_model)

    def forward(self, embeddings):
        # TODO: assert if embeddings.shape[1] == self.d_model

        results = []
        for attention_layer_index in range(0, self.h):
            result_from_one_attention_layer = self.scaled_dot_product_attention_layers[attention_layer_index](embeddings)
            results.append(result_from_one_attention_layer)

        concat_result = torch.cat(results, dim=1)

        return self.final_linear_layer(concat_result)
