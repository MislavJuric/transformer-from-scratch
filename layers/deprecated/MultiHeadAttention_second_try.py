"""
    Note:   I deprecated this code because I didn't need the is_matrix() check

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

    # TODO: maybe this function should have a better name since it returns True for all 2D shapes and beyond
    def is_matrix(self, arg):
        shape_of_arg = arg.shape
        shape_of_arg = list(shape_of_arg)
        if (len(shape_of_arg) > 1):
            return True
        else:
            return False

    def forward(self, embeddings):
        # TODO: assert if embeddings.shape[1] == self.d_model

        results = []

        for attention_layer_index in range(0, self.h):
            result_from_one_attention_layer = self.scaled_dot_product_attention_layers[attention_layer_index](embeddings)
            # debug prints
            """
            print("result_from_one_attention_layer.shape: (MultiHeadAttention)")
            print(result_from_one_attention_layer.shape)
            """
            results.append(result_from_one_attention_layer)

        if (self.is_matrix(results[0])):
            concat_result = torch.cat(results, dim=1)
        else:
            concat_result = torch.cat(results, dim=0)

        # debug prints
        """
        print("concat_result.shape: (MultiHeadAttention)")
        print(concat_result.shape)

        print("self.final_linear_layer(concat_result).shape: (MultiHeadAttention)")
        print(self.final_linear_layer(concat_result).shape)
        """
        return self.final_linear_layer(concat_result)
