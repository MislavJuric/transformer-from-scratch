"""
    Note:   This was my first attempt at implementing Multi-Head Attention, after I had only read the Attention Is All You Need paper and no additional material.
            I misinterpreted what Multi-Head attention did; I thought it took a 512-dimensional embedding and then took 64 dimensions and threw it into one
            Scaled Dot-Product Attention, then took another 64 dimensions and threw that into another Scaled Dot-Product Attention, but that wasn't the case,
            as I learned after reading The Illustrated Transformer.

    Multi-Head Attention implementation (from Attention Is All You Need, section 3.2.2)

    h ... number of parallel attention layers
    d_k ... dimension of queries and keys
    d_v ... dimension of values
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, ".")

import torch
import numpy as np

from layers.ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d_k, d_v):
        super().__init__()

        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.number_of_features_per_linear_layer_for_queries_and_keys = self.d_k // h
        self.number_of_features_per_linear_layer_for_values = self.d_v // h
        self.d_model = self.h * self.number_of_features_per_linear_layer_for_queries_and_keys # TODO: see if this is right

        self.queries_layers = []
        self.keys_layers = []
        self.values_layers = []
        self.scaled_dot_product_attention_layers = []

        for i in range(0, self.h):
            new_layer_queries = torch.nn.Linear(in_features=self.number_of_features_per_linear_layer_for_queries_and_keys, out_features=self.number_of_features_per_linear_layer_for_queries_and_keys)
            self.queries_layers.append(new_layer_queries)
            new_layer_keys = torch.nn.Linear(in_features=self.number_of_features_per_linear_layer_for_queries_and_keys, out_features=self.number_of_features_per_linear_layer_for_queries_and_keys)
            self.keys_layers.append(new_layer_keys)
            new_layer_values = torch.nn.Linear(in_features=self.number_of_features_per_linear_layer_for_values, out_features=self.number_of_features_per_linear_layer_for_values)
            self.values_layers.append(new_layer_values)
            new_layer_attention = ScaledDotProductAttention(d_k=self.number_of_features_per_linear_layer_for_queries_and_keys, d_v=self.number_of_features_per_linear_layer_for_values)
            self.scaled_dot_product_attention_layers.append(new_layer_attention)

        self.final_linear_layer = torch.nn.Linear(in_features=self.h*self.number_of_features_per_linear_layer_for_values, out_features=self.d_model)


    def forward(self, Q, K, V):
        # type casting (if needed)
        if isinstance(Q, np.ndarray):
            Q = torch.from_numpy(Q)
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        if isinstance(V, np.ndarray):
            V = torch.from_numpy(V)

        results = []

        for i in range(0, self.h):
            begin_index_queries_and_keys = i * self.number_of_features_per_linear_layer_for_queries_and_keys
            end_index_queries_and_keys = begin_index_queries_and_keys + self.number_of_features_per_linear_layer_for_queries_and_keys

            begin_index_values = i * self.number_of_features_per_linear_layer_for_values
            end_index_values = begin_index_values + self.number_of_features_per_linear_layer_for_values

            current_Q_features_subset = Q[:, begin_index_queries_and_keys:end_index_queries_and_keys]
            current_K_features_subset = K[:, begin_index_queries_and_keys:end_index_queries_and_keys]
            current_V_features_subset = V[:, begin_index_values:end_index_values]

            # needed so that I avoid RuntimeError: mat1 and mat2 must have the same dtype
            current_Q_features_subset_float = current_Q_features_subset.float()
            current_K_features_subset_float = current_K_features_subset.float()
            current_V_features_subset_float = current_V_features_subset.float()

            results.append(self.scaled_dot_product_attention_layers[i](self.queries_layers[i](current_Q_features_subset_float), self.keys_layers[i](current_K_features_subset_float), self.values_layers[i](current_V_features_subset_float)))

        concat_result = torch.cat(results, dim=1)

        return self.final_linear_layer(concat_result)
