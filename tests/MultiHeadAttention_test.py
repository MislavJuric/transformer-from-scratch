"""
    A simple test to check if everything is allright with the Multi-Head Attention implementation.
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import torch

from layers.MultiHeadAttention import MultiHeadAttention

d_model = 512
h = 8
d_k = 64
d_v = 64

multi_head_attention_layer = MultiHeadAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)

number_of_samples = 100

embeddings = torch.rand(number_of_samples, d_model)

print("multi_head_attention_layer.forward(embeddings):")
print(multi_head_attention_layer.forward(embeddings))
print("print(multi_head_attention_layer.forward(embeddings)).shape:")
print(multi_head_attention_layer.forward(embeddings).shape)

multi_head_attention_layer = MultiHeadAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v, masking=True)

print("multi_head_attention_layer.forward(embeddings):")
print(multi_head_attention_layer.forward(embeddings))
print("print(multi_head_attention_layer.forward(embeddings)).shape:")
print(multi_head_attention_layer.forward(embeddings).shape)
