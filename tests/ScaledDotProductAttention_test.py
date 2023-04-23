"""
    A simple test to check if everything is allright with the Scaled Dot-Product Attention implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import torch

from layers.ScaledDotProductAttention import ScaledDotProductAttention

d_model = 512
d_k = 64
d_v = 64

scaled_dot_product_attention_layer = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, masking=False)

number_of_samples = 100

embeddings = torch.rand(number_of_samples, d_model)

print("scaled_dot_product_attention_layer.forward(embeddings):")
print(scaled_dot_product_attention_layer.forward(embeddings))
print("scaled_dot_product_attention_layer.forward(embeddings).shape:")
print(scaled_dot_product_attention_layer.forward(embeddings).shape)

scaled_dot_product_attention_layer = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, masking=True)

print("scaled_dot_product_attention_layer.forward(embeddings):")
print(scaled_dot_product_attention_layer.forward(embeddings))
print("scaled_dot_product_attention_layer.forward(embeddings).shape:")
print(scaled_dot_product_attention_layer.forward(embeddings).shape)
