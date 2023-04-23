"""
    A simple test to check if everything is allright with the Scaled Dot-Product Attention for the Decoder implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import torch

from layers.ScaledDotProductAttentionDecoder import ScaledDotProductAttentionDecoder

d_model = 512
d_k = 64
d_v = 64

scaled_dot_product_attention_decoder_layer = ScaledDotProductAttentionDecoder(d_model=d_model, d_k=d_k, d_v=d_v, masking=False)

number_of_samples = 100

embeddings = torch.rand(number_of_samples, d_model)
K = torch.rand(number_of_samples, d_k)
V = torch.rand(number_of_samples, d_v)

print("scaled_dot_product_attention_decoder_layer.forward(embeddings, K, V):")
print(scaled_dot_product_attention_decoder_layer.forward(embeddings, K, V))
print("scaled_dot_product_attention_decoder_layer.forward(embeddings, K, V).shape:")
print(scaled_dot_product_attention_decoder_layer.forward(embeddings, K, V).shape)

scaled_dot_product_attention_decoder_layer = ScaledDotProductAttentionDecoder(d_model=d_model, d_k=d_k, d_v=d_v, masking=True)

print("scaled_dot_product_attention_decoder_layer.forward(embeddings, K, V):")
print(scaled_dot_product_attention_decoder_layer.forward(embeddings, K, V))
print("scaled_dot_product_attention_decoder_layer.forward(embeddings, K, V).shape:")
print(scaled_dot_product_attention_decoder_layer.forward(embeddings, K, V).shape)
