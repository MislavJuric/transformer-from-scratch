"""
    A simple test to check if everything is allright with the Encoder-Decoder Attention implementation.
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import torch

from layers.EncoderDecoderAttention import EncoderDecoderAttention

d_model = 512
h = 8
d_k = 64
d_v = 64

encoder_decoder_attention_layer = EncoderDecoderAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)

number_of_samples = 100

embeddings = torch.rand(number_of_samples, d_model)
K = torch.rand(number_of_samples, d_k)
V = torch.rand(number_of_samples, d_v)

print("encoder_decoder_attention_layer.forward(embeddings, K, V):")
print(encoder_decoder_attention_layer.forward(embeddings, K, V))
print("print(encoder_decoder_attention_layer.forward(embeddings, K, V)).shape:")
print(encoder_decoder_attention_layer.forward(embeddings, K, V).shape)

encoder_decoder_attention_layer = EncoderDecoderAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v, masking=True)

print("encoder_decoder_attention_layer.forward(embeddings, K, V):")
print(encoder_decoder_attention_layer.forward(embeddings, K, V))
print("print(encoder_decoder_attention_layer.forward(embeddings, K, V)).shape:")
print(encoder_decoder_attention_layer.forward(embeddings, K, V).shape)
