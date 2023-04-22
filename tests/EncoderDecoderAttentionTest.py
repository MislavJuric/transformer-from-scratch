"""
    A simple test to check if everything is allright with the Encoder-Decoder Attention implementation.
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import numpy as np

from layers.EncoderDecoderAttention import EncoderDecoderAttention

if __name__ == "__main__":
    d_model = 512
    h = 8
    d_k = 64
    d_v = 64

    EncoderDecoderAttentionLayer = EncoderDecoderAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)

    number_of_samples = 100

    embeddings = np.random.rand(number_of_samples, d_model)
    K = np.random.rand(number_of_samples, d_k)
    V = np.random.rand(number_of_samples, d_v)

    print("EncoderDecoderAttentionLayer.forward(embeddings, K, V):")
    print(EncoderDecoderAttentionLayer.forward(embeddings, K, V))
    print("print(EncoderDecoderAttentionLayer.forward(embeddings, K, V)).shape:")
    print(EncoderDecoderAttentionLayer.forward(embeddings, K, V).shape)

    EncoderDecoderAttentionLayer = EncoderDecoderAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v, masking=True)

    print("EncoderDecoderAttentionLayer.forward(embeddings, K, V):")
    print(EncoderDecoderAttentionLayer.forward(embeddings, K, V))
    print("print(EncoderDecoderAttentionLayer.forward(embeddings, K, V)).shape:")
    print(EncoderDecoderAttentionLayer.forward(embeddings, K, V).shape)
