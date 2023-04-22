"""
    A simple test to check if everything is allright with the Scaled Dot-Product Attention for the Decoder implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import numpy as np

from layers.ScaledDotProductAttentionDecoder import ScaledDotProductAttentionDecoder

if __name__ == "__main__":
    d_model = 512
    d_k = 64
    d_v = 64

    ScaledDotProductAttentionDecoderLayer = ScaledDotProductAttentionDecoder(d_model=d_model, d_k=d_k, d_v=d_v, masking=False)

    number_of_samples = 100

    embeddings = np.random.rand(number_of_samples, d_model)
    K = np.random.rand(number_of_samples, d_k)
    V = np.random.rand(number_of_samples, d_v)

    print("ScaledDotProductAttentionDecoderLayer.forward(embeddings, K, V):")
    print(ScaledDotProductAttentionDecoderLayer.forward(embeddings, K, V))
    print("ScaledDotProductAttentionDecoderLayer.forward(embeddings, K, V).shape:")
    print(ScaledDotProductAttentionDecoderLayer.forward(embeddings, K, V).shape)

    ScaledDotProductAttentionDecoderLayer = ScaledDotProductAttentionDecoder(d_model=d_model, d_k=d_k, d_v=d_v, masking=True)

    print("ScaledDotProductAttentionDecoderLayer.forward(embeddings, K, V):")
    print(ScaledDotProductAttentionDecoderLayer.forward(embeddings, K, V))
    print("ScaledDotProductAttentionDecoderLayer.forward(embeddings, K, V).shape:")
    print(ScaledDotProductAttentionDecoderLayer.forward(embeddings, K, V).shape)
