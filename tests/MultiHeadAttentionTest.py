"""
    A simple test to check if everything is allright with the Multi-Head Attention implementation.
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import numpy as np

from layers.MultiHeadAttention import MultiHeadAttention

if __name__ == "__main__":
    d_model = 512
    h = 8
    d_k = 64
    d_v = 64

    MultiHeadAttentionLayer = MultiHeadAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)

    number_of_samples = 100

    embeddings = np.random.rand(number_of_samples, d_model)

    print("MultiHeadAttentionLayer.forward(embeddings):")
    print(MultiHeadAttentionLayer.forward(embeddings))
    print("print(MultiHeadAttentionLayer.forward(embeddings)).shape:")
    print(MultiHeadAttentionLayer.forward(embeddings).shape)

    MultiHeadAttentionLayer = MultiHeadAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v, masking=True)

    print("MultiHeadAttentionLayer.forward(embeddings):")
    print(MultiHeadAttentionLayer.forward(embeddings))
    print("print(MultiHeadAttentionLayer.forward(embeddings)).shape:")
    print(MultiHeadAttentionLayer.forward(embeddings).shape)
