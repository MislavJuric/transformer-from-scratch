"""
    A simple test to check if everything is allright with the Scaled Dot-Product Attention implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import numpy as np

from layers.ScaledDotProductAttention import ScaledDotProductAttention

if __name__ == "__main__":
    d_model = 512
    d_k = 64
    d_v = 64

    ScaledDotProductAttentionLayer = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, masking=False)

    number_of_samples = 100

    embeddings = np.random.rand(number_of_samples, d_model)

    print("ScaledDotProductAttentionLayer.forward(embeddings):")
    print(ScaledDotProductAttentionLayer.forward(embeddings))
    print("ScaledDotProductAttentionLayer.forward(embeddings).shape:")
    print(ScaledDotProductAttentionLayer.forward(embeddings).shape)

    ScaledDotProductAttentionLayer = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, masking=True)

    print("ScaledDotProductAttentionLayer.forward(embeddings):")
    print(ScaledDotProductAttentionLayer.forward(embeddings))
    print("ScaledDotProductAttentionLayer.forward(embeddings).shape:")
    print(ScaledDotProductAttentionLayer.forward(embeddings).shape)
