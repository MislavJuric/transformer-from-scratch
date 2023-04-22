"""
    Note:   This code tests the ScaledDotProductAttention_first_try.py, which is deprecated and due to the deprecation this code isn't updated.

    A simple test to check if everything is allright with the Scaled Dot-Product Attention implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import numpy as np

from layers.ScaledDotProductAttention import ScaledDotProductAttention

if __name__ == "__main__":
    d_k = 512
    d_v = 512

    ScaledDotProductAttentionLayer = ScaledDotProductAttention(d_k=d_k, d_v=d_v)

    number_of_samples = 100

    Q = np.random.rand(number_of_samples, d_k)
    K = np.random.rand(number_of_samples, d_k)
    V = np.random.rand(number_of_samples, d_v)

    print("ScaledDotProductAttentionLayer.forward(Q, K, V):")
    print(ScaledDotProductAttentionLayer.forward(Q, K, V))
