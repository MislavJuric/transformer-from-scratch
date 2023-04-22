"""
    A simple test to check if everything is allright with the Transformer Encoder implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import numpy as np

from models.TransformerEncoder import TransformerEncoder

if __name__ == "__main__":
    embedding_dim = 512
    h = 8
    d_k = 512
    d_v = 512
    d_ff = 2048
    numberOfEncoderBlocks=6

    TransformerEncoderInstance = TransformerEncoder(d_model=embedding_dim, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfEncoderBlocks=numberOfEncoderBlocks)

    number_of_samples = 100

    embeddings = np.random.rand(number_of_samples, embedding_dim)

    encoder_outputs, K, V = TransformerEncoderInstance.forward(embeddings)
    print("encoder_outputs:")
    print(encoder_outputs)
    print("encoder_outputs.shape:")
    print(encoder_outputs.shape)
    print("K.shape:")
    print(K.shape)
    print("V.shape:")
    print(V.shape)
