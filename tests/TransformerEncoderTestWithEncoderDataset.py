"""
    A simple test to check if everything is allright with the Transformer Encoder implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import numpy as np

from models.TransformerEncoder import TransformerEncoder
from utils.EncoderDataset import EncoderDataset

if __name__ == "__main__":
    EncoderDatasetInstance = EncoderDataset(path_to_source_language_txt_file="../dataset/train/train_en.txt", language="en")
    first_sentence_embeddings = EncoderDatasetInstance[0]

    embedding_dim = int(first_sentence_embeddings.shape[1])
    print("embedding_dim:")
    print(embedding_dim)
    h = 8
    d_k = 64
    d_v = 64
    d_ff = 2048
    numberOfEncoderBlocks=6

    TransformerEncoderInstance = TransformerEncoder(d_model=embedding_dim, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfEncoderBlocks=numberOfEncoderBlocks)

    encoder_outputs, K, V = TransformerEncoderInstance.forward(first_sentence_embeddings)
    print("encoder_outputs:")
    print(encoder_outputs)
    print("encoder_outputs.shape:")
    print(encoder_outputs.shape)
    print("K.shape:")
    print(K.shape)
    print("V.shape:")
    print(V.shape)
