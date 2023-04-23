"""
    A simple test to check if everything is allright with the Decoder block implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import torch

from blocks.DecoderBlock import DecoderBlock

embedding_dim = 512
h = 8
d_k = 512
d_v = 512
d_ff = 2048
masking = True

decoder_block = DecoderBlock(d_model=embedding_dim, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, masking=masking)

number_of_samples = 100

embeddings = torch.rand(number_of_samples, embedding_dim)
K = torch.rand(number_of_samples, d_k)
V = torch.rand(number_of_samples, d_v)

print("decoder_block.forward(embeddings, K, V):")
print(decoder_block.forward(embeddings, K, V))
print("decoder_block.forward(embeddings, K, V).shape:")
print(decoder_block.forward(embeddings, K, V).shape)
