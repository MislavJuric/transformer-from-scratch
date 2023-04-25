"""
    Transformer Decoder implementation (from Attention Is All You Need, section 3.1 and The Illustrated Transformer)

    vocab_size ... size of the vocabulary (number of words in the vocabulary)
    d_model ... dimension of the embedding
    h ... number of parallel attention layers
    d_k ... dimension of queries and keys
    d_v ... dimension of values
    d_ff ... dimension of inner feedforward network layer
    number_of_decoder_blocks ... number of encoder blocks
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, ".")

import torch
import numpy as np

from blocks.DecoderBlock import DecoderBlock

class TransformerDecoder(torch.nn.Module):
    def __init__(self, vocab_size, masking, d_model=512, h=8, d_k=64, d_v=64, d_ff=2048, number_of_decoder_blocks=6):
        super().__init__()

        self.vocab_size = vocab_size

        self.d_model = d_model

        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

        self.masking = masking

        self.number_of_decoder_blocks = number_of_decoder_blocks

        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(0, self.number_of_decoder_blocks):
            self.decoder_blocks.append(DecoderBlock(d_model=self.d_model, h=self.h, d_k=self.d_k, d_v=self.d_v, d_ff=self.d_ff, masking=self.masking))

        self.decoder_output_to_logits_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.vocab_size)

    def forward(self, tokenEmbedding, K, V):
        currentResult = tokenEmbedding

        for decoderBlock in self.decoder_blocks:
            currentResult = decoderBlock(embeddings=currentResult, K=K, V=V)
            
        logits = self.decoder_output_to_logits_layer(currentResult)
        return logits
