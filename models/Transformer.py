"""
    Transformer implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, ".")

import torch
import numpy as np

from models.TransformerEncoder import TransformerEncoder
from models.TransformerDecoder import TransformerDecoder

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, masking, d_model_encoder=512, h_encoder=8, d_k_encoder=64, d_v_encoder=64, d_ff_encoder=2048, number_of_encoder_blocks=6, d_model_decoder=512, h_decoder=8, d_k_decoder=64, d_v_decoder=64, d_ff_decoder=2048, number_of_decoder_blocks=6):
        super().__init__()

        self.transformer_encoder = TransformerEncoder(d_model=d_model_encoder, h=h_encoder, d_k=d_k_encoder, d_v=d_v_encoder, d_ff=d_ff_encoder, number_of_encoder_blocks=number_of_encoder_blocks)
        self.transformer_decoder = TransformerDecoder(vocab_size=vocab_size, d_model=d_model_decoder, h=h_decoder, d_k=d_k_decoder, d_v=d_v_decoder, d_ff=d_ff_decoder, number_of_decoder_blocks=number_of_decoder_blocks, masking=masking)

    def forward(self, source_sentence_embeddings, target_sentence_embeddings):
        _, K, V = self.transformer_encoder(source_sentence_embeddings)
        logits = self.transformer_decoder(target_sentence_embeddings, K, V)
        return logits
