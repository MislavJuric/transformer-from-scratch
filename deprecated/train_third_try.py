"""
    Note:   This training loop was written before Transformer class had been written; now I'm using that class, so this is code is deprecated.

    Training loop
"""

import numpy as np
import torch

from models.TransformerEncoder import TransformerEncoder
from models.TransformerDecoder import TransformerDecoder

from utils.TransformerDataset import TransformerDataset

TransformerDatasetInstance = TransformerDataset(path_to_source_language_txt_file="dataset/test/2012/newstest2012_en.txt", path_to_target_language_txt_file="dataset/test/2012/newstest2012_de.txt", source_language="en", target_language="de", source_language_vocab_size=200000, target_language_vocab_size=200000)

first_sentence_embedding_for_encoder = TransformerDatasetInstance[0][0]
embedding_dim_source = int(first_sentence_embedding_for_encoder.shape[1])
h = 8
d_k = 64
d_v = 64
d_ff = 2048
numberOfEncoderBlocks=6

TransformerEncoderInstance = TransformerEncoder(d_model=embedding_dim_source, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfEncoderBlocks=numberOfEncoderBlocks)

vocab_size = 200000
first_sentence_embedding_for_decoder = TransformerDatasetInstance[0][1]
embedding_dim_target = int(first_sentence_embedding_for_decoder.shape[1])
h = 8
d_k = 64
d_v = 64
d_ff = 2048
numberOfDecoderBlocks=6
masking = True

TransformerDecoderInstance = TransformerDecoder(vocab_size=vocab_size, d_model=embedding_dim_target, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfDecoderBlocks=numberOfDecoderBlocks, masking=masking)

# TODO: put epochs
for source_sentence_embeddings_matrix, target_sentence_embeddings_matrix in TransformerDatasetInstance:
    # forward pass
    # encoder pass
    _, K, V = TransformerEncoderInstance(source_sentence_embeddings_matrix)
    # decoder pass
    next_token_probability_distributions_for_each_timestep = TransformerDecoderInstance(target_sentence_embeddings_matrix, K, V)
    # backward pass
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    break
