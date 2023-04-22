"""
    Note:   Here I tried to batch together sentences using dataloader. From what I found, this doesn't work if the sentences aren't the same sizes
            (either padded or shortened to match the size limit). I opted to train the model on one sentence at a time instead; I leave this
            as future work.

    Training loop
"""

import numpy as np
import torch

from models.TransformerEncoder import TransformerEncoder
from models.TransformerDecoder import TransformerDecoder

from utils.TransformerDataset import TransformerDataset

def collate_fn(batch):
    return tuple(zip(*batch))

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
masking = False

TransformerDecoderInstance = TransformerDecoder(vocab_size=vocab_size, d_model=embedding_dim_target, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfDecoderBlocks=numberOfDecoderBlocks, masking=masking)

TransformerDatasetDataLoader = torch.utils.data.DataLoader(TransformerDatasetInstance, batch_size=64, shuffle=True, collate_fn=collate_fn)

for source_sentence_batch, target_sentence_batch in TransformerDatasetDataLoader:
    # encoder pass
    _, K, V = TransformerEncoderInstance.forward(source_sentence_batch)
