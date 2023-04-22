"""
    Note:   I realized here that I needed to redesign my dataset classes for the Encoder and the Decoder
            and since this file uses the dataset class(es), I needed to re-write this as well.

    Training loop
"""

import numpy as np
import torch

from models.TransformerEncoder import TransformerEncoder
from models.TransformerDecoder import TransformerDecoder
from utils.EncoderDataset import EncoderDataset
from utils.DecoderDataset import DecoderDataset

EncoderDatasetInstance = EncoderDataset(path_to_source_language_txt_file="dataset/train/train_en.txt", language="en", vocab_size=200000)

first_sentence_embedding = EncoderDatasetInstance[0]
embedding_dim = int(first_sentence_embedding.shape[1])
h = 8
d_k = 64
d_v = 64
d_ff = 2048
numberOfEncoderBlocks=6

TransformerEncoderInstance = TransformerEncoder(d_model=embedding_dim, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfEncoderBlocks=numberOfEncoderBlocks)

DecoderDatasetInstance = DecoderDataset(path_to_target_language_txt_file="dataset/train/train_de.txt", language="de", vocab_size=200000)
decoder_dataset_bpemb_instance = DecoderDatasetInstance.return_bpemb_instance()
decoder_dataset_embedding_layer = DecoderDatasetInstance.return_embedding_layer()

vocab_size = 200000
first_sentence_embedding = DecoderDatasetInstance[0]
embedding_dim = int(first_sentence_embedding.shape[1])
h = 8
d_k = 64
d_v = 64
d_ff = 2048
numberOfDecoderBlocks=6
masking = False

TransformerDecoderInstance = TransformerDecoder(vocab_size=vocab_size, d_model=embedding_dim, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfDecoderBlocks=numberOfDecoderBlocks, masking=masking)

# dataloaders
# this isn't good because this returns shuffled data instances, where the source sentence may not correspond to the target sentence
# I need to re-implement the Dataset to return both the source and the target sentence
encoder_dataloader = torch.utils.data.DataLoader(EncoderDatasetInstance, batch_size=64, shuffle=True)
decoder_dataloader = torch.utils.data.DataLoader(DecoderDatasetInstance, batch_size=64, shuffle=True)
