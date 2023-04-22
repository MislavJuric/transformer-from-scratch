"""
    Note:   This code is deprecated because when shuffling the data with the PyTorch DataLoader, I need to make sure that the Encoder and the Decoder are looking
            at the same sentence. This wasn't possible if they are in two seperate classes (or, at least, I found it way more convenient to put the Dataset class
            into one class), so this code is deprecated.
"""

import os
import torch
import bpemb

from .helper_functions import positional_encoding

class EncoderDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_source_language_txt_file, language="en", vocab_size=200000):
        self.path_to_source_language_txt_file = path_to_source_language_txt_file
        source_language_txt_file = open(self.path_to_source_language_txt_file, "r")
        self.all_lines_in_the_source_language_txt_file = source_language_txt_file.readlines()
        self.bpemb_instance = bpemb.BPEmb(lang=language, vs=vocab_size) # byte-pair encoding based embedding
        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.tensor(self.bpemb_instance.vectors))

    def __len__(self):
        return len(self.all_lines_in_the_source_language_txt_file)

    def __getitem__(self, index):
        # get the embeddings
        requested_line = self.all_lines_in_the_source_language_txt_file[index]
        token_ids = self.bpemb_instance.encode_ids(requested_line)
        embeddings_for_current_line = self.embedding_layer(torch.tensor(token_ids))
        embedding_dim = int(embeddings_for_current_line[0].shape[0])

        # calculate the positional encoding for each embedding
        positional_encodings_for_current_line = []
        for token_index, embedding in enumerate(embeddings_for_current_line):
            positional_encoding_for_current_embedding = positional_encoding(embedding=embedding, pos=token_index, d_model=embedding_dim)
            positional_encodings_for_current_line.append(positional_encoding_for_current_embedding)
        positional_encodings_for_current_line = torch.FloatTensor(positional_encodings_for_current_line)

        # add the embeddings and the positional encodings and return the result
        embeddings_with_positional_encoding = torch.add(embeddings_for_current_line, positional_encodings_for_current_line)
        return embeddings_with_positional_encoding
