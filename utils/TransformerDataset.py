import os
import torch
import bpemb

from .helper_functions import positional_encoding

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_source_language_txt_file, path_to_target_language_txt_file, source_language="en", target_language="de", source_language_vocab_size=200000, target_language_vocab_size=200000):
        source_language_txt_file = open(path_to_source_language_txt_file, "r")
        target_language_txt_file = open(path_to_target_language_txt_file, "r")
        self.all_lines_in_the_source_language_txt_file = source_language_txt_file.readlines()
        self.all_lines_in_the_target_language_txt_file = target_language_txt_file.readlines()
        self.bpemb_source = bpemb.BPEmb(lang=source_language, vs=source_language_vocab_size) # byte-pair encoding based embedding
        self.bpemb_target = bpemb.BPEmb(lang=target_language, vs=target_language_vocab_size)
        self.embedding_layer_source = torch.nn.Embedding.from_pretrained(torch.tensor(self.bpemb_source.vectors))
        self.embedding_layer_target = torch.nn.Embedding.from_pretrained(torch.tensor(self.bpemb_target.vectors))

    def __len__(self):
        return len(self.all_lines_in_the_source_language_txt_file)

    def __getitem__(self, index):
        # get the embeddings for source and target sentence
        requested_line_source = self.all_lines_in_the_source_language_txt_file[index]
        requested_line_target = self.all_lines_in_the_target_language_txt_file[index]
        token_ids_source_line = self.bpemb_source.encode_ids(requested_line_source)
        # add the BOS token to the beginning and the EOS token to the end of the source sentence
        token_ids_source_line.insert(0, self.bpemb_source.BOS)
        token_ids_source_line.append(self.bpemb_source.EOS)
        token_ids_target_line = self.bpemb_target.encode_ids(requested_line_target)
        # add the BOS token to the beginning and the EOS token to the end of the target sentence
        token_ids_target_line.insert(0, self.bpemb_target.BOS)
        token_ids_target_line.append(self.bpemb_target.EOS)
        embeddings_for_source_line = self.embedding_layer_source(torch.tensor(token_ids_source_line))
        embeddings_for_target_line = self.embedding_layer_target(torch.tensor(token_ids_target_line))
        embedding_dim_source = int(embeddings_for_source_line[0].shape[0])
        embedding_dim_target = int(embeddings_for_target_line[0].shape[0])

        # calculate the positional encoding for each source embedding
        positional_encodings_for_source_line = []
        for token_index, embedding in enumerate(embeddings_for_source_line):
            positional_encoding_for_current_embedding = positional_encoding(embedding=embedding, pos=token_index, d_model=embedding_dim_source)
            positional_encodings_for_source_line.append(positional_encoding_for_current_embedding)
        positional_encodings_for_source_line = torch.FloatTensor(positional_encodings_for_source_line)

        # add the embeddings and the positional encodings and return the result
        source_embeddings_with_positional_encoding = torch.add(embeddings_for_source_line, positional_encodings_for_source_line)

        # calculate the positional encoding for each target embedding
        positional_encodings_for_target_line = []
        for token_index, embedding in enumerate(embeddings_for_target_line):
            positional_encoding_for_current_embedding = positional_encoding(embedding=embedding, pos=token_index, d_model=embedding_dim_target)
            positional_encodings_for_target_line.append(positional_encoding_for_current_embedding)
        positional_encodings_for_target_line = torch.FloatTensor(positional_encodings_for_target_line)

        # add the embeddings and the positional encodings and return the result
        target_embeddings_with_positional_encoding = torch.add(embeddings_for_target_line, positional_encodings_for_target_line)

        return source_embeddings_with_positional_encoding, target_embeddings_with_positional_encoding, token_ids_target_line # returning token_ids_target_line since I find the
                                                                                                                             # reversal from the embedding to the token ID convoluted

    def return_bpemb_target_instance(self):
        return self.bpemb_target

    def return_embedding_layer_target(self):
        return self.embedding_layer_target
