"""
    Note: This is deprecated because I changed the dataset class design.

    Forward pass implementation
"""

import numpy as np
import torch

from models.TransformerEncoder import TransformerEncoder
from models.TransformerDecoder import TransformerDecoder
from utils.EncoderDataset import EncoderDataset
from utils.DecoderDataset import DecoderDataset

EncoderDatasetInstance = EncoderDataset(path_to_source_language_txt_file="dataset/test/2012/newstest2012_en.txt", language="en", vocab_size=200000)

first_sentence_embedding = EncoderDatasetInstance[0]
embedding_dim = int(first_sentence_embedding.shape[1])
h = 8
d_k = 64
d_v = 64
d_ff = 2048
numberOfEncoderBlocks=6

TransformerEncoderInstance = TransformerEncoder(d_model=embedding_dim, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfEncoderBlocks=numberOfEncoderBlocks)

DecoderDatasetInstance = DecoderDataset(path_to_target_language_txt_file="dataset/test/2012/newstest2012_de.txt", language="de", vocab_size=200000)
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

for sentence_index in range(0, len(EncoderDatasetInstance)):
    # encoder pass
    current_sentence = EncoderDatasetInstance[sentence_index]
    # debug prints
    """
    print("current_sentence.shape: (encoder)")
    print(current_sentence.shape)
    """
    _, K, V = TransformerEncoderInstance.forward(current_sentence)
    # debug prints
    """
    print("K.shape: (encoder)")
    print(K.shape)
    print("V.shape: (encoder)")
    print(V.shape)
    """
    # decoder pass (word by word)
    current_token = decoder_dataset_bpemb_instance.BOS_str
    BOS_index = decoder_dataset_bpemb_instance.BOS
    current_token_embedding = decoder_dataset_embedding_layer(torch.tensor(BOS_index))
    current_generated_sequence_length = 0
    generated_sequence = []
    max_sequence_length = 300 # a parameter to prevent endless text generation if EOS token isn't generated
    while ((current_token != decoder_dataset_bpemb_instance.EOS_str) and (current_generated_sequence_length <= max_sequence_length)):
        # debug prints
        """
        print("current_token:")
        print(current_token)
        """
        token_probabilites = TransformerDecoderInstance(current_token_embedding, K, V)
        token_index_with_highest_probability = np.argmax(token_probabilites.detach().numpy())
        # debug prints
        """
        print("token_index_with_highest_probability:")
        print(token_index_with_highest_probability)
        """
        token_index_with_highest_probability = int(token_index_with_highest_probability)
        current_decoded_token = decoder_dataset_bpemb_instance.decode_ids([token_index_with_highest_probability])
        generated_sequence.append(current_decoded_token)
        current_token = current_decoded_token
        current_token_embedding = decoder_dataset_embedding_layer(torch.tensor(token_index_with_highest_probability))
        current_generated_sequence_length = current_generated_sequence_length + 1
    print("generated_sequence:")
    print(generated_sequence)
    break
