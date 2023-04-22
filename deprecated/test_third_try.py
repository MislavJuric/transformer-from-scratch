"""
    Note:   This training loop was written before Transformer class had been written; now I'm using that class, so this is code is deprecated.

    Forward pass implementation
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
masking = False

TransformerDecoderInstance = TransformerDecoder(vocab_size=vocab_size, d_model=embedding_dim_target, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, numberOfDecoderBlocks=numberOfDecoderBlocks, masking=masking)

bpemb_instance_target = TransformerDatasetInstance.return_bpemb_target_instance()
embedding_layer_target = TransformerDatasetInstance.return_embedding_layer_target()

softmax_fn = torch.nn.Softmax(dim=0)

for sentence_index in range(0, len(TransformerDatasetInstance)):
    # encoder pass
    current_sentence = TransformerDatasetInstance[sentence_index][0]
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
    current_token = bpemb_instance_target.BOS_str
    BOS_index = bpemb_instance_target.BOS
    current_token_embedding = embedding_layer_target(torch.tensor(BOS_index))
    current_generated_sequence_length = 0
    generated_sequence = current_token_embedding
    generated_sequence_list = [current_token_embedding]
    generated_sequence_decoded = [current_token]
    generated_sequence_decoded.append(current_token)
    max_sequence_length = 300 # a parameter to prevent endless text generation if EOS token isn't generated
    while ((current_token != bpemb_instance_target.EOS_str) and (current_generated_sequence_length <= max_sequence_length)):
        # debug prints
        """
        print("generated_sequence:")
        print(generated_sequence)
        print("current_token:")
        print(current_token)
        """
        token_logits = TransformerDecoderInstance(generated_sequence, K, V)
        token_probabilities = softmax_fn(token_logits)
        # debug prints
        """
        print("token_probabilites.shape:")
        print(token_probabilites.shape)
        print("token_probabilites[-1].shape:")
        print(token_probabilites[-1].shape)
        print("token_probabilites[-1]:")
        print(token_probabilites[-1])
        """
        token_index_with_highest_probability = np.argmax(token_probabilities.detach().numpy()[-1])
        # debug prints
        """
        print("token_index_with_highest_probability:")
        print(token_index_with_highest_probability)
        """
        token_index_with_highest_probability = int(token_index_with_highest_probability)
        current_decoded_token = bpemb_instance_target.decode_ids([token_index_with_highest_probability])
        current_token = current_decoded_token
        generated_sequence_decoded.append(current_token)
        current_token_embedding = embedding_layer_target(torch.tensor(token_index_with_highest_probability))
        new_generated_sequence_list = []
        # generate the new embedding matrix with the new token embedding added to it
        for generated_embedding in generated_sequence_list:
            new_generated_sequence_list.append(generated_embedding)
        new_generated_sequence_list.append(current_token_embedding)
        generated_sequence_list = new_generated_sequence_list
        generated_sequence = torch.stack(new_generated_sequence_list, dim=0)
        current_generated_sequence_length = current_generated_sequence_length + 1
    print("generated_sequence_decoded:")
    print(generated_sequence_decoded)
    break
