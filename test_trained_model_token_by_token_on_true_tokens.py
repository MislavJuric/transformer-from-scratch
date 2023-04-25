"""
    Forward pass implementation (word by word); I don't take the next most probable token, but rather I get the ground truth token from the target sentence
"""

import numpy as np
import torch

from models.Transformer import Transformer

from utils.TransformerDataset import TransformerDataset

from utils.helper_functions import positional_encoding

# Get cpu, gpu or mps device for testing.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

TransformerDatasetInstance = TransformerDataset(path_to_source_language_txt_file="dataset/mini_train/train_en.txt", path_to_target_language_txt_file="dataset/mini_train/train_de.txt", source_language="en", target_language="de", source_language_vocab_size=200000, target_language_vocab_size=200000)

# encoder parameters
first_sentence_embedding_for_encoder = TransformerDatasetInstance[0][0]
d_model_encoder = int(first_sentence_embedding_for_encoder.shape[1])
h_encoder = 8
d_k_encoder = 64
d_v_encoder = 64
d_ff_encoder = 2048
number_of_encoder_blocks=6
# decoder parameters
vocab_size = 200000
first_sentence_embedding_for_decoder = TransformerDatasetInstance[0][1]
d_model_decoder = int(first_sentence_embedding_for_decoder.shape[1])
h_decoder = 8
d_k_decoder = 64
d_v_decoder = 64
d_ff_decoder = 2048
number_of_decoder_blocks=6
masking = True

TransformerInstance = Transformer(vocab_size=vocab_size, masking=masking, d_model_encoder=d_model_encoder, h_encoder=h_encoder, d_k_encoder=d_k_encoder, d_v_encoder=d_v_encoder, d_ff_encoder=d_ff_encoder, number_of_encoder_blocks=number_of_encoder_blocks, d_model_decoder=d_model_decoder, h_decoder=h_decoder, d_k_decoder=d_k_decoder, d_v_decoder=d_v_decoder, d_ff_decoder=d_ff_decoder, number_of_decoder_blocks=number_of_decoder_blocks)
TransformerInstance.load_state_dict(torch.load("model_weights/transformer_model_trained_on_mini_train_dataset_weights_after_2048_epochs.pth"))

bpemb_instance_target = TransformerDatasetInstance.return_bpemb_target_instance()
embedding_layer_target = TransformerDatasetInstance.return_embedding_layer_target()

softmax_fn = torch.nn.Softmax(dim=-1)

TransformerInstance.eval()
for sentence_index, (source_sentence_embeddings_matrix, target_sentence_embeddings_matrix, token_ids_target_sentence) in enumerate(TransformerDatasetInstance):
    current_token_index = 0
    current_token_embedding_with_positional_encoding = target_sentence_embeddings_matrix[current_token_index, :]
    current_token_embedding_with_positional_encoding = current_token_embedding_with_positional_encoding.unsqueeze(dim=0)

    generated_sequence_list = [current_token_embedding_with_positional_encoding]
    generated_sequence = current_token_embedding_with_positional_encoding
    generated_token_indices = []
    generated_sequence_decoded = []
    max_sequence_length = 50
    next_token_index_with_highest_probability = None
    while ((next_token_index_with_highest_probability != bpemb_instance_target.EOS) and (len(generated_token_indices) <= max_sequence_length)):
        # forward pass
        logits = TransformerInstance(source_sentence_embeddings_matrix, generated_sequence)
        next_token_probability_distributions = softmax_fn(logits)
        try:
            next_token_index_with_highest_probability = np.argmax(next_token_probability_distributions.detach().numpy()[-1, :])
        except:
            next_token_index_with_highest_probability = np.argmax(next_token_probability_distributions.detach().numpy())
        current_decoded_token = bpemb_instance_target.decode_ids([int(next_token_index_with_highest_probability)])
        generated_sequence_decoded.append(current_decoded_token)
        current_token_index = current_token_index + 1
        current_token_embedding_with_positional_encoding = target_sentence_embeddings_matrix[current_token_index, :]
        current_token_embedding_with_positional_encoding = current_token_embedding_with_positional_encoding.unsqueeze(dim=0)
        # I add the true next token embedding regrardless of what the next_token_index_with_highest_probability is
        generated_sequence_list.append(current_token_embedding_with_positional_encoding)
        generated_sequence = torch.cat(generated_sequence_list, dim=0)
        generated_token_indices.append(next_token_index_with_highest_probability)

    print("Translation for sentence at index " + str(sentence_index) + ":")
    print("token_ids_target_sentence:")
    print(token_ids_target_sentence)
    print("generated_token_indices:")
    print(generated_token_indices)
    print("generated_sequence_decoded:")
    print(generated_sequence_decoded)
