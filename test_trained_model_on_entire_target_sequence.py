"""
    Forward pass implementation (on entire target sequence)
"""

import numpy as np
import torch

from models.Transformer import Transformer

from utils.TransformerDataset import TransformerDataset

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
    print("Translation of sentence " + str(sentence_index) + ":")
    print("token_ids_target_sentence:")
    print(token_ids_target_sentence)

    generated_sequence_decoded = []
    predicted_token_ids = []
    with torch.no_grad():
        all_token_logits = TransformerInstance(source_sentence_embeddings_matrix, target_sentence_embeddings_matrix)
        for token_logits in all_token_logits:
            token_probabilities = softmax_fn(token_logits)
            token_index_with_highest_probability = np.argmax(token_probabilities.detach().numpy())
            predicted_token_ids.append(token_index_with_highest_probability)
            current_decoded_token = bpemb_instance_target.decode_ids([int(token_index_with_highest_probability)])
            generated_sequence_decoded.append(current_decoded_token)

    print("predicted_token_ids:")
    print(predicted_token_ids)
    print("generated_sequence_decoded:")
    print(generated_sequence_decoded)
