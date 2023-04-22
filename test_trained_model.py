"""
    Forward pass implementation
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
TransformerInstance.load_state_dict(torch.load("transformer_model_trained_on_mini_train_dataset_weights_after_2048_epochs.pth"))

bpemb_instance_target = TransformerDatasetInstance.return_bpemb_target_instance()
embedding_layer_target = TransformerDatasetInstance.return_embedding_layer_target()

softmax_fn = torch.nn.Softmax(dim=-1)

TransformerInstance.eval()
for sentence_index in range(0, len(TransformerDatasetInstance)):
    current_source_sentence = TransformerDatasetInstance[sentence_index][0]
    # below code geneerates one word at a time (beggining from the BOS token), then appends that newly generated word to already existing sequence,
    # then generates a new word again etc.
    # TODO: could write the below code more concisely, I think
    BOS_index = bpemb_instance_target.BOS
    current_token = bpemb_instance_target.BOS_str
    current_token_embedding = embedding_layer_target(torch.tensor(BOS_index))
    current_generated_sequence_length = 0
    generated_sequence = current_token_embedding
    generated_sequence_list = [current_token_embedding]
    generated_sequence_decoded = [current_token]
    max_sequence_length = 3 # a parameter to prevent endless text generation if EOS token isn't generated
    while ((current_token != bpemb_instance_target.EOS_str) and (current_generated_sequence_length < max_sequence_length)):
        with torch.no_grad():
            print("generated_sequence.shape:")
            print(generated_sequence.shape)
            token_logits = TransformerInstance(current_source_sentence, generated_sequence)
            print("token_logits.shape:")
            print(token_logits.shape)
            token_probabilities = softmax_fn(token_logits)
            print("token_probabilities.shape:")
            print(token_probabilities.shape)
            if (token_probabilities.ndim == 1):
                token_index_with_highest_probability = np.argmax(token_probabilities.detach().numpy())
            elif (token_probabilities.ndim == 2):
                token_index_with_highest_probability = np.argmax(token_probabilities.detach().numpy()[-1, :])
            token_index_with_highest_probability = int(token_index_with_highest_probability)
            print("token_index_with_highest_probability:")
            print(token_index_with_highest_probability)
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
    print("generated_sequence_decoded for sentence at index " + str(sentence_index) + ":")
    print(generated_sequence_decoded)
    break