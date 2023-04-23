"""
    Note:   This is only the forward pass from the training loop. I used this for debugging purposes.

    Training loop
"""

import numpy as np
import torch

from models.Transformer import Transformer

from utils.TransformerDataset import TransformerDataset

# TransformerDatasetInstance = TransformerDataset(path_to_source_language_txt_file="dataset/train/train_en.txt", path_to_target_language_txt_file="dataset/train/train_de.txt", source_language="en", target_language="de", source_language_vocab_size=200000, target_language_vocab_size=200000)
# I'm training on the much smaller test set to see if the loss will go down, as suggested in Andrej Karpathy's "A Recipie for Training Neural Networks"
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

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

TransformerInstance.train()

loss_fn = torch.nn.CrossEntropyLoss()
beta_1 = 0.9
beta_2 = 0.98
epsilon = 10**(-9)
optimizer = torch.optim.Adam(TransformerInstance.parameters(), lr=3e-4, betas=(beta_1, beta_2), eps=epsilon) # learning rate from Andrej Karpathy's blog post
                                                                                                             # "A Recipie for Training Neural Networks"

bpemb_instance_target = TransformerDatasetInstance.return_bpemb_target_instance()

# below code based on https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

number_of_epochs = 2048
softmax_fn = torch.nn.Softmax(dim=-1)
for epoch_number in range(0, number_of_epochs):
    #print("Epoch number: " + str(epoch_number))
    for sentence_index, (source_sentence_embeddings_matrix, target_sentence_embeddings_matrix, token_ids_target_sentence) in enumerate(TransformerDatasetInstance):
        # debug prints
        """
        print("token_ids_target_sentence: (true token ids)")
        print(token_ids_target_sentence)
        """
        # forward pass

        logits = TransformerInstance(source_sentence_embeddings_matrix, target_sentence_embeddings_matrix)
        break
    break
