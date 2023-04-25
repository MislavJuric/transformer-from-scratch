"""
    Note:   This train.py contains an if statement I used to see if the low loss values corresponded to correct next token predictions

    Training loop
"""

import numpy as np
import torch

from models.Transformer import Transformer

from utils.TransformerDataset import TransformerDataset

# I'm training on the much smaller training set to see if the loss will go down, as suggested in Andrej Karpathy's "A Recipie for Training Neural Networks"
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
optimizer = torch.optim.Adam(TransformerInstance.parameters(), lr=3e-5, betas=(beta_1, beta_2), eps=epsilon)

bpemb_instance_target = TransformerDatasetInstance.return_bpemb_target_instance()

# below code based on https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

number_of_epochs = 2048
softmax_fn = torch.nn.Softmax(dim=-1)
for epoch_number in range(0, number_of_epochs):
    print("Epoch number: " + str(epoch_number))
    for sentence_index, (source_sentence_embeddings_matrix, target_sentence_embeddings_matrix, token_ids_target_sentence) in enumerate(TransformerDatasetInstance):
        # forward pass

        logits = TransformerInstance(source_sentence_embeddings_matrix, target_sentence_embeddings_matrix)

        # loss calculation

        # let's construct the target matrix for the loss function; it will have the same shape as logits, but it will contain the probability distribution of the next token
        # to be predicted for each token; that (true) next token will have the probability of 1, while others will have the probability of 0
        true_token_probability_distributions = torch.zeros((int(logits.shape[0]), vocab_size))
        for current_token_probability_distribution_index in range(0, (len(logits) - 1)):
            true_next_token_index = token_ids_target_sentence[current_token_probability_distribution_index + 1]
            true_token_probability_distributions[current_token_probability_distribution_index][true_next_token_index] = 1
        # the true next predicted token of the last token should be EOS; my last two token probability distributions will place the highest probability on EOS
        EOS_token_index_target_language = bpemb_instance_target.EOS
        true_token_probability_distributions[int(logits.shape[0]) - 1][EOS_token_index_target_language] = 1

        loss = loss_fn(logits, true_token_probability_distributions)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # diagnostic prints
        size = len(TransformerDatasetInstance)
        #if sentence_index % 6 == 0:
        loss, current = loss.item(), sentence_index
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # if below used for debugging purposes
        """
        if (loss < 0.02):
            print("token_ids_target_sentence: (true token ids)")
            print(token_ids_target_sentence)
            print("Softmax applied over all logits at once: (predicted next token ids)")
            next_token_probability_distributions = softmax_fn(logits)
            for next_token_probability_distribution in next_token_probability_distributions:
                next_token_index_with_highest_probability = np.argmax(next_token_probability_distribution.detach().numpy())
                print("next_token_index_with_highest_probability:")
                print(next_token_index_with_highest_probability)
            print("Softmax applied to one logit vector at a time: (predicted next token ids)")
            for logits_for_the_next_token in logits:
                next_token_probability_distribution = softmax_fn(logits_for_the_next_token)
                next_token_index_with_highest_probability = np.argmax(next_token_probability_distribution.detach().numpy())
                print("next_token_index_with_highest_probability:")
                print(next_token_index_with_highest_probability)
        """

# save the model
weights_filename = "transformer_model_trained_on_mini_train_dataset_weights_after_" + str(number_of_epochs) + "_epochs.pth"
torch.save(TransformerInstance.state_dict(), weights_filename)
print("Saved PyTorch Transformer State to " + weights_filename)
