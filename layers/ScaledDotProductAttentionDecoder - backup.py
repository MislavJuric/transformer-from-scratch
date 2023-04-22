"""
    Scaled Dot-Product Attention implementation for the Decoder (from The Illustrated Transformer)
    From what I understood, this attention layer takes in Keys and Values matrices from the first_sublock_output
    of the encoder stack, so I have to modify the Scaled Dot-Product Attention implementation for the decoder.

    d_model ... embedding dimension
    d_k ... dimension of queries and keys
    d_v ... dimension of values
    masking ... a boolean indicating if look-ahead masking is to be applied (implemented based on https://medium.com/mlearning-ai/how-do-self-attention-masks-work-72ed9382510f)

    Q ... queries matrix (of dimensions number_of_samples x d_k)
    K ... keys matrix (of dimensions number_of_samples x d_k)
    V ... values matrix (of dimensions number_of_samples x d_v)
"""

import torch
import numpy as np
import math

class ScaledDotProductAttentionDecoder(torch.nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, masking=False):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.masking = masking

        self.EmbeddingsToQueries = torch.nn.Linear(in_features=self.d_model, out_features=self.d_k)

        self.Softmax = torch.nn.Softmax(dim=-1)

    def is_matrix(self, arg):
        shape_of_arg = arg.shape
        shape_of_arg = list(shape_of_arg)
        if (len(shape_of_arg) > 1):
            return True
        else:
            return False

    def forward(self, embeddings, K, V):
        # TODO: an assert here that embeddings.shape[1] == self.d_model
        # TODO: an assert here that V.shape[1] == self.d_v

        number_of_embeddings = embeddings.shape[0]

        # type casting (if needed)
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
            # type conversion below is neccesary to avoid errors
            embeddings = embeddings.to(torch.float32)
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
            K = K.to(torch.float32)
        if isinstance(V, np.ndarray):
            V = torch.from_numpy(V)
            V = V.to(torch.float32)
        """

        Q = self.EmbeddingsToQueries(embeddings)

        if (self.masking == False):
            if (self.is_matrix(K)):
                # debug prints
                """
                print("Q.shape: (ScaledDotProductAttentionDecoder)")
                print(Q.shape)
                print("K.shape: (ScaledDotProductAttentionDecoder)")
                print(K.shape)
                print("self.SoftmaxForVectors(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))).shape: (ScaledDotProductAttentionDecoder)")
                print(self.SoftmaxForVectors(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))).shape)
                print("torch.matmul(self.SoftmaxForVectors(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))), V).shape: (ScaledDotProductAttentionDecoder)")
                print(torch.matmul(self.SoftmaxForVectors(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))), V).shape)
                """
                return torch.matmul(self.Softmax(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))), V)
            else:
                return torch.mul(self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))), V)
        else:
            number_of_embeddings = embeddings.size(dim=0)
            first_dimension_of_K = K.size(dim=0)
            M = torch.zeros(number_of_embeddings, first_dimension_of_K) # the look-ahead mask matrix
            for row_index in range(0, number_of_embeddings):
                for column_index in range(row_index + 1, first_dimension_of_K):
                    M[row_index][column_index] = -float('inf')
            # debug prints
            """
            print("Q.shape: (ScaledDotProductAttentionDecoder)")
            print(Q.shape)
            print("K.shape: (ScaledDotProductAttentionDecoder)")
            print(K.shape)
            print("self.SoftmaxForMatrices(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))).shape (ScaledDotProductAttentionDecoder)")
            print(self.SoftmaxForMatrices(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))).shape)
            print("torch.matmul(self.SoftmaxForMatrices(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))), V).shape: (ScaledDotProductAttentionDecoder)")
            print(torch.matmul(self.SoftmaxForMatrices(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))), V).shape)
            """
            return torch.matmul(self.Softmax(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))), V)
