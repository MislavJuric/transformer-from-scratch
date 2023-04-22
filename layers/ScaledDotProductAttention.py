"""
    Scaled Dot-Product Attention implementation (from Attention Is All You Need, section 3.2.1)

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

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, masking=False):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.masking = masking

        self.EmbeddingsToQueries = torch.nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.EmbeddingsToKeys = torch.nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.EmbeddingsToValues = torch.nn.Linear(in_features=self.d_model, out_features=self.d_v)

        self.Softmax = torch.nn.Softmax(dim=-1)

    def is_matrix(self, arg):
        shape_of_arg = arg.shape
        shape_of_arg = list(shape_of_arg)
        if (len(shape_of_arg) > 1):
            return True
        else:
            return False

    def forward(self, embeddings):
        # TODO: an assert here that embeddings.shape[1] == self.d_model

        #number_of_embeddings = embeddings.shape[0]

        # type casting (if needed)
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
            # type conversion below is neccesary to avoid errors
            embeddings = embeddings.to(torch.float32)
        """

        Q = self.EmbeddingsToQueries(embeddings)
        K = self.EmbeddingsToKeys(embeddings)
        V = self.EmbeddingsToValues(embeddings)

        if (self.masking == False):
            if (self.is_matrix(K)):
                # debug prints
                """
                print("Q.shape: (ScaledDotProductAttention)")
                print(Q.shape)
                print("K.shape: (ScaledDotProductAttention)")
                print(K.shape)
                print("self.Softmax(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))).shape: (ScaledDotProductAttention)")
                print(self.Softmax(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))).shape)
                print("torch.matmul(self.Softmax(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))), V).shape: (ScaledDotProductAttention)")
                print(torch.matmul(self.Softmax(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))), V).shape)
                """
                return torch.matmul(self.Softmax(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))), V)
            else:
                # debug prints
                """
                print("Q.shape: (ScaledDotProductAttention)")
                print(Q.shape)
                print("K.shape: (ScaledDotProductAttention)")
                print(K.shape)
                print("self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))).shape: (ScaledDotProductAttention)")
                print(self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))).shape)
                print("torch.mul(self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))), V).shape: (ScaledDotProductAttention)")
                print(torch.mul(self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))), V).shape)
                """
                return torch.mul(self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))), V)
        else:
            first_dimension_of_M = Q.size(dim=0)
            second_dimension_of_M = K.size(dim=0) # first dimension of K because K gets transposed, so first dimension becomes the second dimension
            M = torch.zeros(first_dimension_of_M, second_dimension_of_M) # the look-ahead mask matrix
            for row_index in range(0, first_dimension_of_M):
                for column_index in range(row_index + 1, second_dimension_of_M):
                    M[row_index][column_index] = -float('inf')

            if (self.is_matrix(K)):
                # debug prints
                """
                print("Q.shape: (ScaledDotProductAttention)")
                print(Q.shape)
                print("K.shape: (ScaledDotProductAttention)")
                print(K.shape)
                print("self.Softmax(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))).shape (ScaledDotProductAttention)")
                print(self.Softmax(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))).shape)
                print("self.Softmax(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))):")
                print(self.Softmax(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))))
                print("torch.matmul(self.Softmax(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))), V).shape: (ScaledDotProductAttention)")
                print(torch.matmul(self.Softmax(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))), V).shape)
                """
                return torch.matmul(self.Softmax(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))), V)
            else:
                # debug prints
                """
                print("Q.shape: (ScaledDotProductAttention)")
                print(Q.shape)
                print("K.shape: (ScaledDotProductAttention)")
                print(K.shape)
                print("self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))).shape: (ScaledDotProductAttention)")
                print(self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))).shape)
                print("(self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))) * V).shape: (ScaledDotProductAttention)")
                print((self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))) * V).shape)
                """
                return self.Softmax(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))) * V
