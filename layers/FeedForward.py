"""
    Position-wise Feed-Forward Network implementation (from Attention Is All You Need, section 3.3)

    d_model ... dimensionsionality of input and output
    d_ff ... dimensionality of the inner layer
"""

import torch
import numpy as np

class FeedForward(torch.nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.first_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_ff)
        self.relu_fn = torch.nn.ReLU()
        self.second_layer = torch.nn.Linear(in_features=self.d_ff, out_features=self.d_model)

    def forward(self, input):
        return self.second_layer(self.relu_fn(self.first_layer(input)))
