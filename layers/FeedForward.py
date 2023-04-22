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

        self.FirstLayer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_ff)
        self.ReLU = torch.nn.ReLU()
        self.SecondLayer = torch.nn.Linear(in_features=self.d_ff, out_features=self.d_model)

    def forward(self, input):
        # type casting (if needed)
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
            # type conversion below is neccesary to avoid errors
            input = input.float()
        """
        
        return self.SecondLayer(self.ReLU(self.FirstLayer(input)))
