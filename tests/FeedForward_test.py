"""
    A simple test to check if everything is allright with the Position-wise Feed-Forward Network implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import torch

from layers.FeedForward import FeedForward

d_model = 512
d_ff = 2048

feed_forward_layer = FeedForward(d_model=d_model, d_ff=d_ff)

number_of_samples = 100

embeddings = torch.rand(number_of_samples, d_model)

print("feed_forward_layer.forward(embeddings):")
print(feed_forward_layer.forward(embeddings))
print("feed_forward_layer.forward(embeddings).shape:")
print(feed_forward_layer.forward(embeddings).shape)
