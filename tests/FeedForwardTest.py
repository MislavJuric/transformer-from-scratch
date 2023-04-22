"""
    A simple test to check if everything is allright with the Position-wise Feed-Forward Network implementation
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, "..")

import numpy as np

from layers.FeedForward import FeedForward

if __name__ == "__main__":
    d_model = 512
    d_ff = 2048

    FeedForward = FeedForward(d_model=d_model, d_ff=d_ff)

    number_of_samples = 100

    embeddings = np.random.rand(number_of_samples, d_model)

    print("FeedForward.forward(embeddings):")
    print(FeedForward.forward(embeddings))
    print("FeedForward.forward(embeddings).shape:")
    print(FeedForward.forward(embeddings).shape)
