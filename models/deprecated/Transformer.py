"""
    Note:   I deprecated this code and decided to opt for decomposing the Transformer implementation into
            Transformer Encoder and Transformer Decoder, since from my perspective it will be easier to run inference
            and train in that way.

    Transformer implementation (from Attention Is All You Need, section 3.1 and The Illustrated Transformer)

    d_model ... dimension of the embedding
    h ... number of parallel attention layers
    d_k ... dimension of queries and keys
    d_v ... dimension of values
    d_ff ... dimension of inner feedforward network layer
    number_of_encoder_blocks ... number of encoder blocks
    number_of_decoder_blocks ... number of decoder blocks
    vocab_size ... vocabulary size (number of words in the vocabulary)
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, ".")

import torch
import numpy as np

from blocks.EncoderBlock import EncoderBlock
from blocks.DecoderBlock import DecoderBlock

class Transformer(torch.nn.Module):

    # I'm mixing camelCase with _ in variable names below; I know this isn't good practice, but here _ denotes a suffix and not a space
    def __init__(self, vocabSize, wordDict, endOfSequenceSymbol, d_model=512, h=8, d_k=64, d_v=64, d_ff=2048, numberOfEncoderBlocks=6, numberOfDecoderBlocks=6):
        super().__init__()

        self.vocabSize = vocabSize
        self.wordDict = wordDict
        self.endOfSequenceSymbol = endOfSequenceSymbol

        self.d_model = d_model

        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

        self.numberOfEncoderBlocks = numberOfEncoderBlocks
        self.numberOfDecoderBlocks = numberOfDecoderBlocks

        self.encoderBlocks = []
        for i in range(0, self.numberOfEncoderBlocks):
            self.encoderBlocks.append(EncoderBlock(d_model=self.d_model, h=self.h, d_k=self.d_k, d_v=self.d_v, d_ff=self.d_ff))

        self.EncoderOutputToKeysLayer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.EncoderOutputToValuesLayer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_v)

        self.decoderBlocks = []
        for i in range(0, self.numberOfDecoderBlocks):
            self.decoderBlocks.append(DecoderBlock(d_model=self.d_model, h=self.h, d_k=self.d_k, d_v=self.d_v, d_ff=self.d_ff))

        self.DecoderOutputToLogitsLayer = torch.nn.Linear(in_features=self.d_model, out_features=self.vocabSize)
        self.Softmax = torch.nn.Softmax(dim=1)

    def forward(self, embeddings):
        currentEncoderResult = embeddings

        for encoderBlock in self.encoderBlocks:
            currentEncoderResult = encoderBlock(embeddings=currentEncoderResult)

        K = self.EncoderOutputToKeysLayer(currentEncoderResult)
        V = self.EncoderOutputToValuesLAyer(currentEncoderResult)

        # go one word at a time and feed the previous output as the next input
        decodedSequenceToReturn = []

        first_word_embedding = embeddings[0]
        currentDecoderResult = first_word_embedding
        currentDecodedWord = ""

        while currentDecodedWord != self.endOfSequenceSymbol: # TODO: maybe put another termination condition here (if self.endOfSequenceSymbol never generates)?
            for decoderBlock in self.decoderBlocks:
                currentDecoderResult = decoderBlock(embeddings=currentDecoderResult, K=K, V=V)
            logits = self.DecoderOutputToLogitsLayer(currentDecoderResult)
            tokenProbabilities = self.Softmax(logits)
            tokenWithHighestProbabilityIndex = np.argmax(tokenProbabilities)
            currentDecodedToken = self.wordDict[tokenWithHighestProbabilityIndex]
            decodedSequenceToReturn.append(currentDecodedToken)
            if (currentDecodedToken == self.endOfSequenceSymbol):
                break

        return decodedSequenceToReturn
