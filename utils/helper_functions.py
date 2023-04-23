import math

# TODO: maybe this could be written more efficiently
def positional_encoding(embedding, pos, d_model):
    # TODO: assert that embedding.shape[1] == d_model
    to_return = []
    for i in range(0, len(embedding)):
        if ((i % 2) == 0):
            sin_argument = pos / 10000**((2*i)/d_model)
            to_return.append(math.sin(sin_argument))
        else:
            cos_argument = pos / 10000**((2*i)/d_model)
            to_return.append(math.cos(pos / 10000**((2*i)/d_model)))
    return to_return
