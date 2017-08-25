from torch import nn


class RNNEncoder(nn.Module):
    def __init__(self, num_characters, dim):
        super(RNNEncoder, self).__init__()

        self.embedding = nn.Embedding(num_characters, dim)
        # TODO
        self.rnn = nn.GRU()

    def forward(self, input):
        # TODO
        return input


class RNNDecoder(nn.Module):
    def __init__(self, num_characters, dim):
        super(RNNDecoder, self).__init__()

        self.embedding = nn.Embedding(num_characters, dim)
        self.rnn_cell = nn.GRUCell(dim, dim)

    def forward(self, input, target):
        # TODO
        return input

    def generate(self, input):
        # TODO
        return input


class Model(nn.Module):
    def __init__(self, num_characters, dim=256):
        super(Model, self).__init__()

        self.encoder = RNNEncoder(num_characters, dim)
        self.decoder = RNNDecoder(num_characters, dim)

    def cost(self, input, target):
        encoded = self.encoder(input)
        # TODO
        return encoded
