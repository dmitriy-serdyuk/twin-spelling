import string
import numpy
from itertools import chain
from torch.utils.data import Dataset

punctuation = " -.''1234567890&$#\\/*"
characters = string.ascii_lowercase + punctuation + "N" + "<" + ">" + "\n"
vocabulary = {c: ind for ind, c in enumerate(characters)}
rev_vocabulary = {ind: c for ind, c in enumerate(characters)}


class Penntree(Dataset):
    def __init__(self, length, subset):
        self.subset = subset
        self.length = length
        with open("data/penn/{}.txt".format(subset), "r") as f:
            self.data = list(chain(*[[vocabulary[c] for c in seq] for seq in f]))

    def __len__(self):
        return len(self.data) // self.length

    def __getitem__(self, i):
        return numpy.array(self.data[i * self.length:(i + 1) * self.length])


class SpellCheckData(Penntree):
    def __init__(self, length, subset, prob, seed=None):
        if seed is None:
            seed = 426

        self.rng = numpy.random.RandomState(seed)
        super(SpellCheckData, self).__init__(length, subset)

        self.prob = prob

    def __getitem__(self, i):
        target = super(SpellCheckData, self).__getitem__(i)
        input = target.copy()
        indeces = self.rng.binomial(1, self.prob, size=input.shape)
        substitute = self.rng.choice(list(rev_vocabulary.keys()), size=input.shape)
        input =  substitute * indeces + input * (1 - indeces)

        return input, target
