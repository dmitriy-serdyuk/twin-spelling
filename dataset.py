import string
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
        return self.data[i * self.length:(i + 1) * self.length]
