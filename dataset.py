import string
from itertools import chain

punctuation = " -.''1234567890&$#\\/*"
characters = string.ascii_lowercase + punctuation + "N" + "<" + ">" + "\n"
vocabulary = {c: ind for ind, c in enumerate(characters)}
rev_vocabulary = {ind: c for ind, c in enumerate(characters)}

with open("data/penn/train.txt", "r") as f:
    train_data = list(chain(*[[vocabulary[c] for c in seq] for seq in f]))

with open("data/penn/valid.txt", "r") as f:
    valid_data = list(chain(*[[vocabulary[c] for c in seq] for seq in f]))

with open("data/penn/test.txt", "r") as f:
    test_data = list(chain(*[[vocabulary[c] for c in seq] for seq in f]))
