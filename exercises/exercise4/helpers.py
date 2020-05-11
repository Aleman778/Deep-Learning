# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch
import re

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Word2Vec tokenization

def tokenize_text(text):
    # Preprocess, lower-case, removing newlines, carrige returns and punctiation
    text.lower()
    text.replace('\n', ' ')
    text.replace('\r', '')
    text = re.sub("[^a-z ]+", '', text)
    words = [w for w in text.split() if w != '']

    print(words)


# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

