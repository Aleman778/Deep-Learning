# https://github.com/spro/char-rnn.pytorch

import numpy as np
from numpy.random import multinomial
from collections import Counter
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

# Plot the change in perplexity for each epoch

def plot_perplexity(perplexity):
    plt.figure(figsize=[15,10])
    plt.plot(perplexity)
    plt.title("Changes in perplexity over time")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.show()

# Balancing the word occurences in the data.

def subsample_frequent_words(text):
    word_counts = dict(Counter(list(text)))
    sum_word_counts = sum(list(word_counts.values()))
    word_counts = {word: word_counts[word]/float(sum_word_counts) for word in word_counts}
    filtered_text = []
    for word in text:
        if random.random() < (1 + math.sqrt(word_counts[word]*1e3))*1e-3 / float(word_counts[word]):
            filtered_text.append(word)
    return filtered_text

def sample_negative(text, sample_size):
    sample_probability = {}
    word_counts = dict(Counter(list(text)))
    normalizing_factor = sum([v**0.75 for v in word_counts.values()])
    for word in word_counts:
        sample_probability[word] = word_counts[word]**0.75 / normalizing_factor
    words = np.array(list(word_counts.keys()))
    while True:
        word_list = []
        sampled_index = np.array(multinomial(sample_size, list(sample_probability.values())))
        for index, count in enumerate(sampled_index):
            for _ in range(count):
                 word_list.append(words[index])
        yield word_list

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

