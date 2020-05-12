#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

import matplotlib.pyplot as plt

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

text, file_len = read_file(args.filename)

def create_training_batches(context_tuple_list, batch_size):
    random.shuffle(context_tuple_list)
    batches, batch_target, batch_context = [], [], []
    for i in tqdm(range(len(context_tuple_list))):
        batch_target.append(word_to_index[context_tuple_list[i][0]])
        batch_context.append(word_to_index[context_tuple_list[i][1]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list) - 1:
            tensor_target = Variable(torch.from_numpy(np.array(batch_target)).long())
            tensor_context = Variable(torch.from_numpy(np.array(batch_context)).long())
            batches.append((tensor_target, tensor_context))
            tensor_target, tensor_context = [], []
    return batches


def train(batches):
    random.shuffle(batches);
    num_batches = int(args.chunk_len / args.batch_len)
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for (target, context) in batches[:num_batches]:
        output, hidden = decoder(context, hidden)
        loss += criterion(output.view(args.batch_size, -1), target)

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '_word.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

    # Preprocess, lower-case, removing newlines, carrige returns and punctiation
    text.lower()
    text.replace('\n', ' ')
    text.replace('\r', '')
    text = re.sub("[^a-z ]+", '', text)
    text = [w for w in text.split() if w != '']
    text = subsample_frequent_words(text)
    vocabulary = set(text)
    word_to_index = {w: idx for (idx, w) in enumerate(vocabulary)}
    index_to_word = {idx: w for (idx, w) in enumerate(vocabulary)}


# Preprocess, lower-case, removing newlines, carrige returns and non-words
text = text.lower()
text = text.replace('\n', ' ')
text = text.replace('\r', '')
text = re.sub("[^a-z ]+", '', text)
text = [w for w in text.split() if w != '']

# Creating the vocabulary
text = subsample_frequent_words(text)
vocab = set(text)
vocab_size = len(vocab)
word_to_index = {w: idx for (idx, w) in enumerate(vocab)}
index_to_word = {idx: w for (idx, w) in enumerate(vocab)}
print("Created vocabulary with {} entries.".format(vocab_size))

# Building continuous bag of words (CBOW)
context_tuple_list = []
#negative_samples = sample_negative(text, 8)
w = 4 # window size

print("Creating word embeddings...")
for i, word in enumerate(text):
    first_context_word_index = max(0,i-w)
    last_context_word_index = min(i+w, len(text))
    for j in range(first_context_word_index, last_context_word_index):
        if i!=j:
            context_tuple_list.append((word, text[j]))
print(context_tuple_list[:20])
print("There are {} pairs of target and context words".format(len(context_tuple_list)))

batches = create_training_batches(context_tuple_list, args.batch_size)
                                      


# Initialize models and start training
    
decoder = CharRNN(
    vocab_size,
    args.hidden_size,
    vocab_size,
    model=args.model,
    n_layers=args.n_layers,
)

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
perplexity = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(batches)
        loss_avg += loss
        perplexity.append(2**loss)

        
        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()

    print("Plotting perplexity over time...")
    plot_perplexity(perplexity)
    

except KeyboardInterrupt:
    print("Saving before quit...")
    save()
    
    print("Plotting perplexity over time...")
    plot_perplexity(perplexity)
    

