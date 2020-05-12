# Exercise 4 - Results

## Task 1
Training the plot for 2000 epochs results in the following change
preplexity:
![Perplexity1](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise4/part1_perplexity_plot.png)

## Task 2
Generating character sequences of length 100 by priming the model with
random character sequences of length 5:

With priming sequence `4gh a` we get:
```
4gh at a call'd I am then: the seems the gone.

ISABELLA:
Alas, her eyes so, let the blood over-tell,
Tha
```

With priming sequence `ea52l` we get:
```
ea52land the death.
One yourselve's love: I shall be said?

ROMEO:
Farewels and bellow it win and him;
Co
```

With priming sequence `s;l2.` we get:
```
s;l2.

Second May!
Then he seest burthen, and I shall shall stop
Should have no king and understand were
```

With priming sequence `abcde` we get:
```
abcdely labour to your our death;
For the kneet to dismerved me infection;
And, if the lands: I cannot st
```

With priming sequence `2l;1+` we get:
```
2l;1+inger to enfries.

SICINIUS:
A guiltly, man, I do to caster the glored:
Shall I shall have abstrik's
```

## Task 3
With priming sequence `The` we get:
```
The daughter I have have play,
And which he in my brokent well,--

FRIAR LAURENCE:
An if the better pit
```

With priming sequence `What is` we get:
```
What is a day's of bandstering gone
To a good we such not such with royalty,
we will he woo'd my father's g
```

With priming sequence `Shall I give` we get:
```
Shall I give the happy was formill
To seety thou and straight was a swear,
And like it you art thou awhore, that
```

With priming sequence `X087hNYB BHN BYFVuhsdbsee` we get:
```
X087hNYB BHN BYFVuhsdbsee
Desumnss to calamatricas the heave them:
So it is your shame of the speediches.

KING RICHARD III
```

## Task 4
We tried to modfiy the char-RNN to work with word embeddings, in particular we tested CBOW continuous bag of words but training this was not successful due to the complexity of the dataset (12 GB of memory was not enough to train the model). One possible method might be to use pretrained embeddings but we don't have time to test this at the moment. The code we tested can be found in `train_word.py`.

# char-rnn.pytorch

A PyTorch implementation of [char-rnn](https://github.com/karpathy/char-rnn) for character-level text generation. This is copied from [the Practical PyTorch series](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb).

## Training

Download [this Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (from the original char-rnn) as `shakespeare.txt`.  Or bring your own dataset &mdash; it should be a plain text file (preferably ASCII).

Run `train.py` with the dataset filename to train and save the network:

```
> python train.py shakespeare.txt

Training for 2000 epochs...
(... 10 minutes later ...)
Saved as shakespeare.pt
```
After training the model will be saved as `[filename].pt`.

### Training options

```
Usage: train.py [filename] [options]

Options:
--model            Whether to use LSTM or GRU units    gru
--n_epochs         Number of epochs to train           2000
--print_every      Log learning rate at this interval  100
--hidden_size      Hidden size of GRU                  50
--n_layers         Number of GRU layers                2
--learning_rate    Learning rate                       0.01
--chunk_len        Length of training chunks           200
--batch_size       Number of examples per batch        100
--cuda             Use CUDA
```

## Generation

Run `generate.py` with the saved model from training, and a "priming string" to start the text with.

```
> python generate.py shakespeare.pt --prime_str "Where"

Where, you, and if to our with his drid's
Weasteria nobrand this by then.

AUTENES:
It his zersit at he
```

### Generation options
```
Usage: generate.py [filename] [options]

Options:
-p, --prime_str      String to prime generation with
-l, --predict_len    Length of prediction
-t, --temperature    Temperature (higher is more chaotic)
--cuda               Use CUDA
```

