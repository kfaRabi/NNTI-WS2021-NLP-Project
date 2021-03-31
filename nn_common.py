from functools import partial
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import string
import timeit
import torch
from torch import nn
import torch.functional as F
import torch.nn.functional as F
import statistics
import re
import warnings
from math import floor
import seaborn as sns

# All the functions in this file have self-explanatory names.
# We have added comments where something is ambigous

def get_device(log = False):
    CUDA_ID = None
    DEVICE = None
    if torch.cuda.is_available():
        CUDA_ID = torch.cuda.current_device()
        DEVICE = torch.device('cuda') # pylint: disable=maybe-no-member, unused-variable
        if log:
            print("Running On GPU")
    else:
        DEVICE = "cpu"
        if log:
            print("No GPU :(")
    return DEVICE

def load_stopwords(filename):
    file = open(filename, "r", encoding="utf8")
    return [x[:-1] for x in file]

def remove_username(text):
    at ='@'
    text = ' '.join(list(map(lambda word : word if word[0] != at else ' ', text))).split()
    return text


def remove_punctuations(text):
    punc = string.punctuation + "…।"
    text = ''.join([ch if ch not in punc else ' ' for ch in ' '.join(text)]).split()
    return text

def remove_url(text):
    text = re.sub('http[s]?://\S+', '', ' '.join(text)) # pylint: disable = anomalous-backslash-in-string
    return text.split()

def remove_stopwords(stopwords, text):
    text = ' '.join([word.strip() if word.strip() not in stopwords  else ' ' for word in text]).split()
    return text

def remove_single_char_and_digit(text):
    single_let_n_sym = [c for c in string.ascii_letters + string.digits + '”सहगईइ॥√ﷺ》तजओप“मक¸‍✅उख–भॐर¶°़】•चए—©अऋब' + "‍‍৺ঁ‍্‍্যঃঅআইঈউঊঋঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরশষসক্ষড়ঢ়য়৷т’"]
    hidden_char = 8294
    text = ' '.join([' ' if (word in single_let_n_sym) or (len(word) == 1 and ord(word) == hidden_char) else word for word in text]).split()
    return text


def preprocess(texts, stopwords, log = True):
    if log:
        print("Preprocessing started")
    start = timeit.default_timer()

    remove_stopwords_callback = partial(remove_stopwords, stopwords)

    texts = list(map(remove_username, texts))
    texts = list(map(remove_url, texts))
    texts = list(map(remove_punctuations, texts))
    texts = list(map(remove_stopwords_callback, texts))
    texts = list(map(remove_single_char_and_digit, texts))
    texts = list(map(lambda text : [word.lower() for word in text], texts))

    stop = timeit.default_timer()
    if log:
        print("Preprocessing finished.\nPreprocessing Took " + str(stop - start) + " Seconds\n\n")

    return texts

def build_vocab(texts, padding = '___PAD___', unknown = '___UNK___'):
    return [padding] + [unknown] + list(set(word for text in texts for word in text))

def get_padding_index(V, PADDING):
  return V.index(PADDING)

def word_to_one_hot(word_idx, ONE_HOT_VECTOR_SIZE):
    v = [0] * ONE_HOT_VECTOR_SIZE
    v[word_idx] = 1
    return v

def convert_word_to_index(V): # assigns an index to each word of the vocabulary
    w2idx = {}
    for idx, word in enumerate(V):
        w2idx[word] = idx
    return w2idx

def batch_word_to_one_hot(batch, ONE_HOT_VECTOR_SIZE): # to produce batches of one hot
    one_hot_batch = []
    for idx_tensor in batch:
        idx = idx_tensor.item()
        x = word_to_one_hot(int(idx), ONE_HOT_VECTOR_SIZE)
        one_hot_batch.append(x)
    return torch.Tensor(one_hot_batch)

def summary_stat(dataf, sentences): # plots distributions and sumaary statistics
    dataf["preprocessed_sen_len"] = [len(text) for text in sentences]

    print(dataf.preprocessed_sen_len.describe())

    plt.hist(list(dataf.preprocessed_sen_len), bins= 100, density = True, cumulative = True, label = 'CDF: Preprocessed Sentence Lenght', histtype='step', alpha=0.55, color='purple')
    plt.show()
    sns.distplot(dataf.preprocessed_sen_len, hist = True, kde = True, 
             bins = int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = "Preprocessed Sentence Length")

    print("")

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename), strict=False)

# Calculates the number of times each word appears in all the sentences and finally devides by the total number of words
def calc_rel_freq(V, texts, ONE_HOT_VECTOR_SIZE, word_to_index):
    rel_freq = [0.] * ONE_HOT_VECTOR_SIZE
    total = 0.

    for word in V:
        cnt = 0.
        for text in texts:
            cnt += text.count(word)
        rel_freq[word_to_index[word]] = cnt
        total += cnt

    for word in V:
        rel_freq[word_to_index[word]] = rel_freq[word_to_index[word]] / total

    return rel_freq

def plot_embedding_losses(losses):
    plt.plot(losses)
    plt.xlabel("Number Of Epochs")
    plt.ylabel("Training Loss")
    plt.show()
    print("\n\n")

def sampling_prob(word_ind, rel_freq):
    rf = rel_freq[word_ind]
    return (sqrt(rf / 0.001) + 1.) * (0.001 / rf)

# creates center, context pair for the given window size consider the sampling probablity
def get_target_context(sentence, word_to_index, rel_freq, window_size = 2):
    sentence_to_indices = [word_to_index[word] for word in sentence]
    sentence_len = len(sentence_to_indices)
    for center_ind in range(sentence_len):
        for context_i in range(-window_size, window_size + 1):
            context_ind = center_ind + context_i
            if context_ind == center_ind \
            or context_ind < 0 \
            or context_ind >= sentence_len:
                continue
            if np.random.random() < sampling_prob(sentence_to_indices[context_ind], rel_freq):
                yield (sentence_to_indices[center_ind], sentence_to_indices[context_ind])

# Helper method to plot training and validation loss and accuracy
def plot_losses(train_losses, train_accs, valid_losses, valid_accs):
    checkpoint = valid_losses.index(min(valid_losses))
    xsteps = int(len(train_losses)/10)
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].plot(train_losses, label = "Training")
    axs[0].plot(valid_losses, label = "Validation")
    axs[0].axvline(x = checkpoint, label = "Early Stopping Check Point", c = "red", ls = "--")
    axs[0].set(xlabel='Number of epochs', ylabel='Loss', title = "Train/Validtion Loss Per Epoch")
    axs[1].plot(train_accs, label = "Training")
    axs[1].plot(valid_accs, label = "Validation")
    axs[1].axvline(x = checkpoint, label = "Early Stopping Check Point", c = "red", ls = "--")
    axs[1].set(xlabel='Number of epochs', ylabel='Accuracy', title = "Train/Validtion Accuracy Per Epoch")
    fig.tight_layout()
    plt.legend()
    plt.show()

# Helper method to store training losses and the model to a file
def save_training_info(train_losses, train_accs, valid_losses, valid_accs, model, path):
    with open(path, 'w') as f:
        print(' '.join([str(ls) for ls in train_losses]), file = f)
        print(' '.join([str(ls) for ls in train_accs]), file = f)
        print(' '.join([str(ls) for ls in valid_losses]), file = f)
        print(' '.join([str(ls) for ls in valid_accs]), file = f)
        print(model, file = f)
        print(path, file = f)
        print("\nTraining Session Info Stored")

# Given the initial input shape of the first convolution layer, and convolution parameters,
# it calculates the output shape of the final convolution layer.
# We use this method to calculate number of neurons of the first fully connected layer
def conv_out_shape(params, h, w):
    # For EncoderCNN, h: sent_len , w: hidden_weights_dim * (2 if bidirectional lstm else 1)
    # For TransformerCNN: h: sent_len, w: embedding_size
    conv_params = params["conv_params"]
    cnt = 0
    for conv in conv_params:
        h = int(floor(h - conv[2][0] + (2 * conv[4]) / conv[3])) + 1
        w = int(floor(w - conv[2][1] + (2 * conv[4]) / conv[3])) + 1
        cnt += 1
        if cnt != len(conv_params): # no max pool after the last conv layer, so it will not be devided by 2
            h = int(floor(h/2))
            w = int(floor(w/2))
        print(f"Output shape after conv{cnt}: {(h, w)}")
    return (h, w)