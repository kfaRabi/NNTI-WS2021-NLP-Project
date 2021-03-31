# Imports
from functools import partial
from math import sqrt
import numpy as np
import pandas as pd
import string
import timeit
import torch
from torch import nn
import torch.functional as F
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import statistics
import re
import warnings
from pathlib import Path
import random
import math

# custom imports
from early_stopping import EarlyStopping
import nn_common as nnc

def seed_all(random_state):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size = 300):
        super(Word2Vec, self).__init__()
        self.E = nn.Linear(vocab_size, embedding_size, bias = False)
        self.W = nn.Linear(embedding_size, vocab_size)


    def forward(self, one_hot):
        z_e = self.E(one_hot)
        z_w = self.W(z_e)

        return F.log_softmax(z_w, dim = 1)

class EncoderCNN(nn.Module):
    def __init__(self, Embedding, params):
        super(EncoderCNN, self).__init__()
        seed_all(60065)
        padding_idx = params["padding_idx"]
        hidden_wieghts_dim = params["hidden_wieghts_dim"]
        num_of_layers = params["num_of_layers"]
        num_of_classes = params["num_of_classes"]
        sent_len = params["sent_len"]
        self.sent_len = params["sent_len"]
        batch_size = params["batch_size"]
        drop_prob = params["drop_prob"]
        conv_params = params["conv_params"]
        self.number_of_fc = params["number_of_fc"]
 
        self.model_path = f"_nol_{num_of_layers}_hwd_{hidden_wieghts_dim}_sent_len_{sent_len}_dp_{drop_prob}"
 
        self.embedding = nn.Embedding(num_embeddings = Embedding.shape[0], embedding_dim = Embedding.shape[1], padding_idx = padding_idx, _weight = Embedding)
        self.embedding.weight.requires_grad = False
 
        self.lstm = nn.LSTM(Embedding.shape[1], hidden_wieghts_dim, num_of_layers, batch_first = True, bidirectional = True, dropout = drop_prob if num_of_layers > 1 else 0)
        self.dropout = nn.Dropout(drop_prob)
        self.conv1 = nn.Conv2d(in_channels = conv_params[0][0], out_channels = conv_params[0][1], kernel_size = conv_params[0][2], stride = conv_params[0][3], padding = conv_params[0][4])
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = conv_params[1][0], out_channels = conv_params[1][1], kernel_size = conv_params[1][2], stride = conv_params[1][3], padding = conv_params[1][4])
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = conv_params[2][0], out_channels = conv_params[2][1], kernel_size = conv_params[2][2], stride = conv_params[2][3], padding = conv_params[2][4])
 
        h, w = nnc.conv_out_shape(params, h = sent_len, w = hidden_wieghts_dim * 2)

        if self.number_of_fc == 2:
            self.fc = nn.Linear(h * w * conv_params[2][1], (h * w * conv_params[2][1]) // 2)
            self.fc2 = nn.Linear((h * w * conv_params[2][1]) // 2, num_of_classes)
        else:
            self.fc = nn.Linear(h * w * conv_params[2][1], num_of_classes)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, batch):
        embedded = self.embedding(batch)
        out, (_, _) = self.lstm(embedded) # hidden = [batch size, num layers * num directions, hidden_wieghts_dim]
        output = out.unsqueeze(1)
        output = self.conv1(output)
        output = F.relu(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = F.relu(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = F.relu(output)
        output = output.view(batch.shape[0], -1)
        output = self.fc(output)
        if self.number_of_fc == 2:
            output = F.relu(output)
            output = self.dropout(output)
            output = self.fc2(output)
        output = self.sigmoid(output)
        output = output.squeeze(0) if batch.shape[0] == 1 else output
        return output

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerCNN(nn.Module):
    def __init__(self, params):
        super(TransformerCNN, self).__init__()
        self.model_path = "transformer_cnn"

        conv_params = params["conv_params"]
        vocab_size = params["vocab_size"]
        embedding_size = params["embedding_size"]
        num_of_head = params["num_of_head"]
        num_of_hidden = params["num_of_hidden"]
        num_of_layers = params["num_of_layers"]
        num_of_classes = params["num_of_classes"]
        dropout = params["dropout"]
        sent_len = params["sent_len"]

        self.embedding_size = embedding_size

        self.positional_embedding = PositionalEmbedding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_of_head, num_of_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_of_layers)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)

        self.conv1 = nn.Conv2d(
            in_channels = conv_params[0][0],
            out_channels = conv_params[0][1],
            kernel_size = conv_params[0][2],
            stride = conv_params[0][3],
            padding = conv_params[0][4])
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(
            in_channels = conv_params[1][0],
            out_channels = conv_params[1][1],
            kernel_size = conv_params[1][2],
            stride = conv_params[1][3],
            padding = conv_params[1][4])
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(
            in_channels = conv_params[2][0],
            out_channels = conv_params[2][1],
            kernel_size = conv_params[2][2],
            stride = conv_params[2][3],
            padding = conv_params[2][4])

        h, w = nnc.conv_out_shape(params, h = sent_len, w = embedding_size)

        self.fc1 = nn.Linear(h * w * conv_params[2][1], num_of_hidden)
        self.fc2 = nn.Linear(num_of_hidden, num_of_classes)


    def forward(self, data):
        data = self.embedding(data) * math.sqrt(self.embedding_size)
        data = self.positional_embedding(data)
        output = self.transformer_encoder(data)
        output = output.unsqueeze(0)
        output = self.pool1(F.relu(self.conv1(output)))
        output = self.pool2(F.relu(self.conv2(output)))
        output = F.relu(self.conv3(output))
        output = output.view(-1)
        output = F.relu(self.fc1(output))
        output = torch.sigmoid(self.fc2(output))

        return output

class EmbeddingContext():
    def __init__(self, params = {}):
        self.dataf = pd.DataFrame()
        self.sentences = None
        self.V = None
        self.word_to_index = None
        self.ONE_HOT_VECTOR_SIZE = 0
        self.rel_freq = None
        self.is_dev = True
        self.train_loader = None
        self.params = params
        self.DEVICE = nnc.get_device()
        self.criterion = lambda x, y : None
        self.optimizer = lambda x : None

    def load_and_preprocess_data(self, data_path, stopwords_path, data_delimiter = ',', sampling_params = {}):
        dataf = pd.read_csv(data_path, delimiter = data_delimiter)
        if sampling_params:
            HATE = sampling_params["HATE"]
            NOT_HATE = sampling_params["NOT_HATE"]
            seed = sampling_params["seed"]
            print(f"\nNumber of rows by categories in the original dataset:\n{dataf.groupby([dataf.category]).size()}")
            df1 = dataf[dataf.hate == 1].groupby([dataf.category]).sample(HATE, random_state = seed)
            df2 = dataf[dataf.hate != 1].groupby([dataf.category]).sample(NOT_HATE, random_state = seed)
            print(f"\nNumber of hate speeches by categories after sampling:\n{df1.groupby([df1.category]).size()}")
            print(f"\nNumber of non hate speeches by categories after sampling:\n{df2.groupby([df2.category]).size()}")
            df = pd.concat([df1, df2]).reset_index(drop = True).sample(frac = 1., random_state = seed)
            self.dataf = df.rename(columns = {"sentence": "text"})

            print(f"\nNumber of combined sampled texts by categories:\n{self.dataf.groupby([self.dataf.category]).size()}")

        self.sentences = [text.split() for text in list(self.dataf.text)]
        print(f'\nExample sentence before preprocessing:\n{self.sentences[4]}\n\n')
        self.sentences = nnc.preprocess(self.sentences, nnc.load_stopwords(stopwords_path))
        print(f'Same sentence after preprocessing: {self.sentences[4]}')
        self.dataf["preprocessed_sen_len"] = [len(text) for text in self.sentences]
        nnc.summary_stat(self.dataf, self.sentences)


    def build_vocab(self, is_dev):
        self.is_dev = is_dev
        self.V = nnc.build_vocab(self.sentences[:100] if self.is_dev else self.sentences)
        self.ONE_HOT_VECTOR_SIZE = len(self.V)
        print(f"Vocabularies length: {self.ONE_HOT_VECTOR_SIZE}")

    def convert_word_to_index(self):
        self.word_to_index = nnc.convert_word_to_index(self.V)

    def calculate_relative_frequency(self):
        self.rel_freq = nnc.calc_rel_freq(self.V, self.sentences[:100] if self.is_dev else self.sentences, self.ONE_HOT_VECTOR_SIZE, self.word_to_index)

    def prepare_dataloader(self, shuffle = True):
        print("Preparing data loader")
        start = timeit.default_timer()

        batch_size = self.params["batch_size"]
        window_size = self.params["window_size"]

        sens = self.sentences[:100] if self.is_dev else self.sentences
        data = []
        sentence_counter = 0
        for sentence in sens:
            for center_idx, context_idx in nnc.get_target_context(sentence, self.word_to_index, self.rel_freq, window_size = window_size):
                data.append([center_idx, context_idx])
        sentence_counter += 1
        if sentence_counter % 500 == 0:
            print("Processed " + str(sentence_counter) + " sentences")
        train_len = len(data)
        self.train_loader = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size = batch_size)

        stop = timeit.default_timer()
        print("Data loader is ready.\nFound "+ str(train_len) +" pairs.\nPreparing Data Loader Took " + str(round((stop - start), 2)) + " Seconds\n\n")


    def train(self, model, print_loss = True, print_step = 500):
        print("Training started")
        start = timeit.default_timer()

        patience = self.params["patience"]
        dataset_name = self.params["dataset_name"]
        batch_size = self.params["batch_size"]
        window_size = self.params["window_size"]
        learning_rate = self.params["learning_rate"]
        epochs = self.params["epochs"]

        Path(type(model).__name__).mkdir(parents = True, exist_ok = True)
        model_path = str(type(model).__name__) + "/" + dataset_name + "_w_" + str(window_size) + "_bs_" +str(batch_size) + "_lr_" + str(learning_rate) + "_" + str(start) +".pt"
        
        early_stopping = EarlyStopping(patience = patience, verbose = True, path = model_path, delta = 1e-3)
        model = model.to(self.DEVICE)
        losses = []
        for iter in range(epochs):
            epoc_loss = 0
            batch_processed = 0
            model.train()
            for X_orig, Y_orig in self.train_loader:
                X = nnc.batch_word_to_one_hot(X_orig, self.ONE_HOT_VECTOR_SIZE).to(self.DEVICE)
                Y = Y_orig.to(self.DEVICE)

                Y_pred = model(X)

                loss = self.criterion(Y_pred, Y)
                epoc_loss += loss.item()
                batch_processed += 1

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if print_loss and (batch_processed % print_step == 0):
                    print(".", end = "")

            losses.append(epoc_loss / batch_processed)
            if print_loss:
                print("\nAvg. loss at epoch " + str(iter + 1) + ": " + str(epoc_loss / batch_processed))

            early_stopping(epoc_loss / batch_processed, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        stop = timeit.default_timer()
        print("Training finished.\nTraining Took " + str(round((stop - start) / 60, 2)) + " Minutes\n")

        print("Plotting Embedding Loss...")
        nnc.plot_embedding_losses(losses)

        with open(f'{dataset_name}_em_learning_rate_{learning_rate}_bs_{batch_size}_ws_{window_size}.txt', 'w') as f:
            print(' '.join([str(ls) for ls in losses]), file=f)


class DataContext():
    def __init__(self):
        self.dataf = None
        self.sentences = []
        self.labels = []
        self.V = []
        self.ONE_HOT_VECTOR_SIZE = 0
        self.word_to_index = {}

    def load_and_preprocess_data(self, data_path, stopwords_path, data_delimiter = '\t', seed = 60065):
        self.dataf = pd.read_csv(data_path, delimiter = data_delimiter)
        self.sentences = [text.split() for text in list(self.dataf.text)]
        self.sentences = nnc.preprocess(self.sentences, nnc.load_stopwords(stopwords_path))
        self.dataf["preprocessed_sen_len"] = [len(text) for text in self.sentences]
        nnc.summary_stat(self.dataf, self.sentences)

    def create_target(self, label_colum_name, positive_value):
        self.dataf["target"] = np.where(self.dataf[label_colum_name] == positive_value, 1, 0)
        self.labels = list(self.dataf.target)

    def build_vocab(self):
        self.V = nnc.build_vocab(self.sentences)
        self.ONE_HOT_VECTOR_SIZE = len(self.V)
        print(f"Vocabularies length: {self.ONE_HOT_VECTOR_SIZE}")

    def convert_word_to_index(self):
        self.word_to_index = nnc.convert_word_to_index(self.V)



class TrainingContext():
    def __init__(self, params : dict):
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.params = params
        self.DEVICE = nnc.get_device(log = True)
        self.criterion = lambda x, y : None
        self.optimizer = lambda x : None
        self.scheduler = lambda x : None

    def prepare_dataloaders(self, data_context, train_percentage = 80, shuffle = True, drop_last = False):
        print("Preparing data loader")
        start = timeit.default_timer()

        padding_idx = self.params["padding_idx"]
        sent_len = self.params["sent_len"]
        batch_size = self.params["batch_size"]

        data = []

        for text, label in zip(data_context.sentences, data_context.labels):
            text_to_indices = [data_context.word_to_index[word] for word in text]
            word_cnt = len(text_to_indices)

            if word_cnt > sent_len:
                text_to_indices = text_to_indices[:sent_len]
            elif word_cnt < sent_len:
                text_to_indices = text_to_indices + ([padding_idx] * (sent_len - word_cnt))
            if word_cnt == 0:
                word_cnt = 1
            data.append([torch.Tensor(text_to_indices), np.float32(label)])

        train_len = int((len(data) * train_percentage) / 100)
        test_len = int((len(data) - train_len) / 2)
        valid_len = len(data) - train_len - test_len

        print(f'Size of Training Set :{train_len}, Validation Set: {valid_len}, and Test Set: {test_len}')

        train_data, validation_data, test_data = torch.utils.data.random_split(data, [train_len, valid_len, test_len])
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle = shuffle, batch_size = batch_size, drop_last = drop_last)
        self.valid_loader = torch.utils.data.DataLoader(validation_data, shuffle = shuffle, batch_size = batch_size, drop_last = drop_last)
        self.test_loader = torch.utils.data.DataLoader(test_data, shuffle = shuffle, batch_size = batch_size, drop_last = drop_last)

        stop = timeit.default_timer()
        print("Data loader is ready.\nProcessed "+ str(len(data_context.labels)) +" sentences.\nPreparing Data Loader Took " + str(round((stop - start), 2)) + " Seconds\n\n")

    def train(self, model, print_loss = True):
        print("Training started")
        start = timeit.default_timer()

        patience = self.params["patience"]
        batch_size = self.params["batch_size"]
        dataset_name = self.params["dataset_name"]
        weight_decay = self.params["weight_decay"]
        epochs = self.params["epochs"]
        learning_rate = self.params["learning_rate"]
        milestones = self.params["milestones"]
        gamma = self.params["gamma"]

        step_size = '-'.join([str(m) for m in milestones])
        DEVICE = self.DEVICE
        print_step = (len(self.train_loader) // 20) + 1

        Path(type(model).__name__).mkdir(parents = True, exist_ok = True)

        path = f"{type(model).__name__}/{dataset_name}_{model.model_path}_bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}_ss_{step_size}_gm_{gamma}_{start}"
        early_stopping = EarlyStopping(patience = patience, verbose=True, path = path + ".pt", delta = 0.00001)

        model = model.to(DEVICE)

        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        for iter in range(epochs):
            epoc_loss = 0
            batch_processed = 0
            correct = 0
            total = 0
            model.train()
            for X_orig, Y_orig in self.train_loader:
                X = X_orig.to(torch.long).to(DEVICE)
                Y = Y_orig.to(torch.float32).to(DEVICE)

                Y_pred = None
                if batch_size > 1:
                    Y_pred = model(X).squeeze()
                else: Y_pred = model(X)

                loss = self.criterion(Y_pred, Y)
                epoc_loss += loss.item()
                batch_processed += 1

                total += 1 if len(Y_pred.size()) == 0 else Y_pred.shape[0]
                correct += torch.sum(torch.round(Y_pred) == Y).item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if print_loss and (batch_processed % print_step == 0):
                    print(".", end = "")

            if print_loss:
                lr = f"{self.scheduler.get_last_lr()[0]:.10f}"
                print(f"\nAt epoch {iter}:\n\t\tLearning Rate: {str(lr).strip('0')}\n\t\tAvg. training loss: {(epoc_loss / batch_processed):.2f}\t\tAvg. training accuracy: {(100. * (correct/total)):.2f}")

            valid_loss, valid_acc = self.validate(model)
            train_losses.append(epoc_loss / batch_processed)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            train_accs.append(100. * (correct/total))

            self.scheduler.step()

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        stop = timeit.default_timer()
        print("\nTraining finished.\nTraining Took " + str(round((stop - start) / 60, 2)) + " Minutes\nRestoring model to the checkpoint\n\n")

        nnc.load_model(model, path + ".pt") # restore the best model
        nnc.plot_losses(train_losses, train_accs, valid_losses, valid_accs)
        nnc.save_training_info(train_losses, train_accs, valid_losses, valid_accs, model, path + ".txt")

    def validate(self, model):
        DEVICE = self.DEVICE

        model.eval()

        batch_size = self.params["batch_size"]

        epoc_loss = 0.
        batches = 0.
        correct = 0.
        total = 0.

        for X_orig, Y_orig in self.valid_loader:
            X = X_orig.to(torch.long).to(DEVICE)
            Y = Y_orig.to(torch.float32).to(DEVICE)

            Y_pred = None
            if batch_size > 1:
                Y_pred = model(X).squeeze()
            else: Y_pred = model(X)

            total += 1 if len(Y_pred.size()) == 0 else Y_pred.shape[0]

            correct += torch.sum(torch.round(Y_pred) == Y).item()

            epoc_loss += self.criterion(Y_pred, Y).item()
            batches += 1.
        print(f"\t\tAvg. validation loss: {(epoc_loss / batches):.2f}\t\tAvg. validation accuracy: {(100. * (correct/total)):.2f}\n\t\t", end = '')
        return (epoc_loss / batches, 100. * (correct / total))

    def test(self, model):
        DEVICE = self.DEVICE

        batch_size = self.params["batch_size"]

        model.eval()
        total = 0.
        correct = 0.

        for X_orig, Y_orig in self.test_loader:
            X = X_orig.to(torch.long).to(DEVICE)
            Y = Y_orig.to(torch.float32).to(DEVICE)
            
            Y_pred = None
            if batch_size > 1:
                Y_pred = model(X).squeeze()
            else: Y_pred = model(X)

            total += 1 if len(Y_pred.size()) == 0 else Y_pred.shape[0]
            correct += torch.sum(torch.round(Y_pred) == Y).item()
        print(f'Test Accuracy: {(100. * (correct / total)):.2f} %')
