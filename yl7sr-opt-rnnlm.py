# Load packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.manual_seed(1)


from torch.autograd import Variable
import os
import numpy as np

MODEL_PATH = "/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/weights/model_opt.h5"
TRN_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/results/trn_log_ws_opt.txt'
DEV_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/results/dev_log_ws_opt.txt'
EPOCH_SIZE = 30

# input_dim = 5
# hidden_dim = 10
# n_layers = 1

# batch_size = 1
# seq_len = 3

# Google Colab stuff
from google.colab import drive
drive.mount('/content/drive')


def load_data(path):
    # Load data
    trn_texts = open(path+"trn-wiki.txt").read().strip().split("\n")
    dev_texts = open(path+"dev-wiki.txt").read().strip().split("\n")
    

    #tokenize data
    trn_tokens = []
    for trn_text in trn_texts:
      tokens = trn_text.split(" ")
      if len(tokens) > 0: trn_tokens.append(tokens)
    dev_tokens = []
    for dev_text in dev_texts:
      tokens = dev_text.split(" ")
      if len(tokens) > 0: dev_tokens.append(tokens)
    # trn_tokens = [trn_text.split(" ") for trn_text in trn_texts]
    # dev_tokens = [dev_text.split(" ") for dev_text in dev_texts]

    print("Training data ...")
    print("%d" % (len(trn_tokens)))
    print("Development data ...")
    print("%d" % (len(dev_tokens)))
    
    #get max feature size
    max_length = max(max([len(tokens) for tokens in trn_tokens]), max([len(tokens) for tokens in dev_tokens]))
    
    return trn_tokens, dev_tokens, max_length


def buildVocabulary(trn_tokens):
    words = set()
    vocab = {}
    index = 0
    for tokens in trn_tokens:
        for token in tokens:
            if token not in vocab:
                vocab[token] = index
                index += 1
    return vocab


# def index_tokens(tokens, vocabulary):
#     tokens_index_list = []
#     for tokens in tokens_list:
#         tokens_index = [vocabulary[token] for token in tokens]
#         tokens_index_list.append(tokens_index)
#     return tokens_index_list    

def prepare_sequence(tokens, vocabulary):
#     tokens_index_list = []
#     for tokens in tokens_list:
#         tokens_index = [vocabulary[token] for token in tokens]
#         tokens_index_list.append(tokens_index)
    tokens_index = [vocabulary[token] for token in tokens]
    return tokens_index    

    
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 32, batch_size = 1, hidden_dim=32, num_layers=1):
        super(LSTMLanguageModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Initialize a default nn.Embedding as word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_layers)
        
        # Linear mid layer to transfer hidden to probablities
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros((1, 1, self.hidden_dim), device=device),
                torch.zeros((1, 1, self.hidden_dim), device=device))
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), self.batch_size, -1))
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), self.batch_size, -1), self.hidden)
        word_space = self.linear(lstm_out.view(-1, self.embedding_dim))
        output = F.log_softmax(word_space, dim=1)
        return output


def predict_save(model, inputs, path):
    #load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMLanguageModel(len(vocabulary))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    loss_function = nn.NLLLoss()
    if torch.cuda.is_available():
        model.cuda()
    
    
    #predict with inputs, get logp(w_i,i)
    log_ws = []
    for sentence in inputs:
        sentence_in = torch.tensor(prepare_sequence(sentence, vocabulary), device = device)
        logps = model1(sentence_in[:-1])
        logpw = []
        for i, logp in enumerate(logps):
            logpw.append(str(logp[sentence_in[i+1]].item()))
        log_ws.append(logpw)
    
    #write the results to a file
    f = open(path,'w')
    for log_w in log_ws:
        s = str(" ").join(log_w)
        f.write(s+'\n')
    
    
# main

# load data
trn_tokens, dev_tokens, max_length = load_data("/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling"+"/data-for-lang-modeling/")
vocabulary = buildVocabulary(trn_tokens)


# train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMLanguageModel(len(vocabulary))
# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0)


if torch.cuda.is_available():
  model.cuda()
    
i = 1
epoch_index = 1
best_loss = 100

losses = []
trn = trn_tokens
for epoch in range(EPOCH_SIZE):
    avg_loss = 0.0
    for sentence in trn:
        # clear out gradients
        model.zero_grad()

        model.hidden = model.init_hidden()
        
        # get inputs 
        sentence_in = torch.tensor(prepare_sequence(sentence, vocabulary)[:-1], device = device)
        targets = torch.tensor(prepare_sequence(sentence, vocabulary)[1:], device = device)
        
        # run for word scores
        word_scores = model(sentence_in)
        
        # compute loss, gradients, update parameters
        loss = loss_function(word_scores, targets)
        avg_loss += loss.item()
        if i%len(trn)==0: 
          print("Epoch:" ,epoch_index , "Loss: ",avg_loss/len(trn))
          i -= len(trn)
          epoch_index += 1
        loss.backward()
        optimizer.step()
        i += 1
    avg_loss /= len(trn)
    if avg_loss < best_loss:
        torch.save(model.state_dict(), MODEL_PATH)
        best_loss = avg_loss

print("Best Loss is: ", best_loss)

model1 = LSTMLanguageModel(len(vocabulary))
model1.load_state_dict(torch.load(MODEL_PATH, map_location=device))
loss_function = nn.NLLLoss()
if torch.cuda.is_available():
  model1.cuda()

predict_save(model1, trn_tokens, TRN_RESULT_PATH)
predict_save(model1, dev_tokens, DEV_RESULT_PATH)