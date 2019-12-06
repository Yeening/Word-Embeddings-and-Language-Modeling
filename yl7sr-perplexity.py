# Load packages
import math
import os
import numpy as np

# TRN_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/trn_log_ws.txt'
# DEV_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/dev_log_ws.txt'

#for embedded
# TRN_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/results/stacked_trn_log_ws.txt'
# DEV_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/results/stacked_dev_log_ws.txt'

#for optimization
# TRN_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/results/trn_log_ws_opt.txt'
# DEV_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/results/dev_log_ws_opt.txt'

#for model size
TRN_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/results/trn_log_ws_model64.txt'
DEV_RESULT_PATH = '/content/drive/My Drive/Colab Notebooks/NLP/Word-Embeddings-and-Language-Modeling/results/dev_log_ws_model64.txt'


# Google Colab stuff
from google.colab import drive
drive.mount('/content/drive')


def perplexity(logp_ws):
    # w1 = [w1_1, w1_2, ..., w1_N1, <stop>]; w2 = [w2_1, w2_2, ..., w2_N2, <stop>]
    # logp_w1 = [logp_w1_1, logp_w1_2, ..., logp_w1_N1, logp_<stop>]
    avg = 0.0
    total_N = 0
    for logp_w in logp_ws:
        logp_w = np.array(logp_w,dtype=float)
        N = logp_w.shape[0]
        avg += np.sum(logp_w)
        total_N += N
    avg /= total_N
    res = math.exp(avg*-1)
    return res

def load_data(path):
    data = open(path,'r').read().split('\n')[:-1]
    logp_ws = [line.split(' ') for line in data]
    return logp_ws


# main
logp_ws_trn = load_data(TRN_RESULT_PATH)
logp_ws_dev = load_data(DEV_RESULT_PATH)

trn_per = perplexity(logp_ws_trn)
dev_per = perplexity(logp_ws_dev)

print("Perplexity on training set: ", trn_per, '\n')
print("Perplexity on development set: ", dev_per)