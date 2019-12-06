# Load packages
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer
from sklearn.linear_model import LogisticRegression
import os
import numpy as np


def load_word_embedding(Path):
    file = open(Path, 'r')
    embedding = {}
    for line in file:
        line = line.split()
        embedding[line[0]] = np.array(line[1:], dtype = float)
    return embedding

def construct_sentence_rep(data, word_embedding):
    sentence_reps = []
    for tokens in data:
        current_res = np.zeros((50,), dtype=float)
        for token in tokens:
            if token in word_embedding:
                current_res += word_embedding[token]
            else:
                current_res += word_embedding['unk']
        if len(tokens)==0: 
            current_res = word_embedding['unk']
        else:
            current_res /= len(tokens)
        sentence_reps.append(current_res)
    return np.array(sentence_reps, dtype=float)

def logestic_regression(trn_x, trn_labels, dev_x, dev_labels):
    # Define a LR classifier
    classifier = LogisticRegression(random_state=0,tol=0.0001,solver='liblinear',
                                    multi_class='auto',C=0.1, penalty='l1') #0.647
#     classifier = LogisticRegression(random_state=0,tol=0.0002,solver='liblinear',
#                                     multi_class='auto',C=0.1, penalty='l1') 
#     classifier = LogisticRegression(verbose=1, C=5, penalty='l2')
    classifier.fit(trn_x, trn_labels)

    # Measure the performance on dev data
    # print("Training accuracy = %f" % classifier.score(trn_x, trn_labels))
    print("Dev accuracy = ", classifier.score(dev_x, dev_labels))
    
    
# Load data
trn_texts = open(os.getcwd()+"/data-for-text-classification/trn-reviews.txt").read().strip().split("\n")
trn_labels = open(os.getcwd()+"/data-for-text-classification/trn-labels.txt").read().strip().split("\n")
print("Training data ...")
print("%d, %d" % (len(trn_texts), len(trn_labels)))

dev_texts = open(os.getcwd()+"/data-for-text-classification/dev-reviews.txt").read().strip().split("\n")
dev_labels = open(os.getcwd()+"/data-for-text-classification/dev-labels.txt").read().strip().split("\n")
print("Development data ...")
print("%d, %d" % (len(dev_texts), len(dev_labels)))


# lower and tokenize data
trn_tokens = [WordPunctTokenizer().tokenize(trn_text.lower()) for trn_text in trn_texts]
dev_tokens = [WordPunctTokenizer().tokenize(dev_text.lower()) for dev_text in dev_texts]


# load pre-trained 50D word embedding
word_embedding = load_word_embedding('glove.6B/glove.6B.50d.txt')


# build sentence representations
trn_representations = construct_sentence_rep(trn_tokens, word_embedding)
dev_representations = construct_sentence_rep(dev_tokens, word_embedding)


# Logistic Regression Text Classification using sentence representations


logestic_regression(trn_representations, trn_labels, dev_representations, dev_labels)


# use CountVectorizer to vectorize the text
vectorizer = CountVectorizer()
trn_x = vectorizer.fit_transform(trn_texts).toarray()
dev_x = vectorizer.transform(dev_texts).toarray()

# concatenate x with sentence representations
combine_trn_x = np.concatenate((trn_representations,trn_x), axis=1)
combine_dev_x = np.concatenate((dev_representations,dev_x), axis=1)


# Logistic Regression Text Classification using features combined sentence representations and vectorizer
logestic_regression(combine_trn_x, trn_labels, combine_dev_x, dev_labels)