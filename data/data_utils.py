"""
module for loading datasets and dealing with word embeddings
"""

import os
import json
from random import shuffle
import numpy as np

from keras.preprocessing import text

DATAPATH = os.path.dirname(os.path.realpath(__file__))
GLOVE_DIR = '/Users/stefan/glove/'

def tokenize_sentence(sentence):
    """
    sentence input is string. returns list of tokens.
    """
    tokens = text.text_to_word_sequence(sentence, lower=True, split=' ')
    return tokens

def prepare_text_data(datadict, embeddings_index, max_len):
    """
    convert all sentences to matrices of size (max_len, embedding_dim).
    """
    labels = list(set(datadict.values()))
    label_id = 0
    labels_index = {}
    for label in labels:
        labels_index[label] = label_id
        label_id += 1
    datalist = datadict.items()
    shuffle(datalist) # shuffles in-place
    n = len(datalist)
    split = int(n / 2)
    train = datalist[0:split]
    test = datalist[split:]
    X_train = [tokenize_sentence(str(x)) for (x,y) in train]
    X_test = [tokenize_sentence(str(x)) for (x,y) in test]
    y_train = [labels_index[y] for (x,y) in train]
    y_test = [labels_index[y]  for (x,y) in test]

    # sentence 2 matrix
    nb_train_samples = len(X_train)
    nb_test_samples = len(X_test)
    steps = max_len
    input_dim = len(embeddings_index['the'])
    X_train_input = np.zeros((nb_train_samples, steps, input_dim))
    X_test_input = np.zeros((nb_test_samples, steps, input_dim))
    for i in range(nb_train_samples):
        tokens = pad_sentences([X_train[i]], max_len)[0]
        mat = sentence2matrix(embeddings_index, tokens[0:max_len])
        X_train_input[i,:,:] = mat.T # mat.T is (time, embedding_dim)
    for i in range(nb_test_samples):
        tokens = pad_sentences([X_test[i]], max_len)[0]
        mat = sentence2matrix(embeddings_index, tokens[0:max_len])
        X_test_input[i,:,:] = mat.T

    return X_train_input, y_train, X_test_input, y_test, labels_index

def load_data_from_json(filename):
    """
    read data from json file. returns dict {'query':'label'}
    currently used for bankcoach dataset
    """
    path_to_file = os.path.join(DATAPATH, 'datasets', filename)
    with open(path_to_file) as datafile:
        data = json.load(datafile)
    return data

def load_bankcoach():
    """
    read bankcoach dataset
    """
    return load_data_from_json('bankcoach/medium-8class.json')

def load_mr():
    """
    read data from mr (movie reviews) datasets.
    OUTPUT: dict of form {'sentence':'label'}
    label is one of 'pos' or 'neg'
    NOTE: rt-polarity folder must contain both rt-polarity.neg and .pos
    """
    neg_file = os.path.join(DATAPATH,
                            'mr/rt-polaritydata/rt-polarity.neg')
    pos_file = os.path.join(DATAPATH,
                            'mr/rt-polaritydata/rt-polarity.pos')
    data = dict()
    with open(neg_file) as negfile:
        lines = negfile.readlines()
    for line in lines:
        query = str(line.strip())
        data[query] = 'neg'
    with open(pos_file) as posfile:
        lines = posfile.readlines()
    for line in lines:
        query = str(line.strip())
        data[query] = 'pos'
    return data

def load_glove(dim):
    """
    loads glove vectors from memory. returns map: word -> embedding
    """
    assert dim in [50, 100, 200, 300]
    embeddings_index = {}
    glove_file = 'glove.6B.' + str(dim) + 'd.txt'
    f = open(os.path.join(GLOVE_DIR, glove_file))
    print 'loading glove embeddings'
    for line in f:
        values = line.split()
        word = values[0]
        if word.isalpha():
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def sentence2matrix(embedding_index, slist):
    """
    slist: list of tokens
    Returns a numpy matrix representation of a sentence. 
    """
    d = len(embedding_index['the']) # embedding dimension
    smatrix = np.zeros([d, len(slist)])
    i = 0
    for word in slist:
        if word in embedding_index:
            embedding = embedding_index[word]
            smatrix[:,i] = embedding
        elif word == 'P_UNK':
            embedding = np.zeros(d)
            smatrix[:,i] = embedding
        else:
            # random vector
            embedding = random_words_matrix(embedding_index, 1)[0]
            smatrix[:,i] = embedding[:,0]
        i += 1
    return smatrix

def pad_sentences(tokens_list, max_len):
    """
    Input: tokens_list - list of list of tokens.
            max_len - int, length to pad sentences to.
    Output: list of list of tokens. Padded words are 'P_UNK'
    """
    for i in range(len(tokens_list)):
        if len(tokens_list[i]) < max_len:
            n = len(tokens_list[i])
            pad = ['P_UNK'] * (max_len - n)
            padded = tokens_list[i] + pad
            tokens_list[i] = padded
    return tokens_list

def random_words_matrix(embedding_index, n):
    """
    Produces a matrix representation of a sentence composed of random words.
    """
    d = len(embedding_index['the'])
    N = len(embedding_index)
    word_list = embedding_index.keys()
    r_matrix = np.zeros([d, n])
    r_sentence = []
    for i in range(n):
        idx = np.random.randint(0, N)
        r_word = word_list[idx]
        r_sentence.append(r_word)
        r_matrix[:,i] = embedding_index[r_word]
    return r_matrix, r_sentence
