"""
Use Siamese CNNs to embed MNIST digits into 2-dimensional space.
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K

import matplotlib
import matplotlib.pyplot as plt

from siamese_nets import mnist_base, text_cnn_base
from data.data_utils import prepare_text_data, load_mr, load_glove

def euclidean_distance(vects):
    """
    euclidean distance between x and y
    """
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss fn based on Hadsell-et-al.'06.
    Please mind the gap! (margin)
    """
    margin = 1
    hit = y_true * K.square(y_pred)
    miss = (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
    return K.mean(hit + miss)

def create_pairs(x, digit_indices, nb_class):
    """
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(nb_class)]) - 1
    for d in range(nb_class):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, nb_class)
            dn = (d + inc) % nb_class
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy with a fixed threshold on distances.
    This metric is arbitrary; use 2D visualization instead to better
    judge the qualitative performance.
    """
    return labels[predictions.ravel() < 0.5].mean()

def text():
    max_len = 20
    glove_dim = 50
    datadict = load_mr()
    embeddings_index = load_glove(glove_dim)
    X_train, y_train, X_test, y_test, labels_index = prepare_text_data(datadict,
                                                            embeddings_index,
                                                            max_len)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    nb_classes = len(labels_index)
    nb_epoch = 10
    text_indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
    tr_pairs, tr_y = create_pairs(X_train, text_indices, nb_classes)
    text_indices = [np.where(y_test == i)[0] for i in range(nb_classes)]
    te_pairs, te_y = create_pairs(X_test, text_indices, nb_classes)

    input_shape = (max_len, glove_dim)
    base_network = text_cnn_base(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)

    # cross fingers and train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=128,
          nb_epoch=nb_epoch)

    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    # plot in 2 dimensions
    pred = base_network.predict(X_train) # TEzT / TRAIN
    x = pred[:,0]
    y = pred[:,1]
    colors = ['red', 'green', 'blue', 'purple', 'cyan',
              'yellow', 'magenta', 'black', 'beige', 'darkorange']
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=list(y_train),
                cmap=matplotlib.colors.ListedColormap(colors[0:nb_classes]))
    plt.show()

def main():
    """
    load and prepare data, train and evaluate network
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_epoch = 1 # mnist takes a while, but good results can be seen after 1 epoch

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(X_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(X_test, digit_indices, 10)

    # network definition
    base_network = mnist_base(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    # cross fingers and train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=128,
          nb_epoch=nb_epoch)

    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    # plot in 2 dimensions
    pred = base_network.predict(X_test)
    x = pred[:,0]
    y = pred[:,1]
    colors = ['red', 'green', 'blue', 'purple', 'cyan',
              'yellow', 'magenta', 'black', 'beige', 'darkorange']
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=list(y_test),
                cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()

if __name__ == "__main__":
    #main()
    text()
