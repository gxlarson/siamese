"""
module for various base networks
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Convolution1D, MaxPooling1D

def mnist_base(input_shape):
    """
    for use on mnist digits
    """
    nb_filters = 16
    kernel_size = (3,3)
    pool_size = (2,2)

    seq = Sequential()
    seq.add(Convolution2D(nb_filters,
                          kernel_size[0],
                          kernel_size[1],
                          border_mode='valid',
                          input_shape=input_shape))
    seq.add(Activation('relu'))
    seq.add(Convolution2D(nb_filters,
                          kernel_size[0],
                          kernel_size[1]))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=pool_size))
    seq.add(Flatten())
    seq.add(Dense(64))
    seq.add(Activation('relu'))
    seq.add(Dense(32)) # output layer is 2 so that we can visualize in 2-D
    seq.add(Activation('tanh'))
    return seq

def text_cnn_base(input_shape):
    """
    for use on text
    """
    nb_filter = 16
    filter_length = 2
    subsample_length = 1
    pool_length = 3

    seq = Sequential()
    seq.add(Convolution1D(nb_filter=nb_filter,
                          filter_length=filter_length,
                          activation='relu',
                          subsample_length=subsample_length,
                          border_mode='valid',
                          input_shape=input_shape))
    seq.add(MaxPooling1D(pool_length=pool_length))
    seq.add(Flatten())
    seq.add(Dense(64))
    seq.add(Activation('relu'))
    seq.add(Dense(2))
    seq.add(Activation('linear'))
    return seq

def text_lstm_base(input_shape):
    """
    for use on text
    """
    model = Sequential()
    model.add(Convolution1D(nb_filter=20,
                            filter_length=2,
                            activation='relu',
                            subsample_length=1,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(MaxPooling1D(pool_length=3))
    model.add(LSTM(32))
    model.add(Dense(16))
    model.add(Activation('linear'))
    return model
