"""
module for various base networks
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten

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
    seq.add(Dense(2)) # output layer is 2 so that we can visualize in 2-D
    seq.add(Activation('linear'))
    return seq
