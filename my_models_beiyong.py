#/usr/bin/python

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Flatten, Input, concatenate, BatchNormalization, \
    TimeDistributed, Bidirectional, LSTM, Reshape, Activation, Convolution1D
from keras import regularizers
import tensorflow as tf
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, Input, concatenate, \
    BatchNormalization, Activation, add, Bidirectional, LSTM, GRU, AveragePooling1D, ZeroPadding1D, recurrent
from keras import regularizers
from keras.layers.embeddings import Embedding

############################


def RNNls(shape = None, params = None, penalty = 0.0005):
    digit_input = Input(shape = shape)
    X = Embedding(input_dim=4,output_dim=4)(digit_input)
    X = tf.reshape(X,(-1,9,1920))
    X = recurrent.LSTM(32, return_sequences=True)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = LSTM(32)(X)
    X = Dropout(params['DROPOUT'])(X)
    X = Dense(32,activation='relu')(X)
    output = Dense(1,activation='sigmoid')(X)
    model = Model(inputs = digit_input, outputs = output)
    print(model.summary())
    return model



