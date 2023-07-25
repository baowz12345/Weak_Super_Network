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

#
# def RNNls(shape = None, params = None, penalty = 0.0005):
#     digit_input = Input(shape = shape)
#     # X = Embedding(input_dim=4,output_dim=4)(digit_input)
#     X = tf.reshape(digit_input,(-1,6,200))
#     X = recurrent.LSTM(32, return_sequences=True)(X)
#     X = Dropout(0.2)(X)
#     X = BatchNormalization()(X)
#     X = LSTM(32)(X)
#     X = Dropout(params['DROPOUT'])(X)
#     output = Dense(1,activation='sigmoid')(X)
#     model = Model(inputs = digit_input, outputs = output)
#     print(model.summary())
#     return model
from Noisy_and import ANDNoisy


# def RNNls(shape = None, params = None, penalty = 0.0005):
#     digit_input = Input(shape = shape)
#     X = Embedding(input_dim=4,output_dim=16)(digit_input)
#     X = tf.reshape(X,(-1,6,3200))
#     X = GRU(32, return_sequences=True)(X)
#     X = Dropout(0.5)(X)
#     X = BatchNormalization()(X)
#     X = GRU(32)(X)
#     X = Dropout(params['DROPOUT'])(X)
#     output = Dense(1,activation='sigmoid')(X)
#     model = Model(inputs = digit_input, outputs = output)
#     print(model.summary())
#     return model
#
def DeepBind(shape=None, params=None, penalty=0.0005):
    model = Sequential()
    model.add(Conv1D(16, 24, padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),
                     input_shape=shape))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    return model

def DeepSite(shape = None , params = None ,penalty = 0.0005):
    digit_input = Input(shape = shape)
    X = Bidirectional(LSTM(256, return_sequences=True))(digit_input)
    X = Conv1D(300,24,activation='relu', kernel_regularizer=regularizers.l2(penalty),padding='same')(X)
    X = Dropout(0.1)(X)
    X = MaxPooling1D(101)(X)
    X = Dropout(0.1)(X)
    X = Dense(32,activation='sigmoid')(X)
    #X = Dense(1)(X)
    output = Dense(1,activation='sigmoid')(X)

    model = Model(inputs = digit_input, outputs = output)
    print(model.summary())
    return model


def DanQ(shape=None, params=None, penalty=0.0005):
    model = Sequential()
    model.add(Conv1D(16, 24, padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),
                     input_shape=shape))
    model.add(MaxPooling1D(8))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    return model




######################################
def WSCNNLSTMwithNoisy(shape = None, params = None, penalty = 0.0005):

    digit_input = Input(shape=shape)
    X = Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape)(digit_input)
    X = MaxPooling2D((1, 8))(X)
    X = Dropout(0.2)(X)
    X = TimeDistributed(Bidirectional(LSTM(32)))(X)
    X = Dropout(params['DROPOUT'])(X)
    X = Conv1D(2, 1, activation='sigmoid', kernel_regularizer=regularizers.l2(penalty))(X)
    output = ANDNoisy(a=7.5)(X)

    model = Model(inputs = digit_input, outputs=output)
    print (model.summary())
    return model
###############################

def RNNls(shape = None, params = None, penalty = 0.0005):
    digit_input = Input(shape = shape)
    X = Embedding(input_dim=4,output_dim=16)(digit_input)
    X = tf.reshape(X,(-1,101,4))
    X = Bidirectional(LSTM(32, return_sequences=True))(X)
    X = Dropout(0.5)(X)
    X = Conv1D(32,32,activation='relu', kernel_regularizer=regularizers.l2(penalty),padding='same')(X)
    X = BatchNormalization()(X)
    X = LSTM(32)(X)
    X = Dropout(params['DROPOUT'])(X)
    output = Dense(1,activation='sigmoid')(X)
    model = Model(inputs = digit_input, outputs = output)
    print(model.summary())
    return model

# def F-cnn(shape = None, params = None, penalty = 0.0005):

