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



def RNNls_bilstm(shape = None, params = None, penalty = 0.0005):
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

def RNNls_gru(shape = None, params = None, penalty = 0.0005):
    digit_input = Input(shape = shape)
    X = Embedding(input_dim=4,output_dim=16)(digit_input)
    X = tf.reshape(X,(-1,101,4))
    X = GRu(32, return_sequences=True)(X)
    X = Dropout(0.5)(X)
    X = Conv1D(32,32,activation='relu', kernel_regularizer=regularizers.l2(penalty),padding='same')(X)
    X = BatchNormalization()(X)
    X = LSTM(32)(X)
    X = Dropout(params['DROPOUT'])(X)
    output = Dense(1,activation='sigmoid')(X)
    model = Model(inputs = digit_input, outputs = output)
    print(model.summary())
    return model



def WSCNNwithNoisy(shape=None, params=None, penalty=0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),
                     input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), kernel_regularizer=regularizers.l2(penalty)))
    model.add(ANDNoisy(a=7.5))

    print(model.summary())
    return model


#
def WSCNNwithMax(shape=None, params=None, penalty=0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),
                     input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalMaxPooling2D())

    print(model.summary())
    return model


#
def WSCNNwithAve(shape=None, params=None, penalty=0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),
                     input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalAveragePooling2D())

    print(model.summary())
    return model


######################################
def WSCNNLSTMwithNoisy(shape=None, params=None, penalty=0.0005):
    digit_input = Input(shape=shape)
    X = Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),
               input_shape=shape)(digit_input)
    X = MaxPooling2D((1, 8))(X)
    X = Dropout(0.2)(X)
    X = TimeDistributed(Bidirectional(LSTM(32)))(X)
    X = Dropout(params['DROPOUT'])(X)
    X = Conv1D(2, 1, activation='softmax', kernel_regularizer=regularizers.l2(penalty))(X)
    output = ANDNoisy(a=7.5)(X)

    model = Model(inputs=digit_input, outputs=output)
    print(model.summary())
    return model


def get_rnn_capnet(shape=None, params=None, penalty=0.0005):
    digit_input = Input(shape = shape)
    X = Embedding(input_dim=4,output_dim=16)(digit_input)
    X = tf.reshape(X,(-1,6,3200))
    X = SpatialDropout1D(params['DROPOUT'])(X)
    # x = Bidirectional(GRU(128, return_sequences=True))(X)
    X = BatchNormalization()(X)
    X = Capsule(10, 16, 3, True)(X)
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    # outputs = Dense(5, activation='sigmoid')(x)
    outputs = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=digit_input, outputs=outputs)
    print(model.summary())
    return model


def get_cnn_capnet(shape=None, params=None, penalty=0.0005):
    """
    Conv_CapNet
    :param n_capsule:
    :param n_routings:
    :param capsule_dim:
    :param dropout_rate:
    :return: Model
    """

    digit_input = Input(shape = shape)
    X = Embedding(input_dim=4,output_dim=4)(digit_input)
    X = tf.reshape(X,(-1,6,800))
    X = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=16, kernel_size=24, padding="same")(X)))
    X = Capsule(10, 16, 3, True)(X)

    X = Flatten()(X)
    # x = concatenate([x_3, x_4, x_5], axis=1)
    X = Dropout(params['DROPOUT'])(X)
    # outputs = Dense(5, activation='sigmoid')(x)
    outputs = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=digit_input, outputs=outputs)

    print(model.summary())

    return model
