#/usr/bin/python

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Flatten, Input, concatenate, BatchNormalization, \
    TimeDistributed, Bidirectional, LSTM, Reshape, Activation, Embedding, recurrent
from keras import regularizers
from Noisy_and import ANDNoisy
import tensorflow as tf


# DeepBind model
def DeepBind(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv1D(16, 24, padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Dense(1, activation='sigmoid'))
    
    print (model.summary())
    return model



# DanQ model
def DanQ(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv1D(16, 24, padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),input_shape=shape))
    model.add(MaxPooling1D(8))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Dense(1, activation='sigmoid'))
    
    print (model.summary())
    return model
# 
def WSCNNwithNoisy(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), kernel_regularizer=regularizers.l2(penalty)))
    model.add(ANDNoisy(a=7.5))

    print (model.summary())
    return model

# 
def WSCNNwithMax(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalMaxPooling2D())
    
    print (model.summary())
    return model

# 
def WSCNNwithAve(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalAveragePooling2D())
    
    print (model.summary())
    return model


######################################
def WSCNNLSTMwithNoisy(shape = None, params = None, penalty = 0.0005):

    digit_input = Input(shape=shape)
    X = Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape)(digit_input)
    X = MaxPooling2D((1, 8))(X)
    X = Dropout(0.2)(X)
    X = TimeDistributed(Bidirectional(LSTM(32)))(X)
    X = Dropout(params['DROPOUT'])(X)
    X = Conv1D(2, 1, activation='softmax', kernel_regularizer=regularizers.l2(penalty))(X)
    output = ANDNoisy(a=7.5)(X)

    model = Model(inputs = digit_input, outputs=output)
    print (model.summary())
    return model
###############################


######################################
def WSCNNLSTMwithNisy(shape = None, params = None, penalty = 0.0005):

    digit_input = Input(shape=shape)
    X = Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape)(digit_input)
    X = MaxPooling2D((1, 8))(X)
    X = Dropout(0.2)(X)
    X = TimeDistributed(Bidirectional(LSTM(32)))(X)
    X = Dropout(params['DROPOUT'])(X)
    X = Conv1D(1, 1, activation='sigmoid', kernel_regularizer=regularizers.l2(penalty))(X)
    output = Dense(1)(X)

    model = Model(inputs = digit_input, outputs=output)
    print (model.summary())
    return model
###############################





# build other models
def WSCNNLSTMwithMax(shape = None, params = None, penalty = 0.005):
    digit_input = Input(shape=shape)
    X = Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape)(digit_input)
    X = MaxPooling2D((1, 8))
    X = Dropout(0.2)
    X = TimeDistributed(Bidirectional(LSTM(32)))
    X = Dropout(params['DROPOUT'])
    X = Conv1D(1, 1, activation='sigmoid', kernel_regularizer=regularizers.l2(penalty))
    output = GlobalMaxPooling1D()

    model = Model(inputs=digit_input, output=output)
    print (model.summary())
    return model

def WSCNNLSTMwithAve(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D((1, 8)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Bidirectional(LSTM(32))))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv1D(1, 1, activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalAveragePooling1D())
    
    print (model.summary())
    return model 


def RNNls(shape = None, params = None, penalty = 0.0005):
    digit_input = Input(shape = shape)
    #X = Embedding(input_dim=4,output_dim=64)(digit_input)
    X = tf.reshape(digit_input,(-1,9,480))
    X = recurrent.LSTM(32, return_sequences=True)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = LSTM(32)(X)
    X = Dropout(params['DROPOUT'])(X)
    output = Dense(1,activation='sigmoid')(X)
    model = Model(inputs = digit_input, outputs = output)
    print(model.summary())
    return model

def DeepSite(shape = None , params = None ,penalty = 0.0005):
    model = Sequential()
    model.add(TimeDistributed(Bidirectional(LSTM(256))))
    model.add(Conv1D(128, 24, padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),
                     input_shape=shape))
    model.add(Dropout(params['DROPOUT']))
    model.add(MaxPooling1D(128))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='sigmoid'))

    print(model.summary())
    return model