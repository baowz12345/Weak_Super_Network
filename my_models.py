#/usr/bin/python

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Flatten, Input, concatenate, BatchNormalization, \
    TimeDistributed, Bidirectional, LSTM, Reshape, Activation, Convolution1D
from keras import regularizers

from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, Input, concatenate, \
    BatchNormalization, Activation, add, Bidirectional, LSTM, GRU, AveragePooling1D, ZeroPadding1D, recurrent
from keras import regularizers
from keras.layers.embeddings import Embedding

############################
from Noisy_and import ANDNoisy


def RNNls(shape = None, params = None, penalty = 0.0005):
    digit_input = Input(shape = shape)

    X = recurrent.LSTM(32, return_sequences=True)(digit_input)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = LSTM(32)(X)
    X = Dropout(params['DROPOUT'])(X)
    output = Dense(1,activation='sigmoid')(X)
    model = Model(inputs = digit_input, outputs = output)
    print(model.summary())
    return model

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



# model = Sequential()
# model.add(Embedding(max_features, output_dim=256))
# model.add(LSTM(128))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# #####################################
# input_tensor = Input(shape=(self.config.max_len, len(self.words)))
#         lstm = LSTM(512, return_sequences=True)(input_tensor)
#         dropout = Dropout(0.6)(lstm)
#         lstm = LSTM(256)(dropout)
#         dropout = Dropout(0.6)(lstm)
#         dense = Dense(len(self.words), activation='softmax')(dropout)
#         self.model = Model(inputs=input_tensor, outputs=dense)
#         optimizer = Adam(lr=self.config.learning_rate)
#         self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


 # input_tensor = Input(shape=(self.config.max_len, len(self.words)))
 #        lstm = LSTM(512, return_sequences=True)(input_tensor)
 #        dropout = Dropout(0.6)(lstm)
 #        lstm = LSTM(256)(dropout)
 #        dropout = Dropout(0.6)(lstm)
 #        dense = Dense(len(self.words), activation='softmax')(dropout)
 #        self.model = Model(inputs=input_tensor, outputs=dense)
 #        optimizer = Adam(lr=self.config.learning_rate)
 #        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
 #