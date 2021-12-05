import json
from keras.initializers import normal, identity
from keras.models import model_from_json,load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def createCNNwithAdam(learningRate = 0.00025, inputDimensions = (12, 10, 3), pretrained=None):

    model = Sequential()

    model.add(Conv2D(32, (8, 8), padding="same", strides=(4, 4), input_shape=inputDimensions))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1) ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5))

    adam = Adam(lr=learningRate)
    model.compile(loss='mse',optimizer=adam)
    if pretrained is not None:
        model.load_weights(pretrained)

    return model
