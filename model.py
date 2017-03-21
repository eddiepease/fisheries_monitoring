##############
#specify the model that is used
##############

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adam

#this is the model which is used to classify the whole image i.e. without bounding box
# so this model in effect classifies the boat which is used.
def create_boat_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(32, 32, 3), dim_ordering='tf'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

#this model is used to classify the images generated from the bounding box script
def create_bb_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(128, 128, 3), dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = Adam()
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model