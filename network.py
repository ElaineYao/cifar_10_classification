from utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D

def network():
    num_classes = 10
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(32,32,3),
              padding='same', activation='relu',
              kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3),
                     padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=0.01, momentum=0.9, decay=(0.01/25), nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model