import numpy as np
from keras.datasets import mnist


def generate_load_data(input_shape, embedding):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.reshape(x_train, (len(x_train), input_shape[0], input_shape[1], 1))
    x_test = np.reshape(x_test, (len(x_test), input_shape[0], input_shape[1], 1))
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    y_train = np.hstack([y_train, ]*embedding)
    y_test = np.hstack([y_test, ]*embedding)
    train = (x_train, y_train)
    test = (x_test, y_test)
    return train, test
