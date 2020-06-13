from keras.models import Model
from keras.layers import (Activation, Conv2D, MaxPooling2D, Dense,
                          Flatten, Dropout, Input, BatchNormalization)


def base_model(input_shape, embedding):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3))(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    output = Dense(embedding)(x)
    model = Model(inputs=input_img, outputs=output)

    return model
