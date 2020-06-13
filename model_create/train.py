import numpy

from keras.optimizers import Adam

from utils import dataset, model, triplet_loss


def main():
    embedding = 32
    batch_size = 50
    epochs = 20
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    train, test = dataset.generate_load_data(input_shape, embedding)

    createModel = model.base_model(input_shape, embedding)
    createModel.summary()

    loss = triplet_loss.loss_function_maker(batch_size, 0.5)
    opt = Adam(learning_rate=0.0001)

    createModel.compile(loss=loss, optimizer=opt)
    history = createModel.fit(train[0], train[1],
                              batch_size=batch_size,
                              epochs=epochs, verbose=1,
                              validation_data=(test[0], test[1]))

    createModel.save('./model.h5', include_optimizer=False)


if __name__ == "__main__":
    main()
