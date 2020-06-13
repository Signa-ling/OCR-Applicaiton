import numpy as np
import h5py
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.datasets import mnist
from sklearn.manifold import TSNE

from utils.dataset import generate_load_data


def main():
    input_shape = (28, 28, 1)
    embedding = 32

    _, (x_test, _) = generate_load_data(input_shape, embedding)
    _, (_, y_test) = mnist.load_data()

    model = load_model("./model.h5")
    model.summary()
    pred = model.predict(x_test)

    print("pred")
    tsne = TSNE()
    print("tsne")
    tsne_train = tsne.fit_transform(pred)
    print(tsne_train)
    plt.figure(figsize=(16, 16))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
    for i, c in zip(range(10), colors):
        print(i, c)
        plt.scatter(tsne_train[y_test == i, 0], tsne_train[y_test == i, 1], c=c, label=str(i))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
