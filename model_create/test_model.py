# ====================================================
# モデルの評価単体
# 読み込むモデルが無い場合は先にcreate_model.pyで作成
# ====================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from utils.net import Net
from utils.data_loader import load_data
from utils.operation import ModelOperation


colors = ["red", "green", "blue", "orange", "purple",
          "brown", "fuchsia", "grey", "olive", "lightblue"]


def plot_tsne(output, label, cnt):
    pt = TSNE(n_components=2, random_state=0).fit_transform(output)
    for p, l in zip(pt, label):
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l])

    print(f"plot count = {cnt}")


def visalize_tsne(model, device, loader):
    model = model.eval()
    plt.figure(figsize=(10, 10))
    for i, (x, label) in enumerate(loader):
        x = x.to(device)
        output = model(x)
        output = output.to("cpu").detach().numpy()
        label = label.to("cpu").detach().numpy()
        plot_tsne(output, label, i+1)
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_name = './data'
    _, test_loader, _ = load_data(root_name, 10000)

    # 読み込み
    model_path = 'model.pth'
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 評価
    ope = ModelOperation(model, device)
    ope.test_model(test_loader)

    # 描画
    visalize_tsne(model, device, test_loader)


if __name__ == "__main__":
    main()
