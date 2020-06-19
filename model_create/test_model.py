# ====================================================
# モデルの評価単体
# 読み込むモデルが無い場合は先にcreate_model.pyで作成
# ====================================================

import torch

from utils.net import Net
from utils.data_loader import load_data
from utils.operation import ModelOperation


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_name = './data'
    _, test_loader, _ = load_data(root_name, 128)

    # 読み込み
    model_path = 'model.pth'
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(model)

    # 評価
    ope = ModelOperation(model, device)
    ope.test_model(test_loader)


if __name__ == "__main__":
    main()
