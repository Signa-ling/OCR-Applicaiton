import torch

from utils.net import Net
from utils.data_loader import load_data
from utils.operation import ModelOperation


def main():
    # ハイパーパラメータ
    epoch = 5
    lr = 0.001
    batch_size = 128

    # GPU or CPUの自動判別
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # データセットの保存先
    root_name = './data'

    # datasetの呼び出し
    train_loader, test_loader, classes = load_data(root_name, batch_size)

    # network呼び出し
    model = Net().to(device)

    # modelの動作クラス
    ope = ModelOperation(model, device)

    # 学習
    ope.train_model(train_loader, epoch, lr)

    # modelのsave
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)

    # 評価
    ope.test_model(test_loader)


if __name__ == "__main__":
    main()
