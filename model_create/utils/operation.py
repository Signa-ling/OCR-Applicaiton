import torch
import torch.optim as optim
import torch.nn as nn


class ModelOperation():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_model(self, train_loader, epoches, lr):
        print('train')
        model = self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epoches):
            running_loss = 0.0

            for i, (x, y) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 100 == 0:
                    print('%03d epoch, %05d, loss=%.5f' %
                          (epoch, i, loss.item()))
                    running_loss = 0.0
        print('Finish Training')

    def test_model(self, test_loader):
        print('test')
        model = self.model.eval()
        total, tp = 0, 0

        with torch.no_grad():
            for (x, label) in test_loader:
                x = x.to(self.device)

                y_ = model.forward(x)
                label_ = y_.argmax(1).to('cpu')

                total += label.shape[0]
                tp += (label_ == label).sum().item()

            acc = tp/total
        print('test accuracy = %.3f' % acc)
