import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def load_data(root_name, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    train_set = torchvision.datasets.MNIST(root=root_name,
                                           train=True,
                                           download=True,
                                           transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_set = torchvision.datasets.MNIST(root=root_name,
                                          train=False,
                                          download=True,
                                          transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)
    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    return train_loader, test_loader, classes
