import re
import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x


def image_preprocessing(image):
    gray_img = ImageOps.invert(image.convert('L').resize((28, 28)))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    image = transform(gray_img).unsqueeze(0)
    return image


def predict_image(request_data):
    image = Image.open(BytesIO(base64.b64decode(request_data.split(',')[1])))
    image = image_preprocessing(image)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'model.pth'
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()
    output = model(image.to(device))
    pred_rate, pred_class = torch.max(output, 1)

    '''
    print(pred_rate)
    print(pred_class)
    print(pred_class[0].item())
    '''
    return pred_class[0].item()
