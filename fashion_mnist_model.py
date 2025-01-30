# fashion_mnist_model.py

import torch.nn as nn
import torch.nn.functional as F

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        # Input: 1x28x28 for Fashion-MNIST
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28x28 -> 14x14 after pool -> 7x7 after 2nd pool
        self.fc2 = nn.Linear(128, 10)  # Fashion-MNIST has 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # -> 64x7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)