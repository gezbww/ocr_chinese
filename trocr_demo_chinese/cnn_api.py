import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        in_channels=5
        out_channels=10
        # 添加CNN层的定义，可以根据需要进行修改
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=1)
        ...

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        ...
        return x
