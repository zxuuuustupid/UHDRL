import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FullyConnectedLayer2(nn.Module):
    def __init__(self):
        super(FullyConnectedLayer2, self).__init__()
        self.fc = nn.Linear(2 * 128 * 28 * 28, 16)  # 输入特征数为 2*128*28*28，输出特征数为 16

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平成二维 (batch_size, 2*128*28*28)
        x = self.fc(x)
        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(128*28*28, 128)  # 输入特征数为128*28*28，输出特征数为128

    def forward(self, x):
        x = x.view(-1, 128*28*28)  # 将输入数据展平为一维
        x = self.fc(x)
        return x

