import torch
import torch.nn as nn

class FullyConnectedLayer(nn.Module):
    def __init__(self):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(128*28*28, 128)  # 输入特征数为128*28*28，输出特征数为128

    def forward(self, x):
        x = x.view(-1, 128*28*28)  # 将输入数据展平为一维
        x = self.fc(x)
        return x