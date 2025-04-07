import torch.nn as nn
import torch

def get_activation(act_name):
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_name == 'elu':
        return nn.ELU()
    elif act_name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {act_name}")

class ClassifierModel(nn.Module):
    def __init__(self, embedding_dim, dropout_rate=0.2, activation='elu'):
        super(ClassifierModel, self).__init__()
        units1 = embedding_dim
        units2 = embedding_dim * 3 // 4
        units3 = embedding_dim // 2
        units4 = embedding_dim // 4

        self.act = get_activation(activation)
        self.fc1 = nn.Linear(embedding_dim, units1)
        self.bn1 = nn.BatchNorm1d(units1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_res1 = nn.Linear(units1, units2)
        self.bn_res1 = nn.BatchNorm1d(units2)
        self.dropout_res1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(units1, units2)
        self.fc3 = nn.Linear(units2, units2)
        self.fc_res2 = nn.Linear(units2, units3)
        self.bn_res2 = nn.BatchNorm1d(units3)
        self.dropout_res2 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(units2, units3)
        self.fc5 = nn.Linear(units3, units3)
        self.fc_final = nn.Linear(units3, units4)
        self.bn_final = nn.BatchNorm1d(units4)
        self.dropout_final = nn.Dropout(dropout_rate)
        self.out = nn.Linear(units4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        res1 = self.fc_res1(x)
        res1 = self.bn_res1(res1)
        res1 = self.act(res1)
        res1 = self.dropout_res1(res1)
        x2 = self.fc2(x)
        x2 = x2 + res1
        x3 = self.fc3(x2)
        x3 = x3 + res1
        res2 = self.fc_res2(x3)
        res2 = self.bn_res2(res2)
        res2 = self.act(res2)
        res2 = self.dropout_res2(res2)
        x4 = self.fc4(x3)
        x4 = x4 + res2
        x5 = self.fc5(x4)
        x5 = x5 + res2
        x6 = self.fc_final(x5)
        x6 = self.bn_final(x6)
        x6 = self.act(x6)
        x6 = self.dropout_final(x6)
        out = self.out(x6)
        return torch.sigmoid(out).squeeze(1)
