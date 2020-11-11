import torch
import torch.nn as nn
from collections import OrderedDict


__all__ = ['MLP']

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)  

    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.fc2(out)
        return out