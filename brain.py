import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        # Define a fully connected (dense) layer with 8 input features and 16 hidden units.
        self.fc1 = nn.Linear(8, 16)
        
        # Define another fully connected layer with 16 hidden units and 3 output units.
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Apply the sigmoid activation function to the first layer.
        x = torch.sigmoid(self.fc2(x))  # Apply the sigmoid activation function to the second layer.
        return x
