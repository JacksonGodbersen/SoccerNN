import math

import torch
from torch import nn
import numpy as np

class NueralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        force_magnitude = torch.sigmoid(logits[:, 0]).item()
        force_direction = (torch.cos(logits[:, 1]) * 2 * torch.pi).item()
        return force_magnitude, force_direction


# model = NueralNetwork()
# input_data = np.array([3, 2, 1, 4, 5, 6, 7, 8, 9, 10])  # A NumPy array with 10 elements
#
# # Convert NumPy array to PyTorch tensor
# input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

