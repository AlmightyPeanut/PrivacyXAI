import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

LAYER_SIZE = 256


class NNClassifier(nn.Module):
    def __init__(self, number_of_features: int, number_of_classes: int):
        super(NNClassifier, self).__init__()

        self.fc1 = nn.Linear(number_of_features, LAYER_SIZE)
        self.bn1 = nn.GroupNorm(LAYER_SIZE, LAYER_SIZE)

        self.fc2 = nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.bn2 = nn.GroupNorm(LAYER_SIZE, LAYER_SIZE)

        self.fc3 = nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.bn3 = nn.GroupNorm(LAYER_SIZE, LAYER_SIZE)

        self.output_layer = nn.Linear(LAYER_SIZE, number_of_classes)
        self.bn_output_layer = nn.GroupNorm(number_of_classes, number_of_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.output_layer(x)
        x = self.bn_output_layer(x)
        # x = F.softmax(x, dim=-1)

        return x
