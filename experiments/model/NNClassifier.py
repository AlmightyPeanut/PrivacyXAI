import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

LAYER_SIZE_1 = 32
LAYER_SIZE_2 = 32
XAVIER_GAIN = 1.0


class NNClassifier(nn.Module):
    def __init__(self, number_of_features: int, number_of_classes: int):
        super(NNClassifier, self).__init__()
        self.number_of_classes = number_of_classes

        self.fc1 = nn.Linear(number_of_features, LAYER_SIZE_1)
        nn.init.xavier_uniform_(self.fc1.weight, XAVIER_GAIN)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(LAYER_SIZE_1, LAYER_SIZE_2)
        nn.init.xavier_uniform_(self.fc2.weight, XAVIER_GAIN)
        nn.init.zeros_(self.fc2.bias)

        self.output_layer = nn.Linear(LAYER_SIZE_2, number_of_classes)
        nn.init.xavier_uniform_(self.output_layer.weight, XAVIER_GAIN)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.output_layer(x)
        if self.number_of_classes > 1:
            x = F.softmax(x, dim=-1)
        else:
            x = F.sigmoid(x)

        return x
