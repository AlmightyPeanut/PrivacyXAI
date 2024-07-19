import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(42)


class LRClassifier(nn.Module):
    def __init__(self, number_of_features: int, number_of_classes: int):
        super(LRClassifier, self).__init__()
        self.number_of_classes = number_of_classes

        self.output_layer = nn.Linear(number_of_features, number_of_classes)
        nn.init.xavier_uniform_(self.output_layer.weight, 1.0)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output_layer(x)
        if self.number_of_classes > 1:
            x = F.softmax(x, dim=-1)
        else:
            x = F.sigmoid(x)

        return x
