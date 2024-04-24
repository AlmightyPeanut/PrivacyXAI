import torch
import torch.nn.functional as F
from torch import nn


class LRClassifier(nn.Module):
    def __init__(self, number_of_features: int, number_of_classes: int):
        super(LRClassifier, self).__init__()

        self.output_layer = nn.Linear(number_of_features, number_of_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)

        x = self.output_layer(x)
        x = F.softmax(x, dim=-1)

        return x