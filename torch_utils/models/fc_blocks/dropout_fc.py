import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.activation_functions import get_activation_function

class DropoutLinear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 dropout: float = 0,
                 activation: str = 'relu'):
        
        super().__init__()
        layers = [
            nn.Linear(in_features=in_features, out_features=out_features),
            get_activation_function(activation)
        ]

        if dropout != 0:
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

