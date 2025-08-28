from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import get_activation_function

class DoubleConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 pool_type: Literal['max', 'avg', 'none'] | None = 'max',
                 padding: Literal['same', 'valid'] = 'same',
                 normalization: bool = True,
                 activation_function: str = 'relu'
                 ):
        
        super().__init__()
        pad_size = kernel_size // 2 if padding == 'same' else 0
        layers = [
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size,
                      padding=pad_size),
            get_activation_function(activation_function),
            nn.Conv2d(in_channels=out_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size,
                      padding=pad_size),
            get_activation_function(activation_function)
        ]

        match pool_type:
            case 'max':
                layers.append(nn.MaxPool2d(pool_size))
            case 'avg':
                layers.append(nn.AvgPool2d(pool_size))
            case 'none':
                pass

        if normalization:
            layers.append(nn.BatchNorm2d(out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
            