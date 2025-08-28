import torch
import torch.nn as nn

from .double_conv_block import DoubleConvBlock

class VGGStyleConvBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_filters: tuple | list,
                 kernel_sizes: tuple | list,
                 pool_types: tuple | list,
                 pool_sizes: tuple | list,
                 padding_types: tuple | list,
                 normalizations: tuple | list,
                 activations: tuple | list
                 ):
        super().__init__()

        layers = []
        last_filter = input_channels
        for out_channels, kernel_size, pool_type, pool_size, pad, normalization, activation in \
            zip(
                num_filters, 
                kernel_sizes, 
                pool_types, 
                pool_sizes, 
                padding_types, 
                normalizations, 
                activations):

            layers.append(
                DoubleConvBlock(
                    in_channels=last_filter,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    pool_type=pool_type,
                    padding=pad,
                    normalization=normalization,
                    activation_function=activation
                )
            )
            
            last_filter = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

