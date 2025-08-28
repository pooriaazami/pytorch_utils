import yaml

import torch
import torch.nn as nn

from .conv_nets import VGGStyleConvBlock
from .fc_blocks import DropoutLinear

class VGGConfig:
    def __init__(self, config, input_channels=3):
        self.num_filters = []
        self.kernel_sizes = []
        self.pool_types = []
        self.pool_sizes = []
        self.padding_types = []
        self.normalizations = []
        self.activations = []
        self.padding_sizes = []

        for layer in config['layers']:
            self.num_filters.append(layer['filters'])
            self.kernel_sizes.append(layer.get('kernels_size', 3))

            if 'pool' in layer:
                self.pool_types.append(layer['pool'].get('type', 'max'))
                if layer['pool']['type'] == 'none':
                    self.pool_sizes.append(layer['pool'].get('size', 2))
                else:
                    self.pool_sizes.append(layer['pool'].get('size', 0))
            else:
                self.pool_types.append('max')
                self.pool_sizes.append(2)

            self.padding_types.append(layer.get('padding', 'same'))
            self.padding_sizes.append(self.kernel_sizes[-1] // 2 if self.padding_types[-1] == 'same' else 0)
            self.normalizations.append(layer.get('normalization', True))
            self.activations.append(layer.get('activation', 'relu'))

        self.input_channels = input_channels

    def build(self):
        return VGGStyleConvBlock(
            input_channels=self.input_channels,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            pool_types=self.pool_types, 
            pool_sizes=self.pool_sizes, 
            padding_types=self.padding_types, 
            normalizations=self.normalizations, 
            activations=self.activations
        )

def build_model(path):
    with open(path) as file:
        config = yaml.safe_load(file.read())
        
        match config['model']['type']:
            case 'cnn':
                return build_conv_block_from_yaml(config['model'])
    
def calculate_conv_output(layers, input_egde, input_channel):
    model = nn.Sequential(*layers)
    
    test_input = torch.zeros((1, input_channel, input_egde, input_egde))
    with torch.no_grad():
        outputs = model(test_input).shape

    return outputs.numel()

def build_conv_block_from_yaml(config):
    layers = []
    output = None
    print(config)
    for block in config['blocks']:
        match block['type']:
            case 'vgg':
                vgg_config = VGGConfig(block, input_channels=config.get('input_channels', 3))
                model = vgg_config.build()
                layers.append(model)
            case 'flatten':
                layers.append(nn.Flatten())

                output = calculate_conv_output(layers, config['image_edge'], config.get('input_channels', 3))
            case 'global_pool':
                layers.extend([
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                ])

                output = calculate_conv_output(layers, config['image_edge'], config.get('input_channels', 3))

            case 'fc':
                if output:
                    layers.append(
                        DropoutLinear(
                            in_features=output,
                            out_features=block['neurons'],
                            activation=block.get('activation', 'relu'),
                            dropout=block.get('dropout', 0)
                        )
                    )
                else:
                    raise RuntimeError()
                
    return nn.Sequential(*layers)