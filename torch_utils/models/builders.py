import yaml

from .conv_nets import VGGStyleConvBlock

def build_model(path):
    with open(path) as file:
        config = yaml.safe_load(file.read())
        
        match config['model']['type']:
            case 'cnn':
                build_conv_block_from_yaml(config['model'])

def build_vgg_block(config):
    print(*config['layers'], sep='\n\n')
    num_filters = []
    kernel_sizes = []
    pool_types = []
    pool_sizes = []
    padding_types = []
    normalizations = []
    activations = []

    for layer in config['layers']:
        num_filters.append(layer['filters'])
        kernel_sizes.append(layer.get('kernels_size', 3))

        if 'pool' in layer:
            pool_types.append(layer['pool'].get('type', 'max'))
            pool_sizes.append(layer['pool'].get('size', 2))
        else:
            pool_types.append('max')
            pool_sizes.append(2)

        padding_types.append(layer.get('padding', 'same'))
        normalizations.append(layer.get('normalization', True))
        activations.append(layer.get('activation', 'relu'))

    return VGGStyleConvBlock(
        input_channels=config.get('input_channels', 3),
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        pool_types=pool_types, 
        pool_sizes=pool_sizes, 
        padding_types=padding_types, 
        normalizations=normalizations, 
        activations=activations
    )

def build_conv_block_from_yaml(config):
    for block in config['blocks']:
        match block['type']:
            case 'vgg':
                built_block = build_vgg_block(block)
                print(built_block)