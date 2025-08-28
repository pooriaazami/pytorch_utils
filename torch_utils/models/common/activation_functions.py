import torch
import torch.nn as nn

def get_activation_function(function: str):
    match function:
        case 'relu':
            return nn.ReLU()
        case 'selu':
            return nn.SELU()
        case 'tanh':
            return nn.Tanh()
        case 'sigmoid':
            return nn.Sigmoid()
        case 'prelu':
            return nn.PReLU()
        case 'leaky_relu':
            return nn.LeakyReLU()