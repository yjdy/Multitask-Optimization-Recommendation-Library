import torch
from torch import nn
def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
            return [get_activation(act) for act in activation]
    return activation

