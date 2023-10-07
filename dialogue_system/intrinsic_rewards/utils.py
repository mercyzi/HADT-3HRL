import collections

import numpy as np
import torch.nn as nn

def prod(iterable):
    p = 1
    for i in iterable:
        p *= i
    return p

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module 

init_ = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
)    

def build_sequential(num_inputs, hiddens, activation="relu", output_activation=True):
    modules = [Flatten()]
    if activation == "relu":
        nonlin = nn.ReLU
    elif activation == "tanh":
        nonlin = nn.Tanh
    else:
        raise ValueError(f"Unknown activation option {activation}!")
    
    assert len(hiddens) > 0
    modules.append(init_(nn.Linear(num_inputs, hiddens[0])))
    for i in range(len(hiddens) - 1):
        modules.append(nonlin())
        modules.append(init_(nn.Linear(hiddens[i], hiddens[i + 1])))
    if output_activation:
        modules.append(nonlin())
    return nn.Sequential(*modules)