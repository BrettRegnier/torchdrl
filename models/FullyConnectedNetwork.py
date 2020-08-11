import torch.nn as nn
import torch.distributions as dis
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from models.BaseNetwork import BaseNetwork

class FullyConnectedNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, final_activation: str):
        super(FullyConnectedNetwork, self).__init__(input_shape)

        if type(n_actions) is not int:
            raise AssertionError("Input shape must be of type int")
        if type(final_activation) is not str and final_activation is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)

        num_hidden_layers = len(hidden_layers)

        net = []
        net.append(nn.Linear(np.prod(input_shape), hidden_layers[0]))
        if len(activations) > 0 and activations[0] is not None:
            net.append(self.GetActivation(activations[0]))

        i = 0
        for i in range(1, num_hidden_layers):
            net.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            if i < len(activations):
                if activations[i] is not None:
                    net.append(self.GetActivation(activations[i]))

        net.append(nn.Linear(hidden_layers[i], n_actions))
        if final_activation is not None:
            net.append(self.GetActivation(final_activation))

        self._net = nn.Sequential(*net)

    def forward(self, state):
        return self._net(state)
