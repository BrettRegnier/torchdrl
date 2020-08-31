import torch.nn as nn
import torch.distributions as dis
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from models.BaseNetwork import BaseNetwork
from models.FullyConnectedNetwork import FullyConnectedNetwork
from models.ConvolutionNetwork import ConvolutionNetwork

class TwoHeadedNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, final_activation: str, convo=None):
        super(TwoHeadedNetwork, self).__init__(input_shape)

        if type(n_actions) is not int:
            raise AssertionError("Input shape must be of type int")
        if type(final_activation) is not str and final_activation is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)

        last_layer = (hidden_layers[-1:])[0]
        last_activation = (activations[-1:])[0]
        self._net = FullyConnectedNetwork(input_shape, last_layer, hidden_layers, activations, last_activation, convo)

        heads = [nn.Linear(last_layer, n_actions)]
        if final_activation is not None:
            heads.append(self.GetActivation(final_activation))

        self._head1 = nn.Sequential(*heads)
        self._head2 = nn.Sequential(*heads)
        
        # TODO add this in
        self._net_list = None

    def forward(self, state):
        state = self._net(state)
        return self._head1(state), self._head2(state)
