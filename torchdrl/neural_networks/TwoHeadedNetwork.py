import torch.nn as nn
import torch.distributions as dis
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from .BaseNetwork import BaseNetwork
from .FullyConnectedNetwork import FullyConnectedNetwork
from .ConvolutionNetwork1D import ConvolutionNetwork1D

class TwoHeadedNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, head1_output:int, head2_output:int, hidden_layers:list, activations:list, final_activation: str, convo=None):
        super(TwoHeadedNetwork, self).__init__(input_shape)

        if type(head1_output) is not int:
            raise AssertionError("Input shape must be of type int")
        if type(head2_output) is not int:
            raise AssertionError("Input shape must be of type int")
        if type(final_activation) is not str and final_activation is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)

        last_layer = hidden_layers[-1:]
        last_activation = activations[-1:]


        self._net = FullyConnectedNetwork(input_shape, last_layer[0], hidden_layers, activations, [], last_activation[0], convo)

        net_output = self._net.OutputSize()

        # TODO fix this garbage
        # last_layer = [512]
        # activations = ['relu']
        self._head1 = FullyConnectedNetwork(net_output, head1_output, last_layer, last_activation, [], None, None)
        self._head2 = FullyConnectedNetwork(net_output, head2_output, last_layer, last_activation, [], None, None)



    def forward(self, state):
        state = self._net(state)
        return self._head1(state), self._head2(state)
