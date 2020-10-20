import torch.nn as nn

import numpy as np

from .BaseNetwork import BaseNetwork
from .ConvolutionNetwork import ConvolutionNetwork

class FullyConnectedNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activation:str, convo=None):
        super(FullyConnectedNetwork, self).__init__(input_shape)

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activation) is not str and final_activation is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        num_hidden_layers = len(hidden_layers)

        if convo is None:
            self._convo = None
            in_features = [np.prod(input_shape)]
        else:
            self._convo = ConvolutionNetwork(input_shape, convo['filters'], convo['kernels'], convo['strides'], convo['paddings'], convo['activations'], convo['pools'], convo['flatten'])
            in_features = self._convo.OutputSize()

        net = []
        if num_hidden_layers > 1:
            net.append(nn.Linear(*in_features, hidden_layers[0]))
            if len(activations) > 0 and activations[0] is not None:
                net.append(self.GetActivation(activations[0]))
            if len(dropouts) > 0 and dropouts[0] is not None:
                net.append(nn.Dropout(dropouts[0]))

            for i in range(1, num_hidden_layers):
                net.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                if len(activations) > i and activations[i] is not None:
                        net.append(self.GetActivation(activations[i]))
                if len(dropouts) > i and dropouts[i] is not None:
                        net.append(nn.Dropout(dropouts[i]))

            net.append(nn.Linear(hidden_layers[i], n_actions))
            if final_activation is not None:
                net.append(self.GetActivation(final_activation))
        else:
            net.append(nn.Linear(*in_features, n_actions))
            if len(activations) > 0 and activations[0] is not None:
                net.append(self.GetActivation(activations[0]))
            
        # initialize weights
        for layer in net:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)

        self._net = nn.Sequential(*net)
        self._net_list = net

    def forward(self, state):
        if self._convo is not None:
            state = self._convo(state)
        return self._net(state)
