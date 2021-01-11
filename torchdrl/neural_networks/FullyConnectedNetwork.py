import torch.nn as nn

import numpy as np

from torchdrl.neural_networks.BaseNetwork import BaseNetwork

class FullyConnectedNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activation:str, bodies:list=[], device="cpu"):
        super(FullyConnectedNetwork, self).__init__(input_shape, bodies, device)

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activation) is not str and final_activation is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        net = []
        in_features = int(np.prod(input_shape))

        if hidden_layers:
            out_features = hidden_layers[0]

            net.append(nn.Linear(in_features, out_features))
            if activations:
                net.append(self.GetActivation(activations[0]))
            if dropouts:
                net.append(nn.Dropout(dropouts[0]))

            for i in range(1, len(hidden_layers)):
                in_features = out_features
                out_features = hidden_layers[i]

                net.append(nn.Linear(in_features, out_features))
                if activations:
                    net.append(self.GetActivation(activations[i]))
                if dropouts:
                    net.append(nn.Dropout(dropouts[i]))

            in_features = out_features

        out_features = n_actions
        net.append(nn.Linear(in_features, out_features))

        if final_activation is not None:
            net.append(self.GetActivation(final_activation))

        self._net = nn.Sequential(*net)
        self.to(self._device)

    def forward(self, state):
        if self._body:
            state = self._body(state)
        return self._net(state)

