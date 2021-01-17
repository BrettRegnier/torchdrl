import numpy as np

import torch.nn as nn

from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.BaseNetwork import BaseNetwork


class DuelingNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activation:str, bodies:list=[], device="cpu"):
        super(DuelingNetwork, self).__init__(input_shape, bodies, device)

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activation) is not str and final_activation is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        adv_net = self.CreateNetList(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation)
        val_net = self.CreateNetList(input_shape, 1, hidden_layers, activations, dropouts, final_activation)

        self._adv = nn.Sequential(*adv_net)
        self._val = nn.Sequential(*val_net)

        self.to(self._device)

    def CreateNetList(self, input_shape, n_actions, hidden_layers, activations, dropouts, final_activation):
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
                if activations and len(activations) > i:
                    net.append(self.GetActivation(activations[i]))
                if dropouts and len(dropouts) > i:
                    net.append(nn.Dropout(dropouts[i]))
            in_features = out_features
            
        out_features = n_actions
        net.append(nn.Linear(in_features, out_features))

        if final_activation is not None:
            net.append(self.GetActivation(final_activation))

        return net

    def forward(self, state):
        if self._body:
            state = self._body(state)
            
        val = self._val(state)
        adv = self._adv(state)

        q = val + adv - adv.mean(dim=-1, keepdim=True)

        return q


