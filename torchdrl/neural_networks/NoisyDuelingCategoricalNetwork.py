import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchdrl.neural_modules.NoisyLinear import NoisyLinear

from torchdrl.neural_networks.BaseNetwork import BaseNetwork
from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork

class NoisyDuelingCategoricalNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, atom_size:int, support, hidden_layers:list, activations:list, dropouts:list, final_activation:str, body:list=[], device="cpu"):
        super(NoisyDuelingCategoricalNetwork, self).__init__(input_shape, body, device)

        self._support = support
        self._atom_size = atom_size
        self._n_actions = n_actions

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activation) is not str and final_activation is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        adv_net = self.CreateNetList(input_shape, n_actions*self._atom_size, hidden_layers, activations, dropouts, final_activation)
        val_net = self.CreateNetList(input_shape, self._atom_size, hidden_layers, activations, dropouts, None)

        self._adv = nn.Sequential(*adv_net)
        self._val = nn.Sequential(*val_net)
        self.to(self._device)

    def CreateNetList(self, input_shape, n_actions, hidden_layers, activations, dropouts, final_activation):
        net = []

        in_features = int(np.prod(input_shape))
        if hidden_layers:
            out_features = hidden_layers[0]

            net.append(NoisyLinear(in_features, out_features))
            if activations:
                net.append(self.GetActivation(activations[0]))
            if dropouts:
                net.append(nn.Dropout(dropouts[0]))

            for i in range(1, len(hidden_layers)):
                in_features = out_features
                out_features = hidden_layers[i]

                net.append(NoisyLinear(in_features, out_features))
                if activations and len(activations) > i:
                    net.append(self.GetActivation(activations[i]))
                if dropouts and len(dropouts) > i:
                    net.append(nn.Dropout(dropouts[i]))
            in_features = out_features
            
        out_features = n_actions
        net.append(NoisyLinear(in_features, out_features))

        if final_activation is not None:
            net.append(self.GetActivation(final_activation))

        return net
        
    def forward(self, state):
        dist = self.DistributionForward(state)
        q = torch.sum(dist * self._support, dim=2)

        return q

    def DistributionForward(self, state):
        x = state
        if self._body:
            x = self._body(x)
        adv = self._adv(x).view(-1, self._n_actions, self._atom_size)
        val = self._val(x).view(-1, 1, self._atom_size)

        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3) # to avoid not a numbers

        return dist

    def ResetNoise(self):
        for net in self._val:
            if type(net) == NoisyLinear:
                net.ResetNoise()

        for net in self._adv:
            if type(net) == NoisyLinear:
                net.ResetNoise()
