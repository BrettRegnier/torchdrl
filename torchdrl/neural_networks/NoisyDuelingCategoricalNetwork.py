import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseNetwork import BaseNetwork
from .NoisyLinear import NoisyLinear
from .FullyConnectedNetwork import FullyConnectedNetwork

class NoisyDuelingCategoricalNetwork(BaseNetwork):
    def __init__(self, atom_size, support, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activations:str, convo=None):
        super(NoisyDuelingCategoricalNetwork, self).__init__(input_shape)

        self._support = support
        self._atom_size = atom_size
        self._n_actions = n_actions

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activations) is not str and final_activations is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        # scrape one layer off for the advantage and value layers
        prev_layer = hidden_layers[-2:][0]
        last_layer = hidden_layers[-1:][0]
        hidden_layers = hidden_layers[:-1]

        last_activation = (activations[-1:])[0]
        activations= activations[:-1]

        self._net = FullyConnectedNetwork(input_shape, prev_layer, hidden_layers, activations, dropouts, last_activation, convo)

        self._adv = nn.Sequential(
            NoisyLinear(prev_layer, last_layer),
            nn.ReLU(),
            NoisyLinear(last_layer, n_actions * self._atom_size)
        )
        self._val = nn.Sequential(
            NoisyLinear(prev_layer, last_layer),
            nn.ReLU(),
            NoisyLinear(last_layer, self._atom_size)
        )
        
    def forward(self, state):
        dist = self.DistributionForward(state)
        q = torch.sum(dist * self._support, dim=2)

        return q

    def DistributionForward(self, state):
        x = self._net(state)
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
