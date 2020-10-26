import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

from torchdrl.neural_networks.BaseNetwork import BaseNetwork
from torchdrl.neural_networks.NoisyLinear import NoisyLinear
from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.Flatten import Flatten

class ConstraintNetwork(BaseNetwork):
    def __init__(self, atom_size, support, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activations:str, convo=None):
        super(ConstraintNetwork, self).__init__(input_shape)

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
        hidden_layers = hidden_layers[-1:]

        last_activation = (activations[-1:])[0]
        activations= activations[-1:]


        channels = input_shape[0]


        self._participant_conv = nn.Sequential(
            nn.Conv2d(channels, 20, [1, 1], 1, 0),
            nn.ReLU(),
            Flatten()
        )
        participant_out = self._participant_conv.forward(torch.zeros(1, *input_shape))
        participant_out_size = int(np.prod(participant_out.size()))
        self._participant_linear = nn.Sequential(
            nn.Linear(participant_out_size, 1024),
            nn.ReLU()
        )


        self._lr_conv = nn.Sequential(
            nn.Conv2d(channels, 20, [1, 4], 1, 0),
            nn.ReLU(),
            Flatten()
        )
        lr_out = self._lr_conv.forward(torch.zeros(1, *input_shape))
        lr_out_size = int(np.prod(lr_out.size()))
        self._lr_linear = nn.Sequential(
            nn.Linear(lr_out_size, 1024),
            nn.ReLU()
        )

        self._fb_conv = nn.Sequential(
            nn.Conv2d(channels, 20, [2, 2], 1, 0),
            nn.ReLU(),
            Flatten()
        )
        fb_out = self._fb_conv.forward(torch.zeros(1, *input_shape))
        fb_out_size = int(np.prod(fb_out.size()))
        self._fb_linear = nn.Sequential(
            nn.Linear(fb_out_size, 1024),
            nn.ReLU()
        )

        self._comb = nn.Linear(3072, 1024)

        self._adv = nn.Sequential(
            NoisyLinear(1024, 1024),
            nn.ReLU(),
            NoisyLinear(1024, n_actions * self._atom_size)
        )
        self._val = nn.Sequential(
            NoisyLinear(1024, 1024),
            nn.ReLU(),
            NoisyLinear(1024, self._atom_size)
        )
        
    def forward(self, state):
        dist = self.DistributionForward(state)
        q = torch.sum(dist * self._support, dim=2)

        return q

    def DistributionForward(self, state):
        lr_x = self._lr_conv(state)
        fb_x = self._fb_conv(state)
        parti_x = self._participant_conv(state)

        lr_x = self._lr_linear(lr_x)
        fb_x = self._fb_linear(fb_x)
        parti_x = self._participant_linear(parti_x)

        combined = torch.cat((lr_x, fb_x, parti_x), dim=1)
        x = self._comb(combined)

        adv = self._adv(x).view(-1, self._n_actions, self._atom_size)
        # val = self._val(x).view(-1, 1, self._atom_size)
        val = self._val(x)
        val = val.view(-1, 1, self._atom_size)

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

        


