import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np

from torchdrl.neural_networks.Flatten import Flatten
from torchdrl.neural_networks.NoisyLinear import NoisyLinear

class TestNet(nn.Module):
    def __init__(self, atom_size, support, input_shape, n_actions, device):
        super(TestNet, self).__init__()

        self._support = support
        self._atom_size = atom_size
        self._device = device
        self._n_actions = n_actions

        self._conv = nn.Sequential(
            nn.Conv1d(input_shape[0], 128, 1),
            nn.LeakyReLU(),
            Flatten()
        )

        print(self._conv)

        out = self._conv(torch.zeros(1, *input_shape))
        conv_out = (int(np.prod(out.size())),)

        hidden_size = 1024

        self._lstm = nn.LSTM(*conv_out, hidden_size, 3)

        self._net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self._adv = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            nn.Tanh(),
            NoisyLinear(hidden_size, n_actions*self._atom_size),
        )

        self._val = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            nn.Tanh(),
            NoisyLinear(hidden_size, self._atom_size),
        )

        self.to(self._device)
        self.ResetHidden()

    def ResetHidden(self):
        self._hidden = torch.zeros((3, 1, 1024), device=self._device), torch.zeros((3, 1,1024), device=self._device)
        
    def forward(self, state):
        dist = self.DistributionForward(state)
        q = torch.sum(dist * self._support, dim=2)

        return q

    def DistributionForward(self, state):
        x = state

        x = self._conv(x)
        x = x.unsqueeze(0)
        x, self._hidden = self._lstm(x, self._hidden)
        x = F.relu(x)
        x = self._net(x)

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
