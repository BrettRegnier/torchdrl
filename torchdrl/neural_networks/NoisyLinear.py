import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .BaseNetwork import BaseNetwork

# Retrieved from
# https://github.com/higgsfield/RL-Adventure/blob/master/common/layers.py

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._std_init = std_init

        self._weight_mu = nn.Parameter(torch.tensor(out_features, in_features, dtype=torch.float32))
        self._weight_sigma = nn.Parameter(torch.tensor(out_features, in_features, dtype=torch.float32))
        self.register_buffer('weight_epsilon', torch.tensor(out_features, in_features, dtype=torch.float32))

        self._bias_mu = nn.Parameter(torch.tensor(out_features, dtype=torch.float32))
        self._bias_sigma = nn.Parameter(torch.tensor(out_features, dtype=torch.float32))
        self.register_buffer('bias_epsilon', torch.tensor(out_features, in_features, dtype=torch.float32))

        self.ResetParameters()
        self.ResetNoise()

    def forward(self, x):
        weight_epsilon = self.weight_epsilon
        bias_epsilon = self.bias_epsilon

        if self.training:
            weight = self._weight_mu + self._weight_sigma.mul(Variable(weight_epsilon))
            bias = self._bias_mu + self._bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self._weight_mu
            bias = self._bias_mu

        return F.linear(x, weight, bias)

    def ResetParameters(self):
        mu_range = 1 / math.sqrt(self._weight_mu.size(1))

        self._weight_mu.data.uniform_(-mu_range, mu_range)
        self._weight_sigma.data.fill_(self._std_init / math.sqrt(self._weight_sigma.size(1)))

        self._bias_mu.data.uniform_(-mu_range, mu_range)
        self._bias_sigma.data.fill_(self._std_init / math.sqrt(self._bias_sigma.size(0)))

    def ResetNoise(self):
        print(self._in_features, self._out_features)

        epsilon_in = self._ScaleNoise(self._in_features)
        epsilon_out = self._ScaleNoise(self._out_features)

        print(epsilon_in)
        print(epsilon_out)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._ScaleNoise(self._out_features))

    def _ScaleNoise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
