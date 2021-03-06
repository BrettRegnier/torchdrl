import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Retrieved from
# https://github.com/higgsfield/RL-Adventure/blob/master/common/layers.py

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._std_init = std_init

        self._weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self._weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('_weight_epsilon', torch.empty(out_features, in_features))

        self._bias_mu = nn.Parameter(torch.empty(out_features))
        self._bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('_bias_epsilon', torch.empty(out_features))

        self.ResetParameters()
        self.ResetNoise()

        # self.__name__

    def forward(self, x):
        if self.training:
            weight = self._weight_mu + self._weight_sigma * self._weight_epsilon
            bias = self._bias_mu + self._bias_sigma * self._bias_epsilon
        else:
            weight = self._weight_mu
            bias = self._bias_mu

        return F.linear(x, weight, bias)

    def ResetParameters(self):
        mu_range = 1 / math.sqrt(self._in_features)

        self._weight_mu.data.uniform_(-mu_range, mu_range)
        self._weight_sigma.data.fill_(self._std_init / math.sqrt(self._in_features))

        self._bias_mu.data.uniform_(-mu_range, mu_range)
        self._bias_sigma.data.fill_(self._std_init / math.sqrt(self._out_features))

    def ResetNoise(self):
        epsilon_in = self._ScaleNoise(self._in_features)
        epsilon_out = self._ScaleNoise(self._out_features)

        self._weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self._bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _ScaleNoise(size):
        x = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=size), dtype=torch.float32)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def __repr__(self):
        return f"NoisyLinear(in_features={self._in_features}, out_features={self._out_features}, std_init={self._std_init})"
