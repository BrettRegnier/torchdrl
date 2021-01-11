import torch
import torch.nn as nn

import numpy as np

from torchdrl.neural_networks.BaseNetwork import BaseNetwork

class CombineNetwork(BaseNetwork):
    def __init__(self, in_networks:list, device="cpu"):
        super(CombineNetwork, self).__init__((0,), None, device)

        self._networks = nn.ModuleList()
        for net in in_networks:
            assert issubclass(type(net), BaseNetwork)
            self._networks.append(net)

        self.to(self._device)

    def forward(self, state):
        outs = torch.tensor([], device=self._device)
        for state, network in zip(state, self._networks):
            out = network(state)
            outs = torch.cat([outs, out], 1)
        return outs

    def OutputSize(self):
        input_shapes = self.InputShape()

        outs = []
        for network, input_shape in zip(self._networks, input_shapes):
            out = network.forward(torch.zeros(1, *input_shape, device=self._device))
            out_size = int(np.prod(out.size()))
            outs.append(out_size)
            
        return (int(np.sum(outs)),)

    def InputShape(self):
        input_shapes = []
        for network in self._networks:
            input_shapes.append(network.InputShape())
        return input_shapes



