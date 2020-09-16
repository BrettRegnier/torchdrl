import torch
import torch.nn as nn
import numpy as np

from .BaseNetwork import BaseNetwork
from .Flatten import Flatten

class ConvolutionNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, filters:list, kernels: list, strides:list, paddings: list, activations:list, pools:list, flatten: bool):
        super(ConvolutionNetwork, self).__init__(input_shape)

        self._input_shape = input_shape

        channels = input_shape[0]
        if channels <= 0:
            raise AssertionError("Number of channels must be at least 1")

        num_filters = len(filters)
        if num_filters <= 0:
            raise AssertionError("Number of filters needs to be greater than 0")
        
        if len(kernels) != num_filters or len(strides) != num_filters or len(paddings) != num_filters:
            msg = "filters: %d, kernels: %d, stides: %d, paddings: %d - Must all be equal size" % (num_filters, len(kernels), len(strides), len(paddings))
            raise AssertionError(msg)

        self.AssertParameter(filters, "filters", int)
        self.AssertParameter(kernels, "kernels", int)
        self.AssertParameter(strides, "strides", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(paddings, "paddings", int, min_value=0)

        convos = []
        convos.append(nn.Conv2d(channels, filters[0], kernels[0], strides[0], padding=paddings[0]))
        if len(activations) > 0 and activations[0] is not None:
            convos.append(self.GetActivation(activations[0]))
        if len(pools) > 0 and pools[0] is not None:
            convos.append(nn.MaxPool2d(pools[0]))
        
        for i in range(1, len(filters)):
            convos.append(nn.Conv2d(filters[i-1], filters[i], kernels[i], strides[i], padding=paddings[i]))
            if len(activations) > i and activations[i] is not None:
                convos.append(self.GetActivation(activations[i]))
            if len(pools) > i and pools[i] is not None:
                convos.append(nn.MaxPool2d(pools[i]))

        if flatten:
            convos.append(Flatten())
            
        # initialize weights
        for layer in convos:
            if type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
            
        self._net = nn.Sequential(*convos)
        self._net_list = convos

        out = self.forward(torch.zeros(1, *self._input_shape))
        self._output_size = (int(np.prod(out.size())),)

    def forward(self, state):
        return self._net(state)

# x = ConvolutionNetwork((2, 3, 3), [2], [1], [1], [0], ["relu"], [], True)
# out = x(torch.zeros(1, *[2, 3, 3]))
# print(x)
# print(out)
# print(out.size())
# out = torch.mean(out)
# print(out)
# out.backward()