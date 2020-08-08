import torch
import torch.nn as nn
import numpy as np

from BaseNetwork import BaseNetwork
from Flatten import Flatten

class ConvolutionNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, filters:list, kernels: list, strides:list, paddings: list, relus:list, pools:list, flatten: bool):
        super(ConvolutionNetwork, self).__init__()
        if type(input_shape) is not tuple:
            raise AssertionError("Input shape must be of type tuple")

        channels = input_shape[0]
        if channels <= 0:
            raise AssertionError("Number of channels must be at least 1")

        num_filters = len(filters)
        if num_filters <= 0:
            raise AssertionError("Number of filters needs to be greater than 0")
        
        if len(kernels) != num_filters or len(strides) != num_filters or len(paddings) != num_filters:
            msg = "filters: %d, kernels: %d, stides: %d, paddings: %d - Must all be equal size" % (num_filters, len(kernels), len(strides), len(paddings))
            raise AssertionError(msg)

        self.AssertParameter(filters, "filters")
        self.AssertParameter(kernels, "kernels")
        self.AssertParameter(strides, "strides")
        self.AssertParameter(paddings, "paddings", min_value=0)

        convos = []
        convos.append(nn.Conv2d(channels, filters[0], kernels[0], strides[0], padding=paddings[0]))
        if len(relus) > 0 and relus[0] is not None:
            convos.append(nn.ReLU())
        if len(pools) > 0 and pools[0] is not None:
            convos.append(nn.MaxPool2d(pools[0]))
        
        for i in range(1, len(filters)):
            convos.append(nn.Conv2d(filters[i-1], filters[i], kernels[i], strides[i], padding=paddings[i]))
            if len(relus) > i and relus[i] is not None:
                convos.append(nn.ReLU())
            if len(pools) > i and pools[i] is not None:
                convos.append(nn.MaxPool2d(pools[i]))

        if flatten:
            convos.append(Flatten())
        self._net = nn.Sequential(*convos)

        out = self.forward(torch.zeros(1, *input_shape))
        self._out_features = [int(np.prod(out.size()))]

    def forward(self, state):
        return self._net(state)

# x = ConvolutionNetwork((2, 3, 3), [1], [1], [1], [1], [1], [1],True)
# out = x(torch.zeros(1, *[2, 3, 3]))
# print(x)
# print(out.size())
# print(x._out_features)