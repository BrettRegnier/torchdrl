import torch
import torch.nn as nn
import numpy as np

from torchdrl.neural_modules.Flatten import Flatten

from torchdrl.neural_networks.BaseNetwork import BaseNetwork

class ConvolutionNetwork1D(BaseNetwork):
    def __init__(self, input_shape:tuple, filters:list, kernels: list, strides:list, paddings: list, activations:list, pools:list, flatten: bool, body=None, device="cpu"):
        super(ConvolutionNetwork1D, self).__init__(input_shape, body, device)

        channels = input_shape[0]
        num_filters = len(filters)

        if not input_shape and channels <= 0:
            raise AssertionError("Number of channels must be at least 1")
        if not filters:
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
        features_in = channels
        features_out = filters[0]

        kernel = 1
        stride = 1
        padding = 0

        if kernels:
            kernel = kernels[0]
        if strides:
            stride = strides[0]
        if paddings:
            padding = paddings[0]

        convos.append(nn.Conv1d(features_in, features_out, kernel, stride, padding=padding))

        if activations:
            convos.append(self.GetActivation(activations[0]))
        if pools:
            convos.append(nn.MaxPool1d(pools[0]))
        
        for i in range(1, len(filters)):
            features_in = features_out
            features_out = filters[i]

            kernel = 1
            stride = 1
            padding = 0

            if kernels:
                kernel = kernels[i]
            if strides:
                stride = strides[i]
            if paddings:
                padding = paddings[i]

            convos.append(nn.Conv1d(features_in, features_out, kernel, stride, padding=padding))

            if activations:
                convos.append(self.GetActivation(activations[i]))
            if pools:
                convos.append(nn.MaxPool1d(pools[i]))

        # TODO
        # if final_activation is not None:
        #     net.append(self.GetActivation(final_activation))

        if flatten:
            convos.append(Flatten())        
            
        # initialize weights
        for layer in convos:
            if type(layer) == nn.Conv1d:
                nn.init.xavier_uniform_(layer.weight)
            
        self._net = nn.Sequential(*convos)
        self.to(self._device)

        self._CalculateOutputSize()

    def forward(self, state):
        return self._net(state)
