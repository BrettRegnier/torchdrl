import torch
import torch.nn as nn
import numpy as np

# TODO add a parameter onto this such that it takes in a body as a parameter.
class BaseNetwork(nn.Module):
    def __init__(self, input_shape:tuple, body=None, device='cpu'):
        super(BaseNetwork, self).__init__()
        if body is not None and not issubclass(type(body), BaseNetwork):
            raise AssertionError("Body must be of type BaseNetwork")
        if type(input_shape) is not tuple:
            raise AssertionError("Input shape must be of type tuple")
        self._input_shape = input_shape
        self._body = body
        self._device = device

    def AssertParameter(self, param, name, dtype, min_value=1):
        for x in param:
            l = x
            if type(l) is not tuple and type(l) is not list:
                l = (x,)
            
            for v in l:
                if type(v) is not dtype:
                    raise AssertionError(name + " invalid type" + str(x) + " list must contain " + str(dtype))
                if min_value > 0 and v < min_value:
                    raise AssertionError((name + " cannot contain values less than %d") % min_value)
    
    def GetActivation(self, activation:str):
        ac = activation.lower()
        if ac == "relu":
            return nn.ReLU()
        elif ac == "leakyrelu":
            return nn.LeakyReLU()
        elif ac == "softmax":
            return nn.Softmax(dim=1)
        elif ac == "sigmoid":
            return nn.Sigmoid()
        elif ac == "tanh":
            return nn.Tanh()
        elif ac == "hardtanh":
            return nn.Hardtanh()
        else:
            raise AssertionError("No recognized activation: " + activation)

    def CopyModel(self, model):
        self.load_state_dict(model.state_dict())

    def NetList(self):
        return self._net_list

    @property
    def OutputSize(self):
        return self._output_size
    
    def _CalculateOutputSize(self):
        input_shape = self._InputShape()

        out = self.forward(torch.zeros(1, *input_shape, device=self._device))
        self._output_size = (int(np.prod(out.size())),)

    def _InputShape(self):
        if self._body:
            return self._body._InputShape()
        return self._input_shape