import torch
import torch.nn as nn
import numpy as np

# TODO add a parameter onto this such that it takes in a body as a parameter.
class BaseNetwork(nn.Module):
    def __init__(self, input_shape:tuple, body:list=None, device='cpu'):
        super(BaseNetwork, self).__init__()

        assert isinstance(input_shape, (list, tuple, np.ndarray))
        
        if body is not None:
            assert issubclass(type(body), BaseNetwork)
        self._body = body
        
        if not body:
            self._input_shape = input_shape
        else:
            self._input_shape = self._body.OutputSize

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

    def OutputSize(self):
        input_shape = self.InputShape()

        if type(input_shape) is list:
            in_features = []
            for shape in input_shape:
                in_features.append(torch.zeros(1, *shape, device=self._device))
        else:
            in_features = torch.zeros(1, *input_shape, device=self._device)   
        out = self.forward(in_features)
        return (int(np.prod(out.size())),)

    def InputShape(self):
        if self._body:
            # recursive call
            return self._body.InputShape()
        return self._input_shape