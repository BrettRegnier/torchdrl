import torch
import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, input_shape:tuple):
        super(BaseNetwork, self).__init__()
        if type(input_shape) is not tuple:
            raise AssertionError("Input shape must be of type tuple")

    def AssertParameter(self, param, name, dtype, min_value=1):
        for x in param:
            l = x
            if type(l) is not tuple:
                l = (x,)
            
            for v in l:
                if type(v) is not dtype:
                    raise AssertionError(name + " invalid type" + str(type(x)) + " list must contain ints or int tuples")
                if min_value > 0 and v < min_value:
                    raise AssertionError((name + " cannot contain values less than %d") % min_value)
    

    def GetActivation(self, activation:str):
        ac = activation.lower()
        if ac == "relu":
            return nn.ReLU()
        elif ac == "sigmoid":
            return nn.Sigmoid()
        elif ac == "tanh":
            return nn.Tanh()
        elif ac == "hardtanh":
            return nn.Hardtanh
        else:
            raise AssertionError("No recognized activation: " + activation)

    # TODO saving and loading
    def Save(self, path):
        pass
    
    def Load(self, path):
        pass

    def CopyModel(self, model):
        self.load_state_dict(model.state_dict())