import torch
import torch.nn as nn

class BaseNetwork(nn.Module):
    def AssertParameter(self, param, name, min_value=1):
        for x in param:
            l = x
            if type(l) is not tuple:
                l = (x,)
            
            for v in l:
                if type(v) is not int:
                    raise AssertionError(name + " invalid type" + str(type(x)) + " list must contain ints or int tuples")
                if type(v) is not int or v < min_value:
                    raise AssertionError(name + " cannot contain values less than 1")
    
    # TODO saving and loading
    def Save(self, path):
        pass
    
    def Load(self, path):
        pass

    def CopyModel(self, model):
        self.load_state_dict(model.state_dict())