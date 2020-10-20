import torch.nn as nn

from .BaseNetwork import BaseNetwork
from .FullyConnectedNetwork import FullyConnectedNetwork

class DuelingNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activations:str, convo=None):
        super(DuelingNetwork, self).__init__(input_shape)

        self._n_actions = n_actions

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activations) is not str and final_activations is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        # scrape one layer off for the advantage and value layers
        prev_layer = hidden_layers[-2:][0]
        last_layer = hidden_layers[-1:][0]
        hidden_layers = hidden_layers[-1:]

        last_activation = (activations[-1:])[0]
        activations= activations[-1:]

        self._net = FullyConnectedNetwork(input_shape, last_layer, hidden_layers, activations, dropouts, last_activation, convo)

        self._adv = nn.Sequential(
            nn.Linear(prev_layer, last_layer),
            nn.ReLU(),
            nn.Linear(last_layer, n_actions)
        )
        self._val = nn.Sequential(
            nn.Linear(prev_layer, last_layer),
            nn.ReLU(),
            nn.Linear(last_layer, 1)
        )
        
    def forward(self, state):
        x = self._net(state)

        val = self._val(x)
        adv = self._adv(x)

        q = val + adv - adv.mean(dim=-1, keepdim=True)

        return q


