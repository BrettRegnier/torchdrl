# TODO make it generic

import torch.nn as nn
import torch.distributions as dis
import torch.nn.functional as F
import torch.optim as optim

from models.BaseNetwork import BaseNetwork

class FullyConnectedNetwork(BaseNetwork):
    def __init__(self, input_shape, n_actions, last_layer_activation=None, convo=False):
        super(FullyConnectedNetwork, self).__init__()		

        self._body = nn.Sequential(
            nn.Linear(*input_shape, 64),
            # nn.ReLU(),
            nn.Linear(64, 64),
            # nn.ReLU()
        )

        head = [nn.Linear(64, n_actions)]
        if last_layer_activation is not None:
            head.append(last_layer_activation)

        self._head = nn.Sequential(*head)
                


    def forward(self, state):
        x = self._body(state)
        x = self._head(x)
        return x
