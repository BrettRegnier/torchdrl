# TODO make it generic

import torch.nn as nn
import torch.distributions as dis
import torch.nn.functional as F
import torch.optim as optim

from models.BaseNetwork import BaseNetwork

class TwoHeadedNetwork(BaseNetwork):
    def __init__(self, input_shape, n_actions, convo=False):
        super(TwoHeadedNetwork, self).__init__()		
        
        self._body = nn.Sequential(
            nn.Linear(*input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # self._fc1 = nn.Linear(*input_shape, 64)
        # self._fc2 = nn.Linear(64, 64)
        # self._head1 = nn.Linear(64, n_actions)
        # self._head2 = nn.Linear(64, n_actions)

        self._head1 = nn.Sequential(
            nn.Linear(64, n_actions),
            nn.Softmax(dim=1)
            )
            
        self._head2 = nn.Sequential(
            nn.Linear(64, n_actions),
            nn.Softmax(dim=1)
            )

    def forward(self, state):
        x = self._body(state)
        out1 = self._head1(x)
        out2 = self._head2(x)
        return out1, out2
