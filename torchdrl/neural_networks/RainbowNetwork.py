import torch.nn as nn
import torch.nn.functional as F

from .BaseNetwork import BaseNetwork
from .FullyConnectedNetwork import FullyConnectedNetwork
from .NoisyLinear import NoisyLinear

# retrieved from https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb
class RainbowNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activations:str, convo=None):
        super(RainbowNetwork, self).__init__(input_shape)

        self._n_actions = n_actions

        # TODO make these into variables
        self._vmin = -10
        self._vmax = 10
        self._num_atoms = 51

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activations) is not str and final_activations is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        self._net = nn.Sequential(
            nn.Linear(n_actions, 1024),
            nn.ReLU(),
            nn.Linear(n_actions, 1024),
            nn.ReLU(),
        )

        self._noisy_value = nn.Sequential(
            NoisyLinear(1024, 1024),
            nn.ReLU(),
            NoisyLinear(1024, self._num_atoms)
        )

        self._noisy_advantage = nn.Sequential(
            NoisyLinear(1024, 1024),
            nn.ReLU(),
            NoisyLinear(1024, self._num_atoms * self._n_actions)
        )


        # TODO dynamic
        # last_layer = (hidden_layers[-1:])[0]
        # last_activation = (activations[-1:])[0]
        # self._net = FullyConnectedNetwork(input_shape, last_layer, hidden_layers, activations, dropouts, last_activation, convo)

        # self._adv = nn.Sequential(
        #     nn.Linear(last_layer, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, n_actions)
        # )
        # self._val = nn.Sequential(
        #     nn.Linear(last_layer, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )
        
    def forward(self, state):
        batch_size = x.size(0)

        x = self._net(x)

        val = self._noisy_value(x)
        adv = self._noisy_advantage(x)

        val = val.view(batch_size, 1, self._num_atoms)
        adv = adv.view(batch_size, self._n_actions, self._num_atoms)

        x = val + adv - adv.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self._num_atoms)).view(-1, self._n_actions, self._num_atoms)

        return x 

    def ResetNoise(self):
        for net in self._noisy_value:
            if type(net) == NoisyLinear:
                net.ResetNoise()

        for net in self._noisy_advantage:
            if type(net) == NoisyLinear:
                net.ResetNoise()
