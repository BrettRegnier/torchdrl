import torch.nn as nn
import torch.nn.functional as F

from .BaseNetwork import BaseNetwork
from .FullyConnectedNetwork import FullyConnectedNetwork
from .ConvolutionNetwork import ConvolutionNetwork
from .NoisyLinear import NoisyLinear

# retrieved from https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb
class RainbowNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activations:str, convo=None):
        super(RainbowNetwork, self).__init__(input_shape)

        self._n_actions = n_actions

        # TODO make these into variables
        # TODO add std as config
        self._vmin = -10
        self._vmax = 10
        self._atoms = 51

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activations) is not str and final_activations is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        self._convo = ConvolutionNetwork(input_shape, convo['filters'], convo['kernels'], convo['strides'], convo['paddings'], convo['activations'], convo['pools'], convo['flatten'])
        in_features = self._convo.OutputSize()

        self._net = nn.Sequential(
            nn.Linear(*in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self._noisy_value = nn.Sequential(
            NoisyLinear(1024, 1024),
            nn.ReLU(),
            NoisyLinear(1024, self._atoms)
        )

        self._noisy_advantage = nn.Sequential(
            NoisyLinear(1024, 1024),
            nn.ReLU(),
            NoisyLinear(1024, self._atoms * self._n_actions)
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
        
    def forward(self, state, log=False):
        x = self._convo(state)
        x = self._net(x)

        val = self._noisy_value(x)
        adv = self._noisy_advantage(x)

        val = val.view(-1, 1, self._atoms)
        adv = adv.view(-1, self._n_actions, self._atoms)

        q = val + adv - adv.mean(1, keepdim=True)
        
        if log:
            q = F.log_softmax(q, dim=2)
        else:
            q = F.softmax(q, dim=2)
        
        return q

    def ResetNoise(self):
        for net in self._noisy_value:
            if type(net) == NoisyLinear:
                net.ResetNoise()

        for net in self._noisy_advantage:
            if type(net) == NoisyLinear:
                net.ResetNoise()
