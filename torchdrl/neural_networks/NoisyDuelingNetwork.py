import torch.nn as nn

from .BaseNetwork import BaseNetwork
from .NoisyLinear import NoisyLinear

class NoisyDuelingNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activations:str, convo=None):
        super(NoisyDuelingNetwork, self).__init__(input_shape)

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

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activation) is not str and final_activation is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        num_hidden_layers = len(hidden_layers)

        if convo is None:
            self._convo = None
            in_features = [np.prod(input_shape)]
        else:
            self._convo = ConvolutionNetwork(input_shape, convo['filters'], convo['kernels'], convo['strides'], convo['paddings'], convo['activations'], convo['pools'], convo['flatten'])
            in_features = self._convo.OutputSize()

        net = []
        net.append(nn.Linear(*in_features, hidden_layers[0]))
        if len(activations) > 0 and activations[0] is not None:
            net.append(self.GetActivation(activations[0]))
        if len(dropouts) > 0 and dropouts[0] is not None:
            net.append(nn.Dropout(dropouts[0]))

        i = 0
        if num_hidden_layers > 1:
            for i in range(1, num_hidden_layers):
                net.append(NoisyLinear(hidden_layers[i-1], hidden_layers[i]))
                if len(activations) > i and activations[i] is not None:
                        net.append(self.GetActivation(activations[i]))
                if len(dropouts) > i and dropouts[i] is not None:
                        net.append(nn.Dropout(dropouts[i]))

        net.append(NoisyLinear(hidden_layers[i], n_actions))
        if final_activation is not None:
            net.append(self.GetActivation(final_activation))
            
        # initialize weights
        for layer in net:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)

        self._net = nn.Sequential(*net)

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

    def ResetNoise(self):
        for net in self._noisy_value:
            if type(net) == NoisyLinear:
                net.ResetNoise()

        for net in self._noisy_advantage:
            if type(net) == NoisyLinear:
                net.ResetNoise()