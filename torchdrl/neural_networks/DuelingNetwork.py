import torch.nn as nn

from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.BaseNetwork import BaseNetwork


# TODO update this
class DuelingNetwork(BaseNetwork):
    def __init__(self, input_shape:tuple, n_actions:int, hidden_layers:list, activations:list, dropouts:list, final_activations:str, convo=None):
        super(DuelingNetwork, self).__init__(input_shape)

        if type(n_actions) is not int:
            raise AssertionError("n_actions must be of type int")
        if type(final_activations) is not str and final_activations is not None:
            raise AssertionError("Last activation must be of type str")

        self.AssertParameter(hidden_layers, "hidden_layers", int)
        self.AssertParameter(activations, "activations", str, -1)
        self.AssertParameter(dropouts, "dropouts", float, 0)

        adv_net = self.CreateNetList(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation)
        val_net = self.CreateNetList(input_shape, 1, hidden_layers, activations, dropouts, final_activation)

        self._adv = nn.Sequential(adv_net)
        self._val = nn.Sequential(val_net)

    def CreateNetList(self, input_shape, n_actions, hidden_layers, activations, dropouts, final_activation):
        net = []

        in_features = int(np.prod(input_shape))
        if hidden_layers:
            out_features = hidden_layers[0]

            net.append(NoisyLinear(in_features, out_features))
            if activations:
                net.append(self.GetActivation(activations[0]))
            if dropouts:
                net.append(nn.Dropout(dropouts[0]))

            for i in range(1, len(hidden_layers)):
                in_features = out_features
                out_features = hidden_layers[i]

                net.append(NoisyLinear(in_features, out_features))
                if activations and len(activations) > i:
                    net.append(self.GetActivation(activations[i]))
                if dropouts and len(dropouts) > i:
                    net.append(nn.Dropout(dropouts[i]))
            in_features = out_features
            
        out_features = n_actions
        net.append(NoisyLinear(in_features, out_features))

        if final_activation is not None:
            net.append(self.GetActivation(final_activation))

        return net

    def forward(self, state):
        x = self._net(state)

        val = self._val(x)
        adv = self._adv(x)

        q = val + adv - adv.mean(dim=-1, keepdim=True)

        return q


