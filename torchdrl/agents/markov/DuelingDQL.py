from torch.optim import Adam

from .DQL import DQL

from torchdrl.neural_networks.DuelingNetwork import DuelingNetwork

class DuelingDQL(DQL):
    def __init__(self, config):
        super(DuelingDQL, self).__init__(config)

        fcc = self._hyperparameters['fc']
        self._net = DuelingNetwork(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._target_net = DuelingNetwork(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._net_optimizer = Adam(self._net.parameters(), lr=self._hyperparameters['lr'])

        self.UpdateNetwork(self._net, self._target_net)