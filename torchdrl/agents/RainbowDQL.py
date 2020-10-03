import torch
from torch.optim import Adam

from .DQL import DQL

from ..neural_networks.RainbowNetwork import RainbowNetwork

class RainbowDQL(DQL):
    def __init__(self, config):
        super(RainbowDQL, self).__init__(config)

        fcc = self._hyperparameters['fc']

        # TODO config variables
        self._vmin = -10
        self._vmax = 10
        self._atoms = 51

        self._net = RainbowNetwork(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._target_net = RainbowNetwork(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._net_optimizer = Adam(self._net.parameters(), lr=self._hyperparameters['lr'])

        self.UpdateNetwork(self._net, self._target_net)

    def Act(self, state):            
        state_t = torch.tensor(state, dtype=torch.float32, device=self._device).detach()
        state_t = state_t.unsqueeze(0)

        dist = self._net(state_t)
        dist = dist * torch.linspace(self._vmin, self._vmax, self._atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action

    def Learn(self):
        states_t, actions_t, next_states_t, rewards_t, dones_t = self.SampleMemoryT(self._batch_size)
        
        projection_distrib = self.ProjectionDistribution(next_states_t, rewards_t, dones_t)

        dist = self._net(states_t)
        actions_t = actions_t.unsqueeze(1).unsqueeze(1).expand(self._batch_size, 1, self._atoms)
        dist = dist.gather(1, actions_t).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(torch.tensor(projection_distrib) * dist.log()).sum(1)
        loss = loss.mean()

        self._net_optimizer.zero_grad()
        loss.backward()
        self._net_optimizer.step()

        self._net.ResetNoise()
        self._target_net.ResetNoise()

    def ProjectionDistribution(self, next_states_t, rewards_t, dones_t):
        batch_size = next_states_t.size(0)

        delta_z = float(self._vmax - self._vmin) / (self._atoms - 1)
        support = torch.linspace(self._vmin, self._vmax, self._atoms)

        next_dist = self._target_net(next_states_t).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))

        rewards_t = rewards_t.unsqueeze(1).expand_as(next_dist)
        dones_t = dones_t.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(1).expand_as(next_dist)

        Tz = rewards_t + (1 - dones) * self._gamma * support
        Tz = Tz.clamp(min=self._vmin, max=self._vmax)
        b = (Tz - self._vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self._atoms, batch_size).long().unsqueeze(1).expand(batch_size, self._atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist
