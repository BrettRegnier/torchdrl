import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from .DQL import DQL

from ..neural_networks.RainbowNetwork import RainbowNetwork
from ..data_structures.PrioritizedExperienceReplay import PrioritizedExperienceReplay

class RainbowDQL(DQL):
    def __init__(self, config):
        super(RainbowDQL, self).__init__(config)

        fcc = self._hyperparameters['fc']

        # TODO config variables
        self._vmin = -10
        self._vmax = 10
        self._atoms = 51
        self._support = torch.linspace(self._vmin, self._vmax, self._atoms).to(self._device)
        self._delta_z = (self._vmax - self._vmin) / (self._atoms - 1)
        self._norm_clip = 10
        # TODO multistep
        self._multi_step = 1

        self._memory = PrioritizedExperienceReplay(config['memory_size'])

        self._net = RainbowNetwork(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._target_net = RainbowNetwork(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._net_optimizer = Adam(self._net.parameters(), lr=self._hyperparameters['lr'])

        self.UpdateNetwork(self._net, self._target_net)

        self._net.train()
        self._target_net.train()
        for param in self._target_net.parameters(): 
            param.requires_grad = False
        

    @torch.no_grad()
    def Act(self, state):            
        state_t = torch.tensor(state, dtype=torch.float32, device=self._device).detach()
        state_t = state_t.unsqueeze(0)

        dist = self._net(state_t)
        dist = dist * self._support
        action = dist.sum(2).argmax(1).item()
        return action

    def Learn(self):
        states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t = self.SampleMemoryT(self._batch_size)
        
        log_ps = self._net(states_t, log=True)
        log_ps_a = log_ps[range(self._batch_size), actions_t] # select the actions based on the batch_size

        with torch.no_grad():
            next_state_probs = self._net(next_states_t)
            next_state_distrib = self._support.expand_as(next_state_probs) * next_state_probs
            next_state_argmax = next_state_distrib.sum(2).argmax(1)

            self._target_net.ResetNoise()
            next_state_target_probs = self._target_net(next_states_t)
            next_state_q_probs = next_state_target_probs[range(self._batch_size), next_state_argmax]

            # compute Tz
            Tz = rewards_t.unsqueeze(1) + (1 - dones_t) * (self._gamma ** self._multi_step)
            Tz = Tz.clamp(min=self._vmin, max=self._vmax)

            b = (Tz - self._vmin) / self._delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            l[(u > 0) * (l == u)] -= 1
            u[(l < (self._atoms - 1)) * (l == u)] += 1

            # distribute probabily of Tz
            m = states_t.new_zeros(self._batch_size, self._atoms)
            offset = torch.linspace(0, ((self._batch_size - 1) * self._atoms), self._batch_size).unsqueeze(1).expand(self._batch_size, self._atoms).to(actions_t)
            # m.view(-1).index_add_(0, (l + offset).view(-1), (next_state_q_probs * (u.float() - b)).view(-1))
            # m.view(-1).index_add_(0, (u + offset).view(-1), (next_state_q_probs * (b - l.float())).view(-1))

        # change to l1 smooth?
        loss = -torch.sum(m * log_ps_a, 1)

        self._net_optimizer.zero_grad()
        (weights_t * loss).mean().backward()
        clip_grad_norm_(self._net.parameters(), self._norm_clip)
        self._net_optimizer.step()
        
        errors = loss.detach().cpu().numpy()
        self._memory.BatchUpdate(indices_np, errors)

    def ProjectionDistribution(self, next_states_t, rewards_t, dones_t):
        batch_size = next_states_t.size(0)


        next_dist = self._target_net(next_states_t).data * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))

        rewards_t = rewards_t.unsqueeze(1).unsqueeze(1).expand_as(next_dist)
        dones_t = dones_t.unsqueeze(1).unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards_t + (1 - dones_t) * self._gamma * support
        Tz = Tz.clamp(min=self._vmin, max=self._vmax)
        b = (Tz - self._vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self._atoms, batch_size, dtype=torch.int64, device=self._device).unsqueeze(1).unsqueeze(1)
        offset = offset.expand_as(next_dist)

        proj_dist = torch.zeros(next_dist.size(), device=self._device)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist
