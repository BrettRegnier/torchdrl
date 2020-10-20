import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random

from .BaseAgent import BaseAgent

from ..neural_networks.NoisyDuelingCategoricalNetwork import NoisyDuelingCategoricalNetwork
from ..representations.Plotter import Plotter

from ..data_structures.NStepPrioritizedExperienceReplay import NStepPrioritizedExperienceReplay

# Followed guide from https://github.com/Curt-Park/rainbow-is-all-you-need

# DQN
# Double DQN
# Dueling DQN
# Prioritized Experience Replay
# Noisy net
# Categorical DQN
# N step

class RainbowDQL(BaseAgent):
    def __init__(self, config):
        super(RainbowDQL, self).__init__(config)
        self._gamma = self._hyperparameters['gamma']

        self._tau = self._hyperparameters['tau']
        self._target_update_steps = 0 
        self._target_update_frequency = self._hyperparameters['target_update']

        self._n_steps = self._hyperparameters['n_steps']
        self._v_min = self._hyperparameters['v_min']
        self._v_max = self._hyperparameters['v_max']
        self._atom_size = self._hyperparameters['atom_size']
        self._support = torch.linspace(self._v_min, self._v_max, self._atom_size).to(self._device)

        self._alpha = self._hyperparameters['alpha']
        self._beta = self._hyperparameters['beta']
        self._priority_epsilon = self._hyperparameters['priority_epsilon']

        self._apex = self._hyperparameters['apex']
        self._batch_size = self._hyperparameters['batch_size']
        memory_type = self._hyperparameters['memory_type']
        memory_size = self._hyperparameters['memory_size']
        if self._apex:
            if 'memory' not in config:
                raise AssertionError("a shared memory needs to be provided for apex learning")
            else:
                self._memory = config['memory']
                self._apex_mini_batch = 64 # TODO parameter
                self._internal_memory = UniformExperienceReplay(self._apex_mini_batch * 4, self._input_shape)
        else:
            if self._hyperparameters['memory_type'] == "PER":
                self._memory = NStepPrioritizedExperienceReplay(memory_size, self._input_shape, self._alpha, self._beta, self._priority_epsilon)
                if self._n_steps > 1:
                    self._memory_n_step = NStepPrioritizedExperienceReplay(memory_size, self._input_shape, self._alpha, self._beta, self._priority_epsilon, self._n_steps, self._gamma)
            else:
                self._memory = UniformExperienceReplay(config['memory_size'], self._input_shape)
                # TODO make uniform experience replay work with rainbow
                if self._n_steps > 1:
                    self._memory_n_step = NStepPrioritizedExperienceReplay(self._hyperparameters['memory_size'], self._alpha, self._input_shape, self._n_steps, self._gamma)

        fcc = self._hyperparameters['fc']
        # and dueling noisy
        self._net = NoisyDuelingCategoricalNetwork(self._atom_size, self._support, self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)

        # double DQN
        self._target_net = NoisyDuelingCategoricalNetwork(self._atom_size, self._support, self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._net_optimizer = Adam(self._net.parameters(), lr=self._hyperparameters['lr'])

        self.UpdateNetwork(self._net, self._target_net)
    
    def PlayEpisode(self, evaluate=False):
        done = False
        steps = 0
        episode_reward = 0
        
        state = self._env.reset()
        while steps != self._max_steps and not done:
            # Noisy - No epsilon
            action = self.Act(state)
                
            next_state, reward, done, info = self._env.step(action)
            transition = (state, action, next_state, reward, done)

            if self._apex:
                self._internal_memory.Append(state, action, next_state, reward, done)
                self.ApexSendMemories()
            else:
                if self._n_steps > 1:
                    # Do n_step transition
                    transition = self._memory_n_step.Append(*transition)

                if transition:
                    self._memory.Append(*transition)
                
                if len(self._memory) > self._batch_size:
                    self.Learn()
            
            episode_reward += reward
            state = next_state

            steps += 1
            self._total_steps += 1
            
        return episode_reward, steps, info

    @torch.no_grad()
    def Act(self, state):
        # Noisy - No random action by dqn
        state_t = torch.tensor(state, dtype=torch.float32, device=self._device).detach()
        state_t = state_t.unsqueeze(0)
        
        q_values = self._net(state_t)
        action = torch.argmax(q_values).item()

        return action
    
    def Learn(self):
        states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t = self.SampleMemoryT(self._batch_size)

        # get errors 
        errors = self.CalculateErrors(states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, self._batch_size, self._gamma)

        weights_t = weights_t.reshape(-1, 1)
        # Prioritized Experience Replay weight importancing
        loss = torch.mean(errors * weights_t)

        # n-step learning with one-step to prevent high-variance
        if self._n_steps > 1:
            gamma = self._gamma ** self._n_steps
            samples = self._memory_n_step.SampleBatchFromIndices(indices_np)
            states_t, actions_t, next_states_t, rewards_t, dones_t = self.ConvertNPMemoryToTensor(*samples)
            errors_n = self.CalculateErrors(states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, self._batch_size, gamma)
            errors += errors_n

            loss = torch.mean(errors * weights_t)

        self._net_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self._net.parameters(), 10.0)
        self._net_optimizer.step()

        # TODO change if needed
        # hard update target for now
        if self._target_update_steps % self._target_update_frequency == 0:
            self._target_net.load_state_dict(self._net.state_dict())
            self._target_update_steps = 0

        self._target_update_steps += 1

        # Noisy
        self._net.ResetNoise()
        self._target_net.ResetNoise()
        
        # Prioritized Experience Replay
        updated_priorities = errors.detach().cpu().numpy()
        self._memory.BatchUpdate(indices_np, updated_priorities)
        
    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size, gamma):
        # categorical
        delta_z = float(self._v_max - self._v_min) / (self._atom_size - 1)

        with torch.no_grad():
            # double dqn
            next_actions = self._net(next_states_t).argmax(1)
            next_dists = self._target_net.DistributionForward(next_states_t)
            next_dists = next_dists[range(self._batch_size), next_actions]

            # categorical
            t_z = rewards_t + (1 - dones_t) * gamma * self._support
            t_z = t_z.clamp(min=self._v_min, max=self._v_max)
            b = (t_z - self._v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (self._batch_size - 1) * self._atom_size, self._batch_size)
                .long()
                .unsqueeze(1)
                .expand(self._batch_size, self._atom_size)
                .to(self._device)
            )

            proj_dist = torch.zeros(next_dists.size(), device=self._device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dists * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dists * (b - l.float())).view(-1)
            )

        dist = self._net.DistributionForward(states_t)
        log_p = torch.log(dist[range(self._batch_size), actions_t])

        errors = -(proj_dist * log_p).sum(1)
        return errors

    def Save(self, filepath="checkpoints"):
        filepath += "/" + self._config['name']
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        filepath += "/episode_" + str(self._episode) + "_score_" + str(round(self._episode_mean_score, 2)) + "_dql.pt"

        torch.save({
            'net_state_dict': self._net.state_dict(),
            'target_net_state_dict': self._target_net.state_dict(),
            'net_optimizer_state_dict': self._net_optimizer.state_dict(),
            'episode': self._episode,
            'total_steps': self._total_steps
        }, filepath)

    def Load(self, filepath):
        checkpoint = torch.load(filepath)

        self._net.load_state_dict(checkpoint['net_state_dict'])
        self._target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self._net_optimizer.load_state_dict(checkpoint['net_optimizer_state_dict'])
        self._episode.load_state_dict(checkpoint['episode'])
        self._total_steps.load_state_dict(checkpoint['total_steps'])
        