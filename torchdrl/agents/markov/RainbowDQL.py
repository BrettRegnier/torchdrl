import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import gym

import numpy as np
import random
import copy

from torchdrl.agents.markov.BaseAgent import BaseAgent

from torchdrl.neural_networks.ConvolutionNetwork1D import ConvolutionNetwork1D
from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.NoisyDuelingCategoricalNetwork import NoisyDuelingCategoricalNetwork
from torchdrl.neural_networks.CombineNetwork import CombineNetwork

from torchdrl.representations.Plotter import Plotter
from torchdrl.tools.NeuralNetworkFactory import *

import time
# DQN
# Double DQN
# Dueling DQN
# Prioritized Experience Replay
# Noisy net
# Categorical DQN
# N step

# TODO go back.


class RainbowDQL(BaseAgent):
    def __init__(self, env, oracle=None, **kwargs):
        super(RainbowDQL, self).__init__(env, oracle, **kwargs)
        self._v_min = self._hyperparameters['v_min']
        self._v_max = self._hyperparameters['v_max']
        self._atom_size = self._hyperparameters['atom_size']
        self._support = torch.linspace(
            self._v_min, self._v_max, self._atom_size).to(self._device)

        # double DQN
        self._net = self.CreateNetwork(self._hyperparameters['network'])
        self._net.train()

        self._target_net = self.CreateNetwork(self._hyperparameters['network'])
        self._target_net.eval()

        # self._optimizer = optim.SGD(self._net.parameters(), lr=self._hyperparameters['lr'], momentum=0.9)
        optimizer = self._hyperparameters['optimizer']
        optimizer_kwargs = self._hyperparameters['optimizer_kwargs']
        self._optimizer = self.CreateOptimizer(
            optimizer, self._net.parameters(), optimizer_kwargs)

        self.UpdateNetwork(self._net, self._target_net)
        print(self._net)

    def CreateNetwork(self, network_args):
        # dueling noisy
        body, input_shape = self.CreateNetworkBody(network_args)

        net = None
        head = network_args['head']
        for key, values in network_args['head'].items():
            values['atom_size'] = self._atom_size
            values['support'] = self._support
            values['out_features'] = self._n_actions
            net = NetworkSelectionFactory(key, input_shape, values, body, device=self._device)
        
        return net

    def Evaluate(self, episodes=100):
        self._net.eval()
        
        for episode_info in super().Evaluate(episodes):
            yield episode_info

        self._net.train()

    def PlayEpisode(self, evaluate=False):
        episode_reward, steps, episode_loss, info = super().PlayEpisode(evaluate)
            
        return episode_reward, steps, episode_loss, info

    @torch.no_grad()
    def Act(self, state):
        # noisy no epsilon
        states_t = self.ConvertStateToTensor(state)

        q_values = self._net(state_t)
        action = q_values.argmax().item()

        return action

    def Learn(self):
        loss = self.Update()
        
        # Noisy
        self._net.ResetNoise()
        self._target_net.ResetNoise()
        return loss.detach().cpu().numpy()

    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size, gamma):
        # categorical
        delta_z = float(self._v_max - self._v_min) / (self._atom_size - 1)

        with torch.no_grad():
            # double dqn
            next_actions = self._net(next_states_t).argmax(1)
            next_dists = self._target_net.DistributionForward(next_states_t)
            next_dists = next_dists[range(batch_size), next_actions]

            # categorical
            t_z = rewards_t + (1 - dones_t) * gamma * self._support
            t_z = t_z.clamp(min=self._v_min, max=self._v_max)
            b = (t_z - self._v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (batch_size - 1) *
                               self._atom_size, batch_size)
                .long()
                .unsqueeze(1)
                .expand(batch_size, self._atom_size)
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
        log_p = torch.log(dist[range(batch_size), actions_t])

        errors = -(proj_dist * log_p).sum(1)
        return errors

    def Save(self, folderpath, filename):
        super().Save(folderpath, filename)

        folderpath += "/" if folderpath[len(folderpath) - 1] != "/" else ""
        filepath = folderpath + filename

        torch.save({
            'name': self._name,
            'net_state_dict': self._net.state_dict(),
            'target_net_state_dict': self._target_net.state_dict(),
            'net_optimizer_state_dict': self._optimizer.state_dict(),
            'episode': self._episode,
            'total_steps': self._total_steps
        }, filepath)

    def Load(self, filepath):
        checkpoint = torch.load(filepath)

        self._net.load_state_dict(checkpoint['net_state_dict'])
        self._target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self._optimizer.load_state_dict(checkpoint['net_optimizer_state_dict'])
        self._episode = checkpoint['episode']
        self._total_steps = checkpoint['total_steps']
