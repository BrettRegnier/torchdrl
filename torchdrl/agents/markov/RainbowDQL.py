import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import gym

import numpy as np
import random
import copy

import torchdrl.tools.Helper as Helper

from torchdrl.agents.markov.MarkovAgent import MarkovAgent

from torchdrl.neural_networks.ConvolutionNetwork1D import ConvolutionNetwork1D
from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.NoisyDuelingCategoricalNetwork import NoisyDuelingCategoricalNetwork
from torchdrl.neural_networks.CombineNetwork import CombineNetwork

from torchdrl.representations.Plotter import Plotter

import time

# DQN
# Double DQN
# Dueling DQN
# Prioritized Experience Replay
# Noisy net
# Categorical DQN
# N step
class RainbowDQL(MarkovAgent):
    def __init__(self, *args, **kwargs):
        super(RainbowDQL, self).__init__(*args, **kwargs)

        get_atom_size_op = getattr(self._model, "GetAtomSize", None)
        if callable(get_atom_size_op):
            self._atom_size = get_atom_size_op()
        else:
            raise Exception("Provided network must have a function called GetAtomSize that returns the number of atoms")

        get_support_bounds_op = getattr(self._model, "GetSupportBounds", None)
        if callable(get_support_bounds_op):
            self._v_min, self._v_max = get_support_bounds_op()
        else:
            raise Exception("Provided network must have a function called GetSupportBounds that return tuple (value min, value max)")

        get_support_op = getattr(self._model, "GetSupport", None)
        if callable(get_support_op):
            self._support = get_support_op()
        else:
            raise Exception("Provided network must have a function called GetSupport that returns a torch tensor support (linspace)")

    def Evaluate(self, episodes=100):
        self._model.eval()
        
        for episode_info in super().Evaluate(episodes):
            yield episode_info

        self._model.train()

    @torch.no_grad()
    def Act(self, state, evaluate=False):
        # noisy no epsilon
        state_t = Helper.ConvertStateToTensor(state, self._device)

        q_values = self._model(state_t)
        action = q_values.argmax().item()

        return action

    def Learn(self):
        if len(self._memory) >= self._batch_size:
            loss = self.Update()
            
            # Noisy
            self._model.ResetNoise()
            self._target_model.ResetNoise()
        
            return loss.detach().cpu().numpy()
        return 0

    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size, gamma):
        # categorical
        delta_z = float(self._v_max - self._v_min) / (self._atom_size - 1)

        with torch.no_grad():
            # double dqn
            next_actions = self._model(next_states_t).argmax(1)
            next_dists = self._target_model.DistributionForward(next_states_t)
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

        dist = self._model.DistributionForward(states_t)
        log_p = torch.log(dist[range(batch_size), actions_t])

        errors = -(proj_dist * log_p).sum(1)
        return errors

    def Save(self, folderpath, filename):
        agent_dict = {
            'name': self._name,
            'net_state_dict': self._model.state_dict(),
            'target_net_state_dict': self._target_model.state_dict(),
            'net_optimizer_state_dict': self._optimizer.state_dict(),
            'episode': self._episode,
            'total_steps': self._total_steps
        }

        Helper.SaveAgent(folderpath, filename, agent_dict)

    def Load(self, filepath):
        checkpoint = Helper.LoadAgent(filepath)

        self._model.load_state_dict(checkpoint['net_state_dict'])
        self._target_model.load_state_dict(checkpoint['target_net_state_dict'])
        self._optimizer.load_state_dict(checkpoint['net_optimizer_state_dict'])
        self._episode = checkpoint['episode']
        self._total_steps = checkpoint['total_steps']
