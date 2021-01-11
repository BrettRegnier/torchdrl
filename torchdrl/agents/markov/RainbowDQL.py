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
    def __init__(self, env, **kwargs):
        super(RainbowDQL, self).__init__(env, **kwargs)
        self._target_update = self._hyperparameters['target_update']
        self._target_update_steps = 0

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
        ndcn = network_args['noisyduelingcategorical']

        input_shape = self._env.observation_space
        networks = []
        body = None
        if isinstance(input_shape, (gym.spaces.Tuple, gym.spaces.Dict)):
            if 'group' in network_args:
                for i, (key, values) in enumerate(network_args['group'].items()):
                    networks.append(NetworkSelectionFactory(
                        key, input_shape[i].shape, values, device=self._device))
                body = CombineNetwork(networks, self._device)
                input_shape = body.OutputSize()
            else:
                raise Exception(
                    "gym tuple/dict detected, requires a grouping of networks")
        else:
            input_shape = input_shape.shape

        if 'sequential' in network_args:
            for i, (key, values) in enumerate(network_args['sequential'].items()):
                body = NetworkSelectionFactory(
                    key, input_shape, values, body, device=self._device)
                input_shape = body.OutputSize()

        return NoisyDuelingCategoricalNetwork(self._atom_size, self._support, input_shape, self._n_actions,
                                              ndcn["hidden_layers"], ndcn['activations'], ndcn['dropouts'], ndcn['final_activation'], body=body, device=self._device)

    def Evaluate(self, episodes=100):
        self._net.eval()
        for episode_info in super().Evaluate(episodes):
            yield episode_info

    def PlayEpisode(self, evaluate=False):
        self._steps = 0
        done = False
        episode_reward = 0
        episode_loss = 0

        state = self._env.reset()
        while self._steps != self._max_steps and not done:
            # print(self._total_steps, state, len(self._memory))
            # Noisy - No epsilon
            action = self.Act(state)

            next_state, reward, done, info = self._env.step(action)

            if not evaluate:
                transition = (state, action, next_state, reward, done)

                # if (self._total_steps == 28):
                #     x =  True
                self.SaveMemory(transition)

                if len(self._memory) >= self._batch_size:
                    episode_loss += self.Learn()

            episode_reward += reward
            state = next_state

            self._steps += 1
            self._total_steps += 1

        return episode_reward, self._steps, episode_loss, info

    @torch.no_grad()
    def Act(self, state):
        # Noisy - No random action by dqn
        if isinstance(state, object) and not isinstance(state, np.ndarray):
            states_t = []
            for val in state:
                state_t = torch.tensor(val, dtype=torch.float32,
                               device=self._device).unsqueeze(0).detach()
                states_t.append(state_t)
        else:
            states_t = torch.tensor(state, dtype=torch.float32,
                                device=self._device).detach()
            states_t = states_t.unsqueeze(0)

        q_values = self._net(states_t)
        action = q_values.argmax().item()

        return action

    def SaveMemory(self, transition):
        if self._memory_n_step:
            transition = self._memory_n_step.Append(*transition)

        if transition:
            self._memory.Append(*transition)

    def Learn(self):
        states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t = self.SampleMemoryT(
            self._batch_size)

        # get errors
        errors = self.CalculateErrors(states_t, actions_t, next_states_t, rewards_t,
                                      dones_t, indices_np, weights_t, self._batch_size, self._gamma)

        weights_t = weights_t.reshape(-1, 1)
        # Prioritized Experience Replay weight importancing
        loss = torch.mean(errors * weights_t)

        # n-step learning with one-step to prevent high-variance
        if self._memory_n_step:
            gamma = self._gamma ** self._n_steps
            states_np, actions_np, next_states_np, rewards_np, dones_np, _ = self._memory_n_step.SampleBatchFromIndices(
                indices_np)
            states_t, actions_t, next_states_t, rewards_t, dones_t = self.ConvertNPMemoryToTensor(
                states_np, actions_np, next_states_np, rewards_np, dones_np)
            errors_n = self.CalculateErrors(
                states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, self._batch_size, gamma)
            errors += errors_n

            loss = torch.mean(errors * weights_t)

        self._optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self._net.parameters(), 10.0)
        self._optimizer.step()

        # TODO change if needed
        # hard update target for now
        if self._target_update_steps % self._target_update == 0:
            self._target_net.load_state_dict(self._net.state_dict())
            self._target_update_steps = 0

        self._target_update_steps += 1

        # Noisy
        self._net.ResetNoise()
        self._target_net.ResetNoise()

        # Prioritized Experience Replay
        updated_priorities = errors.detach().cpu().numpy()
        # print(indices_np)
        # if self._total_steps == 35:
        #     exit()
        self._memory.BatchUpdate(indices_np, updated_priorities)

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
        # print(errors)
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
