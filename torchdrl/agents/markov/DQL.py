import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random

from .BaseAgent import BaseAgent

from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.representations.Plotter import Plotter

from torchdrl.tools.NeuralNetworkFactory import *


class DQL(BaseAgent):
    def __init__(self, env, **kwargs):
        super(DQL, self).__init__(env, **kwargs)
        self._epsilon = self._hyperparameters['epsilon']
        self._epsilon_decay = self._hyperparameters['epsilon_decay']
        self._epsilon_max = self._hyperparameters['epsilon']
        self._epsilon_min = self._hyperparameters['epsilon_min']

        self._net = self.CreateNetwork(self._hyperparameters['network'])
        self._net.train()

        self._target_net = self.CreateNetwork(self._hyperparameters['network'])
        self._target_net.eval()

        self.UpdateNetwork(self._net, self._target_net)

        optimizer = self._hyperparameters['optimizer']
        optimizer_kwargs = self._hyperparameters['optimizer_kwargs']
        self._optimizer = self.CreateOptimizer(
            optimizer, self._net.parameters(), optimizer_kwargs)

        print(self._net)

    def CreateNetwork(self, network_args):
        body, input_shape = self.CreateNetworkBody(network_args)

        net = None
        head = network_args['head']
        for key, values in network_args['head'].items():
            values['out_features'] = self._n_actions
            net = NetworkSelectionFactory(
                key, input_shape, values, body, device=self._device)

        return net

    def Evaluate(self, episodes=100):
        self._net.eval()

        for episode_info in super().Evaluate(episodes):
            yield episode_info

        self._net.train()

    def PlayEpisode(self, evaluate=False):
        episode_reward, steps, episode_loss, info = super().PlayEpisode(evaluate)

        info['epsilon'] = round(self._epsilon, 3)

        return episode_reward, steps, episode_loss, info

    @torch.no_grad()
    def Act(self, state):
        if random.random() < self._epsilon:
            action = self._env.action_space.sample()
        else:
            state_t = self.ConvertStateToTensor(state)

            q_values = self._net(state_t)
            action = torch.argmax(q_values).item()

        # linearly decrease epsilon
        self._epsilon = max(self._epsilon_min, self._epsilon -
                            (self._epsilon_max - self._epsilon_min) * self._epsilon_decay)

        return action

    def Learn(self):
        loss = self.Update()

        return loss.detach().cpu().numpy()

    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size, gamma):
        q_values = self._net(states_t).gather(1, actions_t.unsqueeze(1))
        next_q_values = self._target_net(next_states_t).max(
            dim=1, keepdim=True)[0].detach()
        mask = 1 - dones_t

        q_targets = (rewards_t + self._gamma * next_q_values *
                     (1-dones_t)).to(self._device)

        errors = F.smooth_l1_loss(q_values, q_targets)

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
        self._episode.load_state_dict(checkpoint['episode'])
        self._total_steps.load_state_dict(checkpoint['total_steps'])
