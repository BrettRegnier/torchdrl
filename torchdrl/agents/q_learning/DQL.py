import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random

import torchdrl.tools.Helper as Helper

from torchdrl.agents.q_learning.QLearningAgent import QLearningAgent

from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.representations.Plotter import Plotter

class DQL(QLearningAgent):
    def __init__(self, *args, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, **kwargs):
        super(DQL, self).__init__(*args, **kwargs)

        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_max = epsilon
        self._epsilon_min = epsilon_min

    def Evaluate(self, episodes=100):
        self._model.eval()

        for episode_info in super().Evaluate(episodes):
            yield episode_info

        self._model.train()

    @torch.no_grad()
    def Act(self, state, evaluate=False):    
        if random.random() < self._epsilon and not evaluate:
            action = self._env.action_space.sample()
        else:
            state_t = Helper.ConvertStateToTensor(state, self._device)

            q_values = self._model(state_t)
            action = torch.argmax(q_values).item()

        self._ep_info.update({"epsilon": round(self._epsilon, 3)})

        return action

    def Learn(self):
        if len(self._memory) >= self._batch_size:
            loss = self.Update()

            # linearly decrease epsilon
            self._epsilon = max(self._epsilon_min, self._epsilon -
                                (self._epsilon_max - self._epsilon_min) * self._epsilon_decay)
            
            return loss.detach().cpu().numpy()
        return 0

    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size, gamma):
        q_values = self._model(states_t).gather(1, actions_t.unsqueeze(1))
        
        next_q_values = self._target_model(next_states_t).max(
            dim=1, keepdim=True)[0].detach()

        q_targets = (rewards_t + self._gamma * next_q_values *
                     (1-dones_t)).to(self._device)

        errors = F.smooth_l1_loss(q_values, q_targets, reduction="none")

        return errors

    def Save(self, folderpath, filename):
        self._save_info = {
            'name': self._name,
            'model': self._model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'architecture': self._model.__str__(),
            'optimizer': self._optimizer.state_dict(),
            'episode': self._episode,
            'epsilon': self._epsilon,
            'total_steps': self._total_steps,
            'avg_loss': self._avg_loss,
            'avg_test_score': self._avg_test_score
        }
        
        if self._scheduler:
            self._save_info['sched_step_size'] = self._scheduler.state_dict()['step_size']
            self._save_info['sched_gamma'] = self._scheduler.state_dict()['gamma']
            self._save_info['scheduler'] = self._scheduler.state_dict()
            self._save_info['learning_rate'] = self._optimizer.state_dict()['param_groups'][0]['initial_lr']
        else:
            self._save_info['learning_rate'] = self._optimizer.state_dict()['param_groups'][0]['lr']

        Helper.SaveAgent(folderpath, filename, self._save_info)

    def Load(self, filepath):
        checkpoint = Helper.LoadAgent(filepath)

        self._name = checkpoint['name']
        self._model.load_state_dict(checkpoint['model'])
        self._target_model.load_state_dict(checkpoint['target_model'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        self._episode = checkpoint['episode']
        self._epsilon = checkpoint['epsilon']
        self._total_steps = checkpoint['total_steps']
        self._avg_loss = checkpoint['avg_loss']
        self._avg_test_score = checkpoint['avg_test_score']

        # TODO add in loading of scheduler
