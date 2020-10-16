import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random

from .BaseAgent import BaseAgent

from ..neural_networks.FullyConnectedNetwork import FullyConnectedNetwork 
from ..representations.Plotter import Plotter


class DQL(BaseAgent):
    def __init__(self, config):
        super(DQL, self).__init__(config)
        self._epsilon = self._hyperparameters['epsilon']
        self._epsilon_decay = self._hyperparameters['epsilon_decay']
        self._epsilon_min = self._hyperparameters['epsilon_min']
        self._gamma = self._hyperparameters['gamma']

        self._tau = self._hyperparameters['tau']
        self._target_update_steps = 0 
        self._target_update_frequency = self._hyperparameters['target_update']


        fcc = self._hyperparameters['fc']
        self._net = FullyConnectedNetwork(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._target_net = FullyConnectedNetwork(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['dropouts'], fcc['final_activation'], self._hyperparameters['convo']).to(self._device)
        self._net_optimizer = Adam(self._net.parameters(), lr=self._hyperparameters['lr'])

        self.UpdateNetwork(self._net, self._target_net)
    
    def PlayEpisode(self, evaluate=False):
        done = False
        steps = 0
        episode_reward = 0
        
        state = self._env.reset()

        # this is to optimize the loop a little bit by 
        # avoiding running a useless if statement after warm up
        while steps != self._max_steps and (self._total_steps < self._warm_up or len(self._memory) < self._batch_size) and not done:
            action = self._env.action_space.sample()
            next_state, reward, done, info = self._env.step(action)

            # if not (steps == 0 and done):
            self._memory.Append(state, action, next_state, reward, done)

            episode_reward += reward
            state = next_state
            steps += 1
            self._total_steps += 1

        while steps != self._max_steps and not done:
            action = self.Act(state)
                
            next_state, reward, done, info = self._env.step(action)

            if self._apex:
                self._internal_memory.Append(state, action, next_state, reward, done)
                self.ApexSendMemories()
            else:
                self._memory.Append(state, action, next_state, reward, done)
                self.Learn()

            # update epsilon
            self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)
            
            episode_reward += reward
            state = next_state

            steps += 1
            self._total_steps += 1

        info['epsilon'] = round(self._epsilon, 3)
            
        return episode_reward, steps, info

    @torch.no_grad()
    def Act(self, state):  
        if random.random() < self._epsilon:
            action = self._env.action_space.sample()
        else:
            # why am i detaching?
            state_t = torch.tensor(state, dtype=torch.float32, device=self._device).detach()
            state_t = state_t.unsqueeze(0)
            
            q_values = self._net(state_t)
            action = torch.argmax(q_values).item()

        return action
    
    def Learn(self):
        states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t = self.SampleMemoryT(self._batch_size)


        errors = self.CalculateErrors(states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, self._batch_size)
        loss = torch.mean(errors * weights_t)

        self._net_optimizer.zero_grad()
        loss.backward()
        self._net_optimizer.step()

        # soft update target 
        if self._target_update_steps % self._target_update_frequency == 0:
            self.UpdateNetwork(self._net, self._target_net, self._tau)
            self._target_update_steps = 0

        self._target_update_steps += 1

        # errors = loss.detach().cpu().numpy()
        # print(errors); exit();
        self._memory.BatchUpdate(indices_np, errors.detach().cpu().numpy())
        
    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size):
        q_values = self._net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self._target_net(next_states_t).max(dim=1)[0]
            next_state_values[dones_t] = 0.0
            next_state_values = next_state_values.detach()

        q_target = rewards_t + self._gamma * next_state_values
        
        errors = F.smooth_l1_loss(q_values, q_target, reduction="none")
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
        