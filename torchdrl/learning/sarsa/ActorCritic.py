import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributions as distributions

import matplotlib.pyplot as plt 
import numpy as np 
import gym 

import torchdrl.tools.Helper as Helper
from torchdrl.agents.SARSAAgent import SARSAAgent

class ActorCriticAgent(SARSAAgent):
    def __init__(self, trace_decay, *args, **kwargs):
        super(ActorCriticAgent, self).__init__(*args, **kwargs)

    def Train(self):
        self._policy.train()

        return self.PlayEpisode()
    
    def TrainNoYield(self):
        pass

    def Learn(self, rewards, log_prob_actions, values):
        rewards_t = rewards.detach()

        policy_loss = -(rewards_t * log_prob_actions).sum()

        value_loss = F.smooth_l1_loss(rewards_t, values).sum()

        self._optim.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        self._optim.step()

        return policy_loss.item(), value_loss.item()

    def Evaluate(self):
        self._policy.eval()

        done = False
        episode_reward = 0
        
        state = self._test_env.reset()

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_pred, _ = self._policy(state_t)
                action_prob = F.softmax(action_pred, dim=-1)

            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _ = self._test_env.step(action.item())

            episode_reward += reward

        return episode_reward

    def GetAction(self, state, evaluate=False):
        return self.Act(state)

    def Act(self, state):
        state_t = Helper.ConvertStateToTensor(state, self._device)
        return self._policy(state_t)

    def Save(self, folderpath, filename):
        pass

    def Load(self, filepath):
        pass

    def PlayEpisode(self):
        log_prob_actions = []
        values = []
        rewards = []
        done = False
        episode_reward = 0

        state = self._train_env.reset()

        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_pred, value_pred = self._policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            state, reward, done, _ = self._train_env.step(action.item())
            
            log_prob_actions.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)

            episode_reward += reward

        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values).squeeze(-1)

        discounted_reward = self.Rollout(rewards)

        policy_loss, value_loss = self.Learn(discounted_reward, log_prob_actions, values)

        return policy_loss, value_loss, episode_reward