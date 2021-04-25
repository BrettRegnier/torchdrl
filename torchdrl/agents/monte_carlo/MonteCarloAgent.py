import torch
import torch.nn as nn

from torchdrl.agents.Agent import Agent

class MonteCarloAgent(Agent):
    def __init__(self, actor_critic, optimizer, env, test_env, gamma, max_steps=-1, device='cpu'):
        self._policy = actor_critic.to(device)

        self._optimizer = optimizer
        self._env = env
        self._test_env = test_env
        self._gamma = gamma

        self._max_steps = max_steps
        self._device = device

    def Rollout(self, rewards, normalize=True):
        discounted_rewards = []
        running_reward = 0

        for reward in reversed(rewards):
            running_reward = running_reward * self._gamma + reward
            discounted_rewards.insert(0, running_reward)

        discounted_rewards_t = torch.tensor(discounted_rewards, device=self._device)

        if normalize:
            discounted_rewards_t = (discounted_rewards_t - discounted_rewards_t.mean()) / discounted_rewards_t.std()

        return discounted_rewards_t

def InitWeights(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_normal_(module.weight)
        module.bias.data.fill_(0)