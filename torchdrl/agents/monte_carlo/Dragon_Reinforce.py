import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributions as distributions

import matplotlib.pyplot as plt 
import numpy as np 
import gym 

from DragonFruit.env.BoatLeftRight_v0 import BoatLeftRight_v0

class Network(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions, dropout=0.5):
        super().__init__()

        self._conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 4, 2, 1),
            nn.ReLU()
        )

        out = self._conv(torch.zeros(1, *input_shape))
        output_size = int(np.prod(out.size()))
        self._fc = nn.Sequential(
            nn.Linear(output_size, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state):
        x = self._conv(state)
        x = x.flatten().unsqueeze(0)
        out = self._fc(x)
        return out

    
class Reinforce:
    def __init__(self, train_env, test_env, gamma):
        self._train_env = train_env
        self._test_env = test_env
        self._gamma = gamma
        
        # input_dim = np.sum(*self._train_env.observation_space.shape)
        input_shape = self._train_env.observation_space.shape
        hidden_dim = 1024
        n_actions = self._train_env.action_space.n

        self._policy = Network(input_shape, hidden_dim, n_actions)
        self._policy.apply(InitWeights)

        print(self._policy)
        
        self._optim = optim.Adam(self._policy.parameters(), lr=0.01)

    def Train(self):
        self._policy.train()

        log_prob_actions = []
        rewards = []
        done = False
        episode_reward = 0

        state = self._train_env.reset()

        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_pred = self._policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            state, reward, done, _ = self._train_env.step(action.item())
            
            log_prob_actions.append(log_prob_action)
            rewards.append(reward)

            episode_reward += reward

        log_prob_actions = torch.cat(log_prob_actions)
        discounted_reward = self.Rollout(rewards)
        loss = self.Learn(discounted_reward, log_prob_actions)

        return loss, episode_reward


    def Rollout(self, rewards, normalize=True):
        discounted_rewards = []
        running_reward = 0

        for reward in reversed(rewards):
            running_reward = running_reward * self._gamma + reward
            discounted_rewards.insert(0, running_reward)
        
        discounted_rewards_t = torch.tensor(discounted_rewards)

        if normalize:
            if len(discounted_rewards_t) > 1:
                discounted_rewards_t = (discounted_rewards_t - discounted_rewards_t.mean()) / (discounted_rewards_t.std() + 1e-8)

        return discounted_rewards_t

    def Learn(self, rewards, log_prob_actions):
        rewards_t = rewards.detach()
        loss = -(rewards_t * log_prob_actions).sum()

        self._optim.zero_grad()

        loss.backward()

        self._optim.step()

        return loss.item()

    def Evaluate(self):
        self._policy.eval()

        done = False
        episode_reward = 0
        
        state = self._test_env.reset()

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_pred = self._policy(state_t)
                action_prob = F.softmax(action_pred, dim=-1)

            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _ = self._test_env.step(action.item())

            episode_reward += reward

        return episode_reward

def InitWeights(module):    
    if type(module) == nn.Linear:
        torch.nn.init.xavier_normal_(module.weight)
        module.bias.data.fill_(0)

env_kwargs = {
    "num_participants": 4,
    "num_locked": 0,
    "lr_goal": 0,
    "lr_relax": 0,
    "fb_goal": 30,
    "fb_relax": 0,
    "illegal_stopping": False,
    "reward_type": "shaped",
    "include_weight_input": False
}

train_env = BoatLeftRight_v0(**env_kwargs)
test_env = BoatLeftRight_v0(**env_kwargs)

SEED = 0

# train_env.seed(SEED)
# test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)

max_ep = 10000
gamma = 0.99
n_trials = 25
reward_threshold = 475
print_every = 10

train_rewards = []
test_rewards = []

agent = Reinforce(train_env, test_env, gamma)
for episode in range(1, max_ep):
    loss, train_reward = agent.Train()

    test_reward = agent.Evaluate()

    train_rewards.append(train_reward)
    test_rewards.append(test_reward)

    mean_train_rewards = np.mean(train_rewards[-n_trials:])
    mean_test_rewards = np.mean(test_rewards[-n_trials:])

    if episode % print_every == 0:
        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
    
    if mean_test_rewards >= reward_threshold:
        
        print(f'Reached reward threshold in {episode} episodes')
        
        break

plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.hlines(reward_threshold, 0, len(test_rewards), color='r')
plt.legend(loc='lower right')
plt.grid()
plt.show()