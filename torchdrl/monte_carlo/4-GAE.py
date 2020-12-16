import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributions as distributions

import matplotlib.pyplot as plt 
import numpy as np 
import gym 

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, dropout=0.5):
        super().__init__()
        
        self._fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        out = self._fc(x)
        return out

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self._actor = actor 
        self._critic = critic 

    def forward(self, state):
        action_pred = self._actor(state)
        value_pred = self._critic(state)

        return action_pred, value_pred
    
class A2CAgent:
    def __init__(self, train_env, test_env, gamma, trace_decay):
        self._train_env = train_env
        self._test_env = test_env
        self._gamma = gamma
        self._trace_decay = trace_decay
        
        input_dim = np.sum(*self._train_env.observation_space.shape)
        hidden_dim = 128
        n_actions = self._train_env.action_space.n

        actor = Network(input_dim, hidden_dim, n_actions)
        critic = Network(input_dim, hidden_dim, 1)

        self._policy = ActorCritic(actor, critic)
        self._policy.apply(InitWeights)
        
        self._optim = optim.Adam(self._policy.parameters(), lr=0.01)

    def Train(self):
        self._policy.train()

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

        discounted_rewards = self.Rollout(rewards)
        # uses rewards not the discounted_rewards for advantages unlike a2c
        advantages = self.CalcuateAdvantages(rewards, values)

        policy_loss, value_loss = self.Learn(advantages, log_prob_actions, discounted_rewards, values)

        return policy_loss, value_loss, episode_reward

    def Rollout(self, rewards, normalize=True):
        discounted_rewards = []
        running_reward = 0

        for reward in reversed(rewards):
            running_reward = running_reward * self._gamma + reward
            discounted_rewards.insert(0, running_reward)
        
        discounted_rewards_t = torch.tensor(discounted_rewards)

        if normalize:
            discounted_rewards_t = (discounted_rewards_t - discounted_rewards_t.mean()) / discounted_rewards_t.std()

        return discounted_rewards_t

    def CalcuateAdvantages(self, rewards, values, normalize=True):
        advantages = []
        advantage = 0
        next_value = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + next_value * self._gamma - v
            advantage = td_error + advantage * self._gamma * self._trace_decay
            next_value = v 
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages)

        if normalize:
            advantages = (advantages - advantages.mean()) / advantages.std()

        return advantages

    def Learn(self, advantages, log_prob_actions, rewards, values):
        advantages_t = advantages.detach()
        rewards_t = rewards.detach()

        policy_loss = -(advantages_t * log_prob_actions).sum()

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

def InitWeights(module):    
    if type(module) == nn.Linear:
        torch.nn.init.xavier_normal_(module.weight)
        module.bias.data.fill_(0)



train_env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")

SEED = 0

train_env.seed(SEED)
test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)

max_ep = 500
gamma = 0.99
trace_decay = 0.99
n_trials = 25
reward_threshold = 475
print_every = 10

train_rewards = []
test_rewards = []

agent = A2CAgent(train_env, test_env, gamma, trace_decay)
for episode in range(1, max_ep+1):
    actor_loss, critic_loss, train_reward = agent.Train()

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