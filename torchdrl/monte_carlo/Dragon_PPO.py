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
            nn.Conv2d(input_shape[0], 32, [1,1], [1,1], [0,0]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2,1], [1,1], [1,0]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [4,2], [1,1], [3,1]),
            nn.ReLU()
        )

        out = self._conv(torch.zeros(1, *input_shape))
        output_size = int(np.prod(out.size()))
        self._fc = nn.Sequential(
            nn.Linear(output_size, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state):
        x = self._conv(state)
        x = x.view(x.size(0), -1)
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
    def __init__(self, train_env, test_env, gamma, ppo_steps, ppo_clip):
        self._train_env = train_env
        self._test_env = test_env
        self._gamma = gamma
        self._ppo_steps = ppo_steps
        self._ppo_clip = ppo_clip
        
        # input_dim = np.sum(*self._train_env.observation_space.shape)
        input_shape = self._train_env.observation_space.shape
        hidden_dim = 1024
        n_actions = self._train_env.action_space.n

        actor = Network(input_shape, hidden_dim, n_actions)
        critic = Network(input_shape, hidden_dim, 1)

        self._policy = ActorCritic(actor, critic)
        self._policy.apply(InitWeights)
        
        self._optim = optim.Adam(self._policy.parameters(), lr=0.00001)

    def Train(self):
        self._policy.train()

        states = []
        actions = []
        log_prob_actions = []
        values = []
        rewards = []

        done = False
        episode_reward = 0

        state = self._train_env.reset()

        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # append current state 
            states.append(state)

            action_pred, value_pred = self._policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            state, reward, done, _ = self._train_env.step(action.item())
            
            actions.append(action)
            log_prob_actions.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)

            episode_reward += reward

        states = torch.cat(states)
        actions = torch.cat(actions)
        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values).squeeze(-1)

        discounted_rewards = self.Rollout(rewards)
        advantages = self.CalcuateAdvantages(discounted_rewards, values)

        policy_loss, value_loss = self.Learn(states, actions,log_prob_actions, advantages, discounted_rewards)

        return policy_loss, value_loss, episode_reward

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

    def CalcuateAdvantages(self, discounted_rewards, values, normalize=True):
        advantages = discounted_rewards - values

        if normalize:
            if len(discounted_rewards) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def Learn(self, states, actions,log_prob_actions, advantages, rewards):
        total_policy_loss = 0
        total_value_loss = 0

        advantages_t = advantages.detach()
        log_prob_actions_t = log_prob_actions.detach()
        actions_t = actions.detach()

        for _ in range(self._ppo_steps):
            # get new log prob of actions for all input states
            action_pred, value_pred = self._policy(states)
            value_pred = value_pred.squeeze(-1)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)

            # new log prob using old actions 
            new_log_prob_actions = dist.log_prob(actions_t)

            policy_ratio = (new_log_prob_actions - log_prob_actions_t).exp()

            policy_loss_1 = policy_ratio * advantages_t
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - self._ppo_clip, max = 1.0 + self._ppo_clip) * advantages_t

            policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
            
            value_loss = F.smooth_l1_loss(rewards, value_pred).sum()

            self._optim.zero_grad()

            policy_loss.backward()
            value_loss.backward()

            self._optim.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return total_policy_loss / self._ppo_steps, total_value_loss / self._ppo_steps

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

            # distributed
            dist = distributions.Categorical(action_prob)
            action = dist.sample()

            # greedy
            # action = torch.argmax(action_prob, dim=-1)
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

gamma = 0.05
ppo_steps = 5
ppo_clip = 10

max_ep = 1000
n_trials = 25
reward_threshold = 1000
print_every = 10

train_rewards = []
test_rewards = []

agent = A2CAgent(train_env, test_env, gamma, ppo_steps, ppo_clip)
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