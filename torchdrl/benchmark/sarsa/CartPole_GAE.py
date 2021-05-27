
import torch
import torch.optim as optim
import torch.nn as nn
import gym
import numpy as np
from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.ActorCriticNetwork import ActorCriticNetwork
from torchdrl.agents.monte_carlo.GAE import GAE
import matplotlib.pyplot as plt 

from DragonFruit.env.BoatLeftRight_v0 import BoatLeftRight_v0

env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")

SEED = 0

env.seed(SEED)
test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)

gamma = 0.99
trace_decay = 0.99

max_ep = 10000
n_trials = 25
reward_threshold = 475
print_every = 10

lr = 0.01
max_steps = 500

train_rewards = []
test_rewards = []


input_shape = env.observation_space.shape
n_actions = env.action_space.n
hidden_layers = [128]
dropouts = [0.5]
activations = ["relu"]

device = "cpu"
actor = nn.Sequential(
    nn.Linear(np.sum(*input_shape), 128),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(128, n_actions)
)
critic = nn.Sequential(
    nn.Linear(np.sum(*input_shape), 128),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(128, 1)
)

model = ActorCriticNetwork(actor, critic).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr)

agent = GAE(trace_decay, model, optimizer, env, test_env, gamma, max_steps=max_steps, device=device)

episode = 1
mean_test_rewards = 0

while episode < max_ep or mean_test_rewards < reward_threshold:
    actor_loss, critic_loss, train_reward = agent.Train()

    test_reward, test_info = agent.Evaluate()

    train_rewards.append(train_reward)
    test_rewards.append(test_reward)

    mean_train_rewards = np.mean(train_rewards[-n_trials:])
    mean_test_rewards = np.mean(test_rewards[-n_trials:])

    if episode % print_every == 0:
        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
        win = 0
        lose = 0
    
    if mean_test_rewards >= reward_threshold:
        print(f'Reached reward threshold in {episode} episodes')
        break

    episode += 1

# win = 0
# lose = 0
# for _ in range(1000):
#     test_reward, test_info = agent.Evaluate()
#     if test_info['win']:
#         win += 1
#     else:
#         lose += 1
# print(f"Wins: {win}, Loses {lose}")

plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.hlines(reward_threshold, 0, len(test_rewards), color='r')
plt.legend(loc='lower right')
plt.grid()
plt.show()