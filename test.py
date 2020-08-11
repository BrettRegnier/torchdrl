import gym       
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, mem_size, state_shape):
        self.mem_size = mem_size
        self.mem_count = 0

        self.states     = np.zeros((self.mem_size, *state_shape),dtype=np.float32)
        self.actions    = np.zeros( self.mem_size,               dtype=np.int64  )
        self.rewards    = np.zeros( self.mem_size,               dtype=np.float32)
        self.states_    = np.zeros((self.mem_size, *state_shape),dtype=np.float32)
        self.dones      = np.zeros( self.mem_size,               dtype=np.bool   )

    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % self.mem_size 
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index]   = done

        self.mem_count += 1

    def sample(self, sample_size):
        mem_max = min(self.mem_count, self.mem_size)
        batch_indices = np.random.choice(mem_max, sample_size, replace=True)

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class Network(torch.nn.Module):
    def __init__(self, alpha, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.fc1_dims = 1024
        self.fc2_dims = 512

        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent():
    def __init__(self, lr, state_shape, num_actions):
        self.net = Network(lr, state_shape, num_actions)
        self.future_net = Network(lr, state_shape, num_actions)
        self.memory = ReplayBuffer(mem_size=100000, state_shape=state_shape)
        self.batch_size = 64
        self.gamma = 0.99

        self.epsilon = 0.1
        self.epsilon_decay = 0.00005
        self.epsilon_min = 0.001

        self.learn_step_counter = 0
        self.net_copy_interval = 10

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            state = torch.tensor(observation).float().detach()
            state = state.to(self.net.device)
            state = state.unsqueeze(0)

            q_values = self.net(state)
            action = torch.argmax(q_values).item()
        return action

    def store_memory(self, state, action, reward, state_, done):
        self.memory.add(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        states  = torch.tensor(states , dtype=torch.float32).to(self.net.device)
        actions = torch.tensor(actions, dtype=torch.long   ).to(self.net.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.net.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.net.device)
        dones   = torch.tensor(dones  , dtype=torch.bool   ).to(self.net.device)

        batch_indices = np.arange(self.batch_size, dtype=np.int64)
        q_values  =   self.net(states)[batch_indices, actions]

        q_values_ =   self.future_net(states_)
        actions_ =   self.net(states_).max(dim=1)[1]
        action_qs_ = q_values_[batch_indices, actions_]

        action_qs_[dones] = 0.0
        q_target = rewards + self.gamma * action_qs_

        td = q_target - q_values

        self.net.optimizer.zero_grad()
        loss = ((td ** 2.0)).mean()
        loss.backward()
        self.net.optimizer.step()

        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        if self.learn_step_counter % self.net_copy_interval == 0:
            self.future_net.load_state_dict(self.net.state_dict())

        self.learn_step_counter += 1

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(lr=0.001, state_shape=(4,), num_actions=2)

    high_score = -math.inf
    episode = 0

    num_samples = 0         
    samples_processed = 0  
    while True:
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            # env.render()

            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            agent.store_memory(state, action, reward, state_, done)
            
            agent.learn()
            samples_processed += agent.batch_size
            
            state = state_

            score += reward
            frame += 1
            num_samples += 1 
            
        high_score = max(high_score, score)

        print(( "samples: {}, samps_procd: {}, ep {:4d}: high-score {:12.3f}, "
                "score {:12.3f}, epsilon {:5.3f}").format(
            num_samples, samples_processed, episode, 
            high_score, score, agent.epsilon))

        episode += 1