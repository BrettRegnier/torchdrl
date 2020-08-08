import collections
import numpy as np


Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'next_state', 'reward', 'done'])

class UniformExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def Append(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def Sample(self, batch_size):
        indices = np.random.choice(
            len(self.buffer), batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for idx in indices:
            x = self.buffer[idx]
            states.append(self.buffer[idx].state)
            actions.append(self.buffer[idx].action)
            next_states.append(self.buffer[idx].next_state)
            rewards.append(self.buffer[idx].reward)
            dones.append(self.buffer[idx].done)

        states_np = np.vstack(states)
        actions_np = np.vstack(actions)
        next_states_np = np.vstack(next_states)
        rewards_np = np.vstack(rewards)
        dones_np = np.vstack(dones)

        return states_np, actions_np, next_states_np, rewards_np, dones_np
