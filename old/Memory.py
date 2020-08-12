import collections
import numpy as np


Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'reward', 'next_state', 'done'])


class UniformExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def Append(self, state, action, reward, next_state, done):
        experience = Experience(state, [action], [reward], next_state, [done])
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
            rewards.append(self.buffer[idx].reward)
            next_states.append(self.buffer[idx].next_state)
            dones.append(self.buffer[idx].done)

        return states, actions, rewards, next_states, dones
        # return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.int8)
