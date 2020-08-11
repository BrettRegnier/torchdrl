import collections
import numpy as np


Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'next_state', 'reward', 'done'])

class UniformExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def Append(self, state, action, next_state, reward, done):
        experience = Experience(state, action, next_state, reward, done)
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
            states.append(self.buffer[idx].state)
            actions.append(self.buffer[idx].action)
            next_states.append(self.buffer[idx].next_state)
            rewards.append(self.buffer[idx].reward)
            dones.append(self.buffer[idx].done)

        states_np = np.array(states)
        actions_np = np.array(actions)
        next_states_np = np.array(next_states)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)

        return states_np, actions_np, next_states_np, rewards_np, dones_np

# alternative
import numpy as np
class NewUEP:
    def __init__(self, mem_size, state_shape):
        self._mem_size = mem_size
        self._mem_count = 0

        self._states = np.zeros((self._mem_size, *state_shape), dtype=np.float32)
        self._actions = np.zeros(self._mem_size, dtype=np.int64)
        self._next_states = np.zeros((self._mem_size, *state_shape), dtype=np.float32)
        self._rewards = np.zeros(self._mem_size, dtype=np.float32)
        self._dones = np.zeros(self._mem_size, dtype=np.bool)

    def __len__(self):
        return self._mem_count

    def Append(self, state, action, next_state, reward, done):
        mem_idx = self._mem_count % self._mem_size

        self._states[mem_idx] = state
        self._actions[mem_idx] = action
        self._next_states[mem_idx] = next_state
        self._rewards[mem_idx] = reward
        self._dones[mem_idx] = done

        self._mem_count += 1

    def Sample(self, batch_size):
        mem_max = min(self._mem_count, self._mem_size)
        batch_indices = np.random.choice(mem_max, batch_size, replace=True)

        states = self._states[batch_indices]
        actions = self._actions[batch_indices]
        next_states = self._next_states[batch_indices]
        rewards = self._rewards[batch_indices]
        dones = self._dones[batch_indices]

        return states, actions, next_states, rewards, dones