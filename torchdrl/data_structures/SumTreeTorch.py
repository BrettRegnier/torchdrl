import collections
import numpy as np
import random

import torch

class SumTreeTorch:
    def __init__(self, capacity, input_shape, device='cpu'):
        self._capacity = capacity
        self._device = device
        self._tree_capacity = 2 * self._capacity - 1

        self._tree = torch.zeros(self._tree_capacity, device=device)

        self._states = torch.zeros([capacity, *input_shape], dtype=torch.float32, device=device)
        self._actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self._next_states = torch.zeros([capacity, *input_shape], dtype=torch.float32, device=device)
        self._rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self._dones = torch.zeros(capacity, dtype=torch.int64, device=device)

        self._pointer = 0
        self._entries = 0

    @torch.no_grad()
    def Total(self):
        return self._tree[0]

    @torch.no_grad()
    def Retrieve(self, idx, sample):
        parent_idx = idx
        left_idx = parent_idx * 2 + 1
        right_idx = left_idx + 1

        while left_idx < self._tree_capacity:
            node_value = self._tree[left_idx]
            if sample <= node_value:
                parent_idx = left_idx
            else:
                parent_idx = right_idx
                sample -= node_value

            left_idx = parent_idx * 2 + 1
            right_idx = left_idx + 1        

        return parent_idx

    @torch.no_grad()
    def Add(self, priority, state, action, next_state, reward, done):
        idx = self._pointer + self._capacity - 1

        self._states[self._pointer] = torch.tensor(state, dtype=torch.float32)
        self._actions[self._pointer] = torch.tensor(action, dtype=torch.int64)
        self._next_states[self._pointer] = torch.tensor(next_state, dtype=torch.float32)
        self._rewards[self._pointer] = torch.tensor(reward, dtype=torch.float32)
        self._dones[self._pointer] = torch.tensor(done, dtype=torch.int64)
        self.Update(idx, priority)

        self._pointer += 1
        # if we are past the capacity begin overwriting
        if self._pointer >= self._capacity:
            self._pointer = 0
    
        if self._entries < self._capacity:
            self._entries += 1

    @torch.no_grad()
    def Update(self, idx, priority):
        change = priority - self._tree[idx]
        self._tree[idx] = priority

        # propogate
        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] += change

    @torch.no_grad()
    def GetLeaf(self, sample):
        idx = self.Retrieve(0, sample)
        data_idx = idx - self._capacity + 1

        state = self._states[data_idx]
        action = self._actions[data_idx]
        next_state = self._next_states[data_idx]
        reward = self._rewards[data_idx]
        done = self._rewards[data_idx]
        
        return (idx, self._tree[idx], state, action, next_state, reward, done)

    @torch.no_grad()
    def GetLeaves(self, samples):
        indices = []
        data_indices = []

        for sample in samples:
            idx = self.Retrieve(0, sample)
            indices.append(idx)
            data_indices.append(idx - self._capacity + 1)
        
        indices_t = torch.tensor(indices, dtype=torch.int64, device=self._device)

        priorities = self._tree[indices]
        states = self._states[data_indices]
        actions = self._actions[data_indices]
        next_states = self._next_states[data_indices]
        rewards = self._rewards[data_indices]
        dones = self._dones[data_indices]

        return indices_t, priorities, states, actions, next_states, rewards, dones

