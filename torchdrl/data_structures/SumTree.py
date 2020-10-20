import collections
import numpy as np
import numpy
import random
import math

class SumTree:
    def __init__(self, capacity):
        self._capacity = capacity
        self._tree_capacity = 2 * self._capacity - 1

        # total nodes 
        self._tree = np.zeros(self._tree_capacity)

        # data is stored on the leaf nodes
        self._data = np.zeros(self._capacity, dtype=object)
        
        self._data_pointer = 0
        self._entries = 0
        self._minimum_priority = math.inf

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

    def Total(self):
        return self._tree[0]

    def Add(self, priority, data):
        idx = (self._data_pointer + self._capacity) - 1

        self._data[self._data_pointer] = data
        self.Update(idx, priority)

        self._data_pointer = (self._data_pointer + 1) % self._capacity
           
        if self._entries < self._capacity:
            self._entries += 1

    def Update(self, idx, priority):
        self._tree[idx] = priority

        # propogate
        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] = self._tree[2*idx+1] + self._tree[2*idx+2]

    def GetLeaf(self, sample):
        idx = self.Retrieve(0, sample)
        data_idx = idx - self._capacity + 1

        return (idx, self._tree[idx], self._data[data_idx])

    # def GetIndex(self, idx):

