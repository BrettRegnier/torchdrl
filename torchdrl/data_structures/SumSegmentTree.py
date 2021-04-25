import numpy as np

class SumSegmentTree:
    def __init__(self, capacity):
        #TODO assert even capacity
        # assert capacity % 2 == 0
        assert capacity > 1

        self._capacity = capacity - 1
        self._tree = [0 for _ in range(2 * self._capacity + 1)]
        self._pointer = 0
        self._entries = 0

    def Total(self):
        return self._tree[0]

    def Retrieve(self, sample):
        parent_idx = 0
        while parent_idx < self._capacity:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            node_value = self._tree[left_idx]

            if node_value > sample:
                parent_idx = left_idx
            else:
                parent_idx = right_idx
                sample -= node_value
        
        return parent_idx - self._capacity

    def Add(self, priority):
        self.UpdatePriority(self._pointer, priority)

        self._pointer = (self._pointer % self._capacity) + 1
        if self._entries < self._capacity:
            self._entries += 1

    def Get(self, idx):
        assert (0 <= idx <= self._capacity)

        return self._tree[idx + self._capacity]
    
    def __getitem__(self, idx):
        return self.Get(idx)

    def UpdatePriority(self, idx, priority):
        idx += self._capacity
        self._tree[idx] = priority

        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] = self._tree[2*idx+1] + self._tree[2*idx+2]

