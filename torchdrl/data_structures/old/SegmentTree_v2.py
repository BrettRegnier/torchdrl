import numpy as np

class SegmentTree_v2:
    def __init__(self, capacity):
        # TODO make sure is > 0 and is a power of 2

        self._capacity = capacity - 1 
        self._tree = np.zeros(2 * self._capacity + 1)
        self._entries = 0

    def Total(self):
        return self._tree[0]
    
    def __setitem__(self, idx, value):
        idx = idx + self._capacity
        self._tree[idx] = value
        
        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] = self._tree[2*idx+1] + self._tree[2*idx+2]

        if self._entries < self._capacity:
            self._entries += 1

    def Retrieve(self, idx, sample):
        num_leafs = self._capacity
        while idx < num_leafs:
            left = idx * 2 + 1 
            right = left + 1
            node_value = self._tree[left]

            if node_value > sample:
                idx = left
            else:
                idx = right
                sample -= node_value
        return idx
    
    def GetLeaf(self, sample):
        idx = self.Retrieve(0, sample)
        priority = self._tree[idx]

        return (idx - self._capacity, priority)

    def GetPriority(self, idx):
        idx = idx + self._capacity
        return self._tree[idx]
    