import numpy as np

class SegmentTree_v2:
    def __init__(self, capacity):
        # TODO make sure is > 0 and is a power of 2

        self._capacity = capacity        
        self._tree = np.zeros(2 * self._capacity - 1)

        self._pointer = 0
        self._entries = 0

    def Total(self):
        return self._tree[0]
    
    def Add(self, priority):        
        self.Update(self._pointer, priority)
        
        self._pointer = (self._pointer + 1) % self._capacity        
        if self._entries < self._capacity:
            self._entries += 1
    
    def Update(self, idx, priority):
        idx = (idx + self._capacity) - 1
        self._tree[idx] = priority
        
        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] = self._tree[2*idx+1] + self._tree[2*idx+2]
    
    def Retrieve(self, idx, sample):
        num_leafs = self._capacity - 1
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

        return (idx - (self._capacity - 1), priority)

    def GetPriority(self, idx):
        idx = (idx + self._capacity) - 1
        return self._tree[idx]
    