import numpy as np

class SegmentTree:
    def __init__(self, capacity):
        # TODO make sure is > 0 and is a power of 2

        self._capacity = capacity
        self._tree_capacity = 2 * capacity - 1
        
        
        self._data = np.zeros(self._capacity, dtype=object)
        self._tree = np.zeros(self._tree_capacity)

        self._data_pointer = 0
        self._entries = 0
    
    def Add(self, priority, data):
        tree_pointer = (self._data_pointer + self._capacity) - 1
        
        self._data[self._data_pointer] = data
        self.Update(tree_pointer, priority)
        
        self._data_pointer = (self._data_pointer + 1) % self._capacity        
        if self._entries < self._capacity:
            self._entries += 1
    
    def Update(self, idx, priority):
        delta = priority - self._tree[idx]
        self._tree[idx] = priority
        
        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] += delta
    
    def Retrieve(self, idx, sample):
        pass
    
    def GetLeaf(self, sample):
        pass
    
    