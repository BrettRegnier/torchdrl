import gym
from gym import spaces
import random
import math


class NumberSort(gym.Env):
    def __init__(self, **kwargs):
        super(NumberSort, self).__init__()

        self._state = [4, 1, 2, 5, 3]
        if "numbers" in kwargs.keys():
            self._state = kwargs['numbers']

        self._correct_state = self._state[:]
        self._correct_state.sort()

        self._prev_state = self._state

        largest = max(self._state)
        self._compressed_state = [1/(1 + math.e ** -(x/largest)) for x in self._state]

        self._actions = []
        for x in range(len(self._state) - 1):
            for y in range(x + 1, len(self._state)):
                self._actions.append((x, y))

        # for x in range(len(self._state) -1):
        #     for y in range(x+1,len(self._state)):
        #         if x == y:
        #             continue
        #         self._actions.append((x,y))

        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(low=0, high=10, shape=(1, len(self._state)))
        self._steps = 0

    def step(self, action):
        self._steps += 1
        done = True
        reward = -0

        self._prev_state = self._state[:]
        i, j = self._actions[action]
        x, y = self.swap(i, j)

        # if y < x:
        #     reward = 0.1
        # else:
        #     reward = -0.1

        # reward
        # for x in range(0, )

        for i in range(len(self._state)):
            if self._state[i] == self._correct_state[i]:
                reward += 0
            else:
                done = False
            # if self._prev_state[i] == self._correct_state[i]:
            #     reward -= 1.0

        if done:
            reward = 1
            self.render()

        return self.state(), reward, done, {}

    def render(self, mode="human"):
        print(self._state, end=" ")

    def swap(self, i, j):
        x = self._state[i]
        y = self._state[j]

        self._state[i] = y
        self._state[j] = x


        k = self._compressed_state[i]
        l = self._compressed_state[j]

        self._compressed_state[i] = l
        self._compressed_state[j] = k

        return x, y    
        
    def reset(self):
        self._steps = 0
        
        in_order = True
        while in_order:
            for i in range(len(self._state) * 10):
                i, j = random.choice(self._actions)
                self.swap(i, j)

            for x in range(0, len(self._state) - 1):
                if self._state[x] > self._state[x+1]:
                    in_order = False
        

        return self.state()

    def seed(self, seed):
        self._seed = seed
        random.seed(seed)

    def state(self):
        return self._compressed_state

# test = FiveNumSort()
# test.render()
# test.swap(0, 1)
# test.render()
