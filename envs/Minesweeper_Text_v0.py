import gym
from gym import spaces
import numpy as np
import random

import collections

UNREVEALED_TILE = -1
MINE_TILE = 9

STATE = 0
NEIGHBOURS = 1
NUM_MINES = 2
IS_MINE = 3

# TODO change soft variable to seed, its the same thing.


class Minesweeper_Text_v0(gym.Env):
    def __init__(self, **kwargs):
        super(Minesweeper_Text_v0, self).__init__()
        
        difficulty = 1
        if "difficulty" in kwargs.keys():
            difficulty = kwargs["difficulty"]
            
        self._flat = False
        if "flat" in kwargs.keys():
            self._flat = kwargs["flat"]

        self._rows = int(6 * difficulty + 2)
        self._columns = int(11 * difficulty - difficulty**2)
        self._mines = max(int(((0.006895 * difficulty**2) + 0.013045 *
                           difficulty + 0.10506) * (self._rows * self._columns)), 4)


        self._board = None
        self._unrevealed_remaining = (self._rows * self._columns) - self._mines

        self._done = False
        self._win = False
        self._steps = 0
        self._MAX_STEPS = (self._rows * self._columns) - self._mines

        self.action_space = spaces.Discrete(self._rows * self._columns)
        self.observation_space = spaces.Box(
            low=-1, high=9, shape=(1, self._rows, self._columns), dtype=np.float32)
        # self.observation_space = spaces.Box(low=0, high=10, shape=(
        #     1, self._rows * self._columns), dtype=np.int32)
        self._seed = 0

    def step(self, action):
        tile = self._board[action]
        self._done = False
        self._win = False
        state = None
        # reward = 0.3
        reward = 0

        # check if mine or out of steps
        if tile[IS_MINE] == True:
            self._done = True
            reward = -1
            tile[STATE] = MINE_TILE

        # check if unrevealed
        if tile[STATE] == UNREVEALED_TILE and not self._done:
            # reward = 0.7
            reward = 1.0

            # reveal all neighbouring tiles that can be
            queue = [tile]
            while len(queue) > 0:
                queue_tile = queue.pop(0)
                num_mines = queue_tile[NUM_MINES]
                if queue_tile[STATE] == UNREVEALED_TILE:
                    queue_tile[STATE] = num_mines
                    self._unrevealed_remaining -= 1
                if queue_tile[NUM_MINES] == 0 and not queue_tile[IS_MINE]:
                    for n_idx in queue_tile[NEIGHBOURS]:
                        if n_idx != -1:
                            adj = self._board[n_idx]
                            if adj[STATE] == UNREVEALED_TILE:
                                queue.append(adj)

        # check win condition
        if self._unrevealed_remaining == 0:
            self._done = True
            self._win = True
            reward = 1
            
        # if self._steps == 0 and self._done:
        #     reward = 0

        self._steps += 1

        state = self.State()

        # print(state, reward, done, win)
        return state, reward, self._done, {'win':self._win}

    def reset(self, soft=False, load=False):
        self._unrevealed_remaining = (self._rows * self._columns) - self._mines
        self._done = False
        self._win = False
        self._steps = 0

        if soft and self._board is not None:
            for tile in self._board:
                tile[STATE] = UNREVEALED_TILE
        else:
            self._board = []

            mine_indices = []
            to_make_mine = self._mines
            if load == False: # for when I want to load in a seed
                self._seed = random.random()
            random.seed(self._seed)
            choices = [c for c in range(self._rows * self._columns)]
            while to_make_mine > 0:
                idx = random.choice(choices)
                mine_indices.append(idx)
                choices.remove(idx)
                to_make_mine -= 1

            for row in range(self._rows):
                for column in range(self._columns):
                    neighbours = []
                    neighbouring_mines = 0
                    for i in range(row-1, row+2):
                        if i < 0 or i >= self._rows:
                            neighbours.append(-1)
                            neighbours.append(-1)
                            neighbours.append(-1)
                            continue
                        for j in range(column-1, column+2):
                            if i == row and j == column:
                                continue
                            if j < 0 or j >= self._columns:
                                neighbours.append(-1)
                                continue

                            n_idx = i * self._columns + j
                            neighbours.append(n_idx)
                            if n_idx in mine_indices:
                                neighbouring_mines += 1

                    # tiles
                    is_mine = (row * self._columns + column) in mine_indices
                    state = UNREVEALED_TILE  # unrevealed
                    self._board.append(
                        [state, neighbours, neighbouring_mines, is_mine])

        # create state
        return self.State()

    def render(self, mode="human", close=False):
        for row in range(self._rows):
            for column in range(self._columns):
                tile = self._board[row * self._columns + column]
                print("{:>2}".format(tile[STATE]), end=" ")
            print("")

        print("")

    def State(self):
        states = []
        neighbours = []
        for t in self._board:
            states.append(t[STATE])
        state_np = np.array(states, dtype=np.float32)
        
        # convolution
        if not self._flat:
            state_np = np.reshape(state_np, (-1, self._columns))
            state_np = np.expand_dims(state_np, axis=0)

        return state_np

    def GetAction(self, row, column):
        return row * self._columns + column

    def Play(self):
        self.reset()
        self.render()

        while True:
            i, j = input("choose tile [row column] - rows: " + str(self._rows-1) + " columns: " + str(self._columns-1) + "\n>").split()
            action = self.GetAction(int(i), int(j))
            state, reward, done, info = self.step(action)
            win = info['win']
            print(reward, done, win)
            self.render()

            if done:
                if win:
                    print("You won!")
                else:
                    print("You lose!")

                _ = input("press enter to restart")
                self.reset()
                self.render()


# env = Minesweeper_Text_v0(0.5)
# env.Play()