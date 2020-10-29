import collections
import numpy as np
import random
import math

from .ExperienceReplay import ExperienceReplay
from .SegmentTree import SegmentTree


# TODO make base class experience replay
Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'next_state', 'reward', 'done'])
class NStepPrioritizedExperienceReplay:
    def __init__(self, capacity, input_shape, alpha, beta, priority_epsilon, n_step=1, gamma=0.99):
        # super(NStepPrioritizedExperienceReplay, self).__init__()
        self._capacity = capacity
        self._input_shape = input_shape
        self._n_step = n_step
        self._gamma = gamma

        # data is stored here
        self._sum_tree = SegmentTree(capacity)
        # tranisition buffer for N-step learning
        self._n_step_buffer = collections.deque(maxlen=self._n_step)


        # avoid - hyperparameter to avoid some experiences that have 0 probabilty
        self._priority_epsilon = priority_epsilon
        # greedy action probability
        self._alpha = alpha
        # sampling, from initial value increasing to 1
        self._beta = beta
        self._beta_inc = 0.00001 # TODO maybe parameter?

        self._max_priority = 1 
        self._min_priority = math.inf

    def BatchAppend(self, states, actions, next_states, rewards, dones, priorities, batch_size):
        for i in range(batch_size):
            self.Append(states[i], actions[i], next_states[i], rewards[i], dones[i], priorities[i] + self._priority_epsilon)

    def Append(self, state, action, next_state, reward, done, priority=0):
        # experience = Experience(state, action, next_state, reward, done)
        experience = (state, action, next_state, reward, done)
        self._n_step_buffer.append(experience)

        # If step transition is ready
        if len(self._n_step_buffer) >= self._n_step:
            n_next_state, n_reward, n_done, = self._GetNStepInfo(self._n_step_buffer, self._gamma)
            n_state, n_action = self._n_step_buffer[0][:2]
            
            n_experience = (n_state, n_action, n_next_state, n_reward, n_done)

            if priority > self._max_priority:
                self._max_priority = priority
            
            if priority < self._min_priority:
                self._min_priority = priority

            if priority == 0:
                priority = self._max_priority ** self._alpha
                
            self._sum_tree.Add(priority, n_experience)
        else:
            return ()

        return self._n_step_buffer[0]

    def Sample(self, batch_size):
        self._ready = False
        indices = []
        priorities = []

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        priority_segment = (self._sum_tree.Total() - self._min_priority) / batch_size

        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)

            # pick a random value between a and b
            value = random.uniform(a, b)

            idx, priority, data = self._sum_tree.GetLeaf(value)
            # print(self._sum_tree.Total(), idx, value)

            # with open('error.txt', 'w') as f:
            #     for item in self._sum_tree._tree.tolist():
            #         f.write("%s\n" % str(item))

            #     f.write("%s\n" % str(value))
            


            priorities.append(priority)
            indices.append(idx)

            # states.append(data.state)
            # actions.append(data.action)
            # next_states.append(data.next_state)
            # rewards.append(data.reward)
            # dones.append(data.done)

            states.append(data[0])
            actions.append(data[1])
            next_states.append(data[2])
            rewards.append(data[3])
            dones.append(data[4])


        indices_np = np.array(indices)

        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        next_states_np = np.array(next_states, dtype=np.float32)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.int64)

        weights = []
        for i in indices:

            p_min = self._min_priority / self._sum_tree.Total() + 1e-5

            max_weight = (p_min * self._sum_tree._entries) ** (-self._beta)

            # calculate weights
            p_sample = self._sum_tree._tree[i] / self._sum_tree.Total()
            weight = ((p_sample * self._sum_tree._entries) + 1e-5) ** (-self._beta)
            weight = weight / max_weight
            weights.append(weight)

        weights_np = np.array(weights)


        # sampling_probs = priorities / self._sum_tree.Total()
        # weights = (self._sum_tree._entries * sampling_probs) ** -self._beta
        # weights /= weights.max()
        # weights_np = np.array(weights)

        # update after
        self._beta = np.min([1.0, self._beta + self._beta_inc])

        return states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np

    def SampleBatchFromIndices(self, indices):
        indices = indices - (self._capacity - 1)
        experiences = self._sum_tree._data[indices]
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for exp in experiences:
            states.append(exp[0])
            actions.append(exp[1])
            next_states.append(exp[2])
            rewards.append(exp[3])
            dones.append(exp[4])

        states_np = np.array(states)
        actions_np = np.array(actions)
        next_states_np = np.array(next_states)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)

        return (states_np, actions_np, next_states_np, rewards_np, dones_np)

    def BatchUpdate(self, indices, errors):
        for idx, error in zip(indices, errors):
            self._sum_tree.Update(idx, self._GetPriority(error))

    def _GetPriority(self, error):
        return (error + self._priority_epsilon) ** self._alpha

    def _GetNStepInfo(self, n_step_buffer, gamma):
        next_state, reward, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            n_s, r, d = transition[-3:]

            reward = r + gamma * reward * (1-d)
            next_state, done = (n_s, d) if d else (next_state, done)
        
        return next_state, reward, done

    def __len__(self):
        return self._sum_tree._entries
