import collections
import numpy as np

from .SumTree import SumTree

Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'next_state', 'reward', 'done'])
class PrioritizedExperienceReplay:
    def __init__(self, capacity):
        self._sum_tree = SumTree(capacity)

        # avoid - hyperparameter to avoid some experiences that have 0 probabilty
        self._epsilon = 0.01 
        # greedy action probability
        self._alpha = 0.6 
        # self._beta-sampling, from initial value increasing to 1
        self._beta = 0.4 
        self._beta_inc = 0.001

        self._absolute_error_upper = 1 

    def Append(self, state, action, next_state, reward, done):
        experience = Experience(state, action, next_state, reward, done)
        max_priority = np.max(self._sum_tree._tree[-self._sum_tree._capacity:])

        if max_priority == 0:
            max_priority = self._absolute_error_upper

        self._sum_tree.Add(max_priority, experience)

    def Sample(self, batch_size):
        indices = []
        priorities = []

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        priority_segment = self._sum_tree.Total() / batch_size

        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)

            # pick a random value between a and b
            value = np.random.uniform(a, b)

            idx, priority, data = self._sum_tree.GetLeaf(value)

            priorities.append(priority)
            indices.append(idx)

            states.append(data.state)
            actions.append(data.action)
            next_states.append(data.next_state)
            rewards.append(data.reward)
            dones.append(data.done)


        indices_np = np.array(indices)

        states_np = np.array(states)
        actions_np = np.array(actions)
        next_states_np = np.array(next_states)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)

        sampling_probs = priorities / self._sum_tree.Total()
        weights = (self._sum_tree._entries * sampling_probs) ** -self._beta
        weights /= weights.max()
        weights_np = np.array(weights)

        # update after
        self._beta = np.min([1.0, self._beta + self._beta_inc])

        return states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np

    def BatchUpdate(self, tree_idx, errors):
        abs_errors = abs(errors + self._epsilon)
        clipped_errors = np.minimum(abs_errors, self._absolute_error_upper)
        priorities = np.power(clipped_errors, self._alpha)

        for ti, p in zip(tree_idx, priorities):
            self._sum_tree.Update(ti, p)

    def _GetPriority(self, error):
        return (np.abs(error) + self._epsilon) ** self._alpha

    def __len__(self):
        return self._sum_tree._entries