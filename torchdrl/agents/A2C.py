import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from .BaseAgent import BaseAgent


class A2C:
    def __init__(self, env, steps_count=2, steps_delta=1):
        # declare hyperparams

        # declare memory type?

        # declare network

        # declare num of envs for this agent

        if isinstance(env, (list, tuple))
            self._envs = env
        else:
            self._envs = [env]
        
        self._steps_count = steps_count
        self._steps_delta = steps_delta
        self._total_rewards = []
        self._total_steps = []

    def Train(self):
        batch = []
        for step_idx, exp in enumerate(self.PlayEpisode()):
            batch.append(exp)

            if len(batch) < 32:
                continue





    def Evaluate(self, episodes=100):
        pass

    def PlayEpisode(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []

        for env in self._envs:
            state = env.reset()
            states.append(state)

            histories.append(deque(maxlen=self._steps_count))
            cur_rewards.append(0)
            cur_steps.append(0)

        curr_iter = 0
        # TODO add some condition like max steps, or num total episodes?
        while True:
            # action per env
            actions = [None] * len(states)
            states_input = []
            states_indices = []

            # save the current state
            for idx, state in enumerate(states):
                states_input.append(state)
                states_indices.append(idx)

            # get the action for each game
            states_actions = self.Act(states)
            for idx, action in enumerate(states_actions):
                game_idx = states_indices[idx]

                # set the action
                actions[game_idx] = [action]

            # perform the action for each environment and 
            # vectorized the outputs
            for env_idx, (env, action_n) in enumerate(zip(self._envs, actions)):
                next_state, reward, done, info = env.step(action_n[0])
                next_state_n, reward_n, done_n = [next_state], [reward], [done]

                for idx, (action, next_state, reward, done) in enumerate(zip(action_n, next_state_n, reward_n, done_n)):
                    state = states[idx]
                    history = histories[idx]
                    cur_rewards[idx] += reward
                    cur_steps[idx] += 1

                    # append the history of the env
                    history.append((state, action, reward, done))

                    if len(history) == self._steps_count and curr_iter % self._steps_delta == 0:
                        # set up the first and last experience
                        yield(self.FirstLast(tuple(history)))
                        

                    states[idx] = next_state
                    if done:
                        # send history if it didn't manage to be sent before
                        if 0 < len(history) < self._steps_count:
                            yield(self.FirstLast(tuple(history)))
                        
                        while len(history) > 1:
                            history.popleft()
                            yield(self.FirstLast(tuple(history)))

                        self._total_rewards.append(cur_rewards[idx])
                        self._total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0

                        states[idx] = env.reset()
                        history.clear()
                curr_iter += 1

    # AKA n-step
    def FirstLast(self, history):
        for exp in history:
            if exp[-1][3] and len(exp) <= self._steps_count:
                last_state = None
                elems = exp
            else:
                # isnt done
                last_state = exp[-1][0]
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self._gamma
                total_reward += e[2]

        yield (exp[0][0], exp[0][1], total_reward, last_state)

    def Act(self, state):
        state_np = np.array(state, dtype=np.float32)
        state_t = torch.tensor(state_np, device=self._device)

        probs_v = self._net(state_t)
        probs_v = F.softmax(probs_v, dim=1)

        actions = []
        for probs in probs_v:
            actions.append(np.random.choice(len(probs), p=probs))

        return np.array(actions)

    def Learn(self):
        pass

    def CalculateErrors(self, batch, batch_size, gamma):

        

    def Save(self, folderpath, filename):
        pass

    def Load(self, filepath):
        pass
