import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
import numpy as np
from collections import deque

from .BaseAgent import BaseAgent

from ..neural_networks.TwoHeadedNetwork import TwoHeadedNetwork

# TODO tensorboard tracking 
# TODO give atleast one linear unit to the heads.


class A2C:
    def __init__(self, env, lr=0.0001, gamma=0.99, entropy_beta=0.01, 
                batch_size=128, reward_steps=4, 
                clip_grad=0.1, steps_count=5, steps_delta=1):

        self._device = 'cuda'

        # declare hyperparams
        self._lr = lr
        self._gamma = gamma
        self._entropy_beta = entropy_beta
        self._batch_size = batch_size
        self._reward_steps = reward_steps
        self._clip_grad = clip_grad
        self._steps_count = steps_count
        self._steps_delta = steps_delta

        # declare num of envs for this agent
        if isinstance(env, (list, tuple)):
            self._envs = env
        else:
            self._envs = [env]

        # TODO modularize this
        # hard declare network for now
        input_shape = self._envs[0].observation_space.shape
        actions = self._envs[0].action_space.n
        value = 1
        hidden_layers = [1024, 1024, 512]
        activations = ['relu', 'relu', 'relu']
        final_activation = None
        convo = {
            "filters": [32],
            "kernels": [[1,1]],
            "strides": [1],
            "activations": ['relu'],
            "paddings": [0],
            "pools": [],
            "flatten": True
        }
        convo = None
        self._net = TwoHeadedNetwork(input_shape, actions, value, hidden_layers, activations, final_activation, convo).to(self._device)
        print(self._net)
        # optimizer
        self._optimizer = optim.Adam(self._net.parameters(), lr=self._lr, eps=1e-3)

        self._steps_count = steps_count
        self._steps_delta = steps_delta
        self._total_rewards = []
        self._total_steps = []
        self._episodes = []

    def Train(self):
        batch = []
        
        # each loop adds some experiences onto the batch.
        for step_idx, exp in enumerate(self.PlayEpisode()):
            batch.append(exp)

            if len(batch) >= self._batch_size:
                # learn
                self.Learn(batch)
                batch.clear() # empty the batch and continue

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
            self._episodes.append(0)

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
            global_idx = 0
            for env_idx, (env, action_n) in enumerate(zip(self._envs, actions)):
                next_state, reward, done, info = env.step(action_n[0])
                next_state_n, reward_n, done_n = [next_state], [reward], [done]

                for local_idx, (action, next_state, reward, done) in enumerate(zip(action_n, next_state_n, reward_n, done_n)):
                    idx = global_idx + local_idx
                    state = states[idx]
                    history = histories[idx]
                    cur_rewards[idx] += reward
                    cur_steps[idx] += 1

                    # append the history of the env
                    history.append((state, action, reward, done))

                    if len(history) == self._steps_count and curr_iter % self._steps_delta == 0:
                        # set up the first and last experience
                        yield self.FirstLast(tuple(history))
                        

                    states[idx] = next_state
                    if done:
                        # print stuff out I guess
                        print("ep:", self._episodes[idx], "steps:", cur_steps[idx], "rewards:", cur_rewards[idx])

                        # send history if it didn't manage to be sent before
                        if 0 < len(history) < self._steps_count:
                            yield self.FirstLast(tuple(history))
                        
                        while len(history) > 1:
                            history.popleft()
                            yield self.FirstLast(tuple(history))

                        # increment and reset
                        self._total_rewards.append(cur_rewards[idx])
                        self._total_steps.append(cur_steps[idx])
                        self._episodes[idx] += 1
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0

                        states[idx] = env.reset()
                        history.clear()
                global_idx += 1
            curr_iter += 1

    # AKA n-step -> compresses the steps into one via rewards
    def FirstLast(self, history):
        if history[-1][3] and len(history) <= self._steps_count:
            last_state = None
            elems = history
        else:
            # isnt done
            last_state = history[-1][0]
            elems = history[:-1]
        total_reward = 0.0
        for e in reversed(elems):
            total_reward *= self._gamma
            total_reward += e[2]

        return (history[0][0], history[0][1], total_reward, last_state)

    def Act(self, state):
        state_np = np.array(state, dtype=np.float32)
        state_t = torch.tensor(state_np, device=self._device)

        probs_t = self._net(state_t)[0]
        probs_t = F.softmax(probs_t, dim=1)
        probs_np = probs_t.cpu().detach().numpy()

        actions = []
        for probs in probs_np:
            actions.append(np.random.choice(len(probs), p=probs))

        return np.array(actions)

    def Learn(self, batch):
        self._optimizer.zero_grad()

        loss_policy, loss_value_t, loss_entropy = self.CalculateErrors(batch, self._batch_size, self._gamma)

        # calculate policy gradients 
        loss_policy.backward(retain_graph=True)
        
        # apply entropy
        loss_value = loss_entropy + loss_value_t
        loss_value_t.backward()

        nn_utils.clip_grad_norm_(self._net.parameters(), self._clip_grad)
        
        
        self._optimizer.step()

        loss_total = loss_value + loss_policy

        # TODO add to tensorboard
        # loss_policy
        # loss_value
        # loss_entropy
        # loss_total

    def CalculateErrors(self, batch, batch_size, gamma):
        states_t, actions_t, target_vals_t = self.UnpackBatch(batch, gamma)

        policy_t, value_t = self._net(states_t)
        loss_value_t = F.mse_loss(value_t.squeeze(-1), target_vals_t)

        log_probs_t = F.log_softmax(policy_t, dim=1)
        adv_t = target_vals_t - value_t.detach()
        log_probs_actions_t = adv_t * log_probs_t[range(batch_size), actions_t]
        loss_policy = -log_probs_actions_t.mean()

        # entropy loss
        probs_values = F.softmax(policy_t, dim=1)
        loss_entropy = self._entropy_beta * (probs_values * log_probs_t).sum(dim=1).mean()
        
        # for logging
        # grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
        #                         for p in net.parameters()
        #                         if p.grad is not None])


        return loss_policy, loss_value_t, loss_entropy

        # loss_value.backward()
        # nn_utils.clip_grad_norm_(net.parameters(), 0.1)
        # self._optimizer.step()

        # # get total loss to return
        # loss_value += loss_policy

        # TODO add to tensorboard 

    def UnpackBatch(self, batch, gamma):
        states = []
        actions = []
        rewards = []
        not_done_idx = []
        last_states = []

        for idx, exp in enumerate(batch):
            states.append(np.array(exp[0], copy=False))
            actions.append(int(exp[1]))
            rewards.append(exp[2])
            if exp[3] is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp[3], copy=False))

        states_t = torch.tensor(np.array(states, copy=False), dtype=torch.float32, device=self._device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self._device)
        
        # reward time
        rewards_np = np.array(rewards, dtype=np.float32)
        if not_done_idx:
            last_states_t = torch.tensor(np.array(last_states, copy=False), dtype=torch.float32, device=self._device)
            last_vals_t = self._net(last_states_t)[1] # get vals of the state
            last_vals_np = last_vals_t.cpu().detach().numpy()[:, 0]
            last_vals_np *= gamma ** self._reward_steps 
            rewards_np[not_done_idx] += last_vals_np
        
        target_vals_t = torch.tensor(rewards_np, dtype=torch.float32, device=self._device)

        return states_t, actions_t, target_vals_t

    def Save(self, folderpath, filename):
        pass

    def Load(self, filepath):
        pass
