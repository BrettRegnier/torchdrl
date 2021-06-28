import os
import copy
import math
import random
import numpy as np
import gym

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import torchdrl.tools.Helper as Helper

from torchdrl.agents.Agent import Agent

class QLearningAgent(Agent):
    def __init__(
            self, 
            name, 
            envs,
            network,
            action_function,
            rl_algorithm,
            optimizer,
            batch_size,
            memory,
            memory_n_step=None,
            scheduler=None,
            clip_grad=-1,
            gamma=0.99,
            max_steps_per_episode=-1,
            target_update_frequency=100,
            tau=1,
            visualize=False,
            visualize_frequency=-1,
            seed=-1, # REMOVE
            warm_up=0,
            oracle=None,
            device="cpu"
        ):

        self._name = name
        if isinstance(envs, (list, tuple)):
            self._envs = envs
        else:
            self._envs = [envs]
        self._network = network
        self._target_network = copy.deepcopy(self._network)
        self._rl_algorithm = rl_algorithm
        self._action_function = action_function
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._memory = memory
        self._memory_n_step = memory_n_step
        self._scheduler = None # TODO implement

        # hyperparameters
        self._clip_grad = clip_grad
        self._gamma = gamma
        self._target_update_frequency = target_update_frequency
        self._target_update_steps = 0
        self._tau = tau
        self._max_steps_per_episode = max_steps_per_episode

        self._visualize = visualize
        self._visualize_frequency = visualize_frequency
        self._visualize_count = 0

        self._warm_up = warm_up
        self._oracle = oracle
        self._device = device 

        self._train_last_state_dict = 0
        self._evaluate_last_state_dict = 0
        
        # Assume: object will be input
        # Assume: all envs are the same
        if isinstance(self._envs[0].observation_space, (gym.spaces.Tuple, gym.spaces.Dict)):
            self._state_is_tuple = True
        else:
            self._state_is_tuple = False    

        # TODO?
        self._steps_history = []
        self._stop = False
        
        self._ep_info = {}

        # after variable init
        self._network.train()
        self._target_network.eval()
        Helper.UpdateNetwork(self._network, self._target_network)

        print(self._network)

    def PlayEpisode(self, evaluate=False):
        states = [None] * len(self._envs)
        actions = [None] * len(self._envs)
        # next_states = [None] * len(self._envs)
        # rewards = [None] * len(self._envs)       
        # dones = [True] * len(self._envs)
        infos = [{} for _ in range(len(self._envs))]

        episode_rewards = [0] * len(self._envs)
        episode_loss = [0] * len(self._envs)

        steps = [0] * len(self._envs)   

        # Init all envs 
        for i, env in enumerate(self._envs):
            states[i] = env.reset()

        while not self._stop:
            actions = self.GetActions(states)
            for idx, (env, action) in enumerate(zip(self._envs, actions)):
                if self._stop:
                    break

                next_state, reward, done, info = env.step(action)

                if not evaluate:
                    # Could maybe combine these two so there is less to change
                    # in Ape-X
                    episode_loss[idx] += self.MemoryLearn(
                        states[idx], 
                        actions[idx], 
                        next_state, 
                        reward, 
                        done
                    )

                episode_rewards[idx] += reward
                steps[idx] += 1

                if self._visualize:
                    self._visualize_count = (self._visualize_count + 1) % self._visualize_frequency
                    if self._visualize_count == 0:
                        env.render()

                if steps[idx] == self._max_steps_per_episode or done:
                    env.close()
                    infos[idx].update(info)
                    yield (episode_rewards[idx], steps[idx], round(episode_loss[idx], 2), infos[idx])

                    # RESET
                    states[idx] = env.reset()

                    steps[idx] = 0
                    episode_rewards[idx] = 0
                    episode_loss[idx] = 0
                else:
                    states[idx] = next_state

    @torch.no_grad()
    def EvaluateEpisode(self, episodes=100):
        states = [None] * len(self._envs)
        actions = [None] * len(self._envs)
        infos = [{} for _ in range(len(self._envs))]

        episode_rewards = [0] * len(self._envs)
        steps = [0] * len(self._envs)   

        # Init all envs 
        for idx, env in enumerate(self._envs):
            states[idx] = env.reset()

        test_idx = 1
        while test_idx < episodes + 1:
            actions = self.GetActions(states, evaluate=True)
            env_selection = min(len(self._envs), (episodes+1) - test_idx)
            envs = self._envs[:env_selection]
            actions = actions[:env_selection]

            for idx, (env, action) in enumerate(zip(envs, actions)):
                next_state, reward, done, info = env.step(action)

                episode_rewards[idx] += reward
                steps[idx] += 1

                if steps[idx] == self._max_steps_per_episode or done:
                    yield (steps[idx], episode_rewards[idx], info)
                    test_idx += 1

                    states[idx] = env.reset()

                    steps[idx] = 0
                    episode_rewards[idx] = 0
                else:
                    states[idx] = next_state

    @torch.no_grad()
    def GetActions(self, states, evaluate=False):
        if self._oracle and not evaluate:
            return self._oracle.Act(states)
        else:
            states_t = Helper.ConvertStateToTensor(states, device=self._device, state_is_tuple=self._state_is_tuple)

            q_values = self._network(states_t)
            return self._action_function(q_values, evaluate)

    def Learn(self):
        if len(self._memory) >= self._batch_size and len(self._memory) >= self._warm_up:
            loss = self.Update()

            return loss.detach().cpu().numpy()
        return 0
            
    def Update(self):
        states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t = self.SampleMemoryT(
            self._batch_size)
            
        # get errors
        errors = self._rl_algorithm(
            self._network,
            self._target_network,
            states_t, 
            actions_t, 
            next_states_t, 
            rewards_t,
            dones_t, 
            self._batch_size, 
            self._gamma
        )

        weights_t = weights_t.reshape(-1, 1)
        # Prioritized Experience Replay weight importancing
        loss = torch.mean(errors * weights_t)

        # n-step learning_ with one-step to prevent high-variance
        if self._memory_n_step:
            gamma = self._gamma ** self._memory_n_step.GetNStep()
            states_np, actions_np, next_states_np, rewards_np, dones_np, _ = self._memory_n_step.SampleBatchFromIndices(
                indices_np)
            states_t, actions_t, next_states_t, rewards_t, dones_t = self.ConvertNPMemoryToTensor(
                states_np, actions_np, next_states_np, rewards_np, dones_np)
            errors_n = self._rl_algorithm(
                states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, self._batch_size, gamma)
            errors += errors_n

            loss = torch.mean(errors * weights_t)

        self._optimizer.zero_grad()
        loss.backward()
        if self._clip_grad > -1:
            clip_grad_norm_(self._network.parameters(), self._clip_grad)
        self._optimizer.step()
        
        if self._target_update_steps % self._target_update_frequency == 0:
            Helper.UpdateNetwork(self._network, self._target_network, self._tau)
            # self._target_network.load_state_dict(self._network.state_dict())
            self._target_update_steps = 0

        self._target_update_steps += 1
        
        # for PER
        updated_priorities = errors.detach().cpu().numpy()
        self._memory.BatchUpdate(indices_np, updated_priorities)
        
        return loss

    # def state_dict(self, folderpath, filename):
    #     if os.path.exists(folderpath):
    #         list_of_files = os.listdir(folderpath)
    #         if len(list_of_files) >= self._state_dict_max_num:
    #             full_path = [folderpath + "/{0}".format(x) for x in list_of_files]

    #             oldest_file = min(full_path, key=os.path.getctime)
    #             os.remove(oldest_file)
        
    #     self.Save(folderpath, filename)

    def GetSaveInfo(self):
        save_info = {
            'name': self._name,
            'network': self._network.state_dict(),
            'target_network': self._target_network.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'hyperparameters': {
                'rl_algorithm': self._rl_algorithm.name,
                'architecture': self._network.__str__(),
                'action_function': self._action_function.state_dict(),
                'target_update_frequency': self._target_update_frequency,
                'clip_grad': self._clip_grad,
                'gamma': self._gamma,
                'max_steps_per_episode': self._max_steps_per_episode,
                'batch_size': self._batch_size
            }
        }
        
        if self._scheduler:
            save_info['scheduler'] = self._scheduler.state_dict()
            save_info['hyperparameters']['sched_step_size'] = self._scheduler.state_dict()['step_size']
            save_info['hyperparameters']['sched_gamma'] = self._scheduler.state_dict()['gamma']
            save_info['hyperparameters']['learning_rate'] = self._optimizer.state_dict()['param_groups'][0]['initial_lr']
        else:
            save_info['hyperparameters']['learning_rate'] = self._optimizer.state_dict()['param_groups'][0]['lr']

        return save_info

    def Save(self, folderpath, filename):
        save_info = self.GetSaveInfo()
        Helper.SaveAgent(folderpath, filename, save_info)

    def Load(self, filepath):
        state_dict = Helper.LoadAgent(filepath)

        self.LoadStateDict(state_dict)
        
        if self._scheduler:
            self._scheduler.load_state_dict(state_dict['scheduler'])

    def LoadStateDict(self, state_dict):
        self._name = state_dict['name']
        self._network.load_state_dict(state_dict['network'])
        self._target_network.load_state_dict(state_dict['target_network'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._action_function.load_state_dict(state_dict['action_function'])

    def MemoryLearn(self, states, action, next_state, reward, done):
        self.StoreMemory(states, action, next_state, reward, done)
        return self.Learn()

    def StoreMemory(self, states, action, next_state, reward, done):
        transition = (states, action, next_state, reward, done)
        if self._memory_n_step:
            transition = self._memory_n_step.Append(*transition)

        if transition:
            self._memory.Append(*transition)

    # HELPER FUNCTIONS
    def Stop(self):
        self._stop = True

    def SampleMemoryT(self, batch_size):
        states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np, indices_np = self._memory.Sample(batch_size)
        states_t, actions_t, next_states_t, rewards_t, dones_t, weights_t = self.ConvertNPWeightedMemoryToTensor(states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np)
        return states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t

    def ConvertNPWeightedMemoryToTensor(self, states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np):
        weights_t = torch.tensor(weights_np, dtype=torch.float32, device=self._device)
        states_t, actions_t, next_states_t, rewards_t, dones_t = self.ConvertNPMemoryToTensor(states_np, actions_np, next_states_np, rewards_np, dones_np)
        return states_t, actions_t, next_states_t, rewards_t, dones_t, weights_t
        
    def ConvertNPMemoryToTensor(self, states_np, actions_np, next_states_np, rewards_np, dones_np):
        if self._state_is_tuple:
            states_t = []
            for states in states_np:
                states_t = torch.tensor(states, dtype=torch.float32, device=self._device)
                states_t.append(states_t)

            next_states_t = []
            for next_state in next_states_np:
                next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self._device)
                next_states_t.append(next_state_t)            
        else:
            states_t = torch.tensor(states_np, dtype=torch.float32, device=self._device)
            next_states_t = torch.tensor(next_states_np, dtype=torch.float32, device=self._device)
            
        actions_t = torch.tensor(actions_np, dtype=torch.int64, device=self._device)
        rewards_t = torch.tensor(rewards_np.reshape(-1, 1), dtype=torch.float32, device=self._device)
        dones_t = torch.tensor(dones_np.reshape(-1, 1), dtype=torch.int64, device=self._device)

        return states_t, actions_t, next_states_t, rewards_t, dones_t

    def LoadNetworkStateDict(self, state_dict):
        self._network.load_state_dict(state_dict)
        self._target_network.load_state_dict(state_dict)