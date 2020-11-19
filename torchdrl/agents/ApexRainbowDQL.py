import os
import time
import random
import threading
import itertools
import copy

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

import numpy as np

from .BaseAgent import BaseAgent

from .RainbowDQL import RainbowDQL

from ..neural_networks.NoisyDuelingCategoricalNetwork import NoisyDuelingCategoricalNetwork
from ..representations.Plotter import Plotter

from ..data_structures.ExperienceReplay import ExperienceReplay
from ..data_structures.ApexExperienceReplay import ApexExperieceReplay
from ..data_structures.UniformExperienceReplay import UniformExperienceReplay
from ..data_structures.NStepPrioritizedExperienceReplay import NStepPrioritizedExperienceReplay
from ..data_structures.PER import PER

from ..neural_networks.ConstraintNetwork import ConstraintNetwork

# TODO try again. the other apex worked better...
# feels bad.

class ApexRainbowDQLActor(RainbowDQL):
    def __init__(self, config, agent_id, learner):
        # need to set a device first.
        apex_parameters = config['apex_parameters']
        self._device = apex_parameters['actor_device']

        super(ApexRainbowDQLActor, self).__init__(config)
        
        self._agent_id = agent_id
        self._name = self._name + "_" + str(agent_id)
        self._learner = learner

        # every n steps
        self._sync_steps = 0

        self._learner_sync_frequency = apex_parameters['learner_sync_frequency']
        self._mini_batch_size = apex_parameters['mini_batch_size']
        self._internal_memory = ExperienceReplay(self._mini_batch_size, self._input_shape)
        self._memory = ApexExperieceReplay(self._mini_batch_size, self._input_shape)
        self._request_update = False
        self._memories_waiting = False

    def PlayEpisode(self, evaluate):
        done = False
        episode_reward = 0
        self._steps = 0
        
        state = self._env.reset()
        while self._steps != self._max_steps and not done:
            # Noisy - No epsilon
            action = self.Act(state)
                
            next_state, reward, done, info = self._env.step(action)
            transition = (state, action, next_state, reward, done)

            self.SaveMemory(transition)            
            self.PrepareMemories()

            episode_reward += reward
            state = next_state

            self._steps += 1
            self._total_steps += 1

            # update the agent
            self._sync_steps += 1
            if self._sync_steps % self._learner_sync_frequency == 0:
                # self._request_update = True 
                self.SyncToLearner()
                self._sync_steps = 0
        
            # while self._request_update:
            #     time.sleep(0.0001)


        self.SyncToLearner()
        time.sleep(0.0001)
            
        return episode_reward, self._steps, info

    def SaveMemory(self, transition):
        # if self._n_steps > 1:
        #     transition = self._memory_n_step.Append(*transition)
            
            # if a n_step transition was returned from the memory.
        if transition:
            self._internal_memory.Append(*transition)

    def PrepareMemories(self):
        if len(self._internal_memory) >= self._mini_batch_size:
            while self._memories_waiting:
                time.sleep(0.0001)

            states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np = self._internal_memory.Pop(self._mini_batch_size)
            states_t, actions_t, next_states_t, rewards_t, dones_t, weights_t = self.ConvertNPWeightedMemoryToTensor(states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np)
            errors = self.CalculateErrors(states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, self._mini_batch_size, self._gamma)
            errors_np = errors.cpu().detach().numpy()

            self._memory.BatchAppend(states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, self._mini_batch_size)
            self._memories_waiting = True

    # I want the manager to do the checkpointing from the actors
    def Checkpoint(self):
        pass

    def SyncToLearner(self):
        self.UpdateNetwork(self._learner._net, self._net)
        self.UpdateNetwork(self._learner._target_net, self._target_net)

class ApexRainbowDQLLearner(RainbowDQL):
    def __init__(self, config):
        # need to set a device first.
        apex_parameters = config['apex_parameters']
        self._device = apex_parameters['learner_device']

        super(ApexRainbowDQLLearner, self).__init__(config)
        self._config = config
        self._hyperparameters = self._config['hyperparameters']
        self._copying = False

        memory_type = self._hyperparameters['memory_type']
        memory_size = self._hyperparameters['memory_size']
        # memory
        if memory_type == "PER":
            alpha = self._hyperparameters['alpha']
            beta = self._hyperparameters['beta']
            priority_epsilon = self._hyperparameters['priority_epsilon']
            self._memory = PER(memory_size, self._input_shape, alpha, beta, priority_epsilon)
        else:
            self._memory = UniformExperienceReplay(memory_size, self._input_shape)

        if self._n_steps > 1:
            self._memory_n_step = PER(memory_size, self._input_shape, alpha, beta, priority_epsilon, self._n_steps, self._gamma)


class ApexRainbowDQL:
    def __init__(self, config, learner_class=ApexRainbowDQLLearner, actor_class=ApexRainbowDQLActor):
        assert issubclass(learner_class, BaseAgent)
        assert issubclass(actor_class, BaseAgent)

        self._config = config
        self._hyperparameters = self._config['hyperparameters']
        self._apex_parameters = self._config['apex_parameters']
        self._learner_sync_frequency = self._apex_parameters['learner_sync_frequency']

        self._batch_size = self._hyperparameters['batch_size']

        self._learner = learner_class(self._config)
        self._actors = []

        self._mini_batch_size = self._apex_parameters['mini_batch_size']

        # actors
        self._num_actors = self._apex_parameters['num_actors']
        self._num_threads = self._apex_parameters['num_threads']

        for i in range(self._num_actors):
            actor = actor_class(config, i, self._learner)
            # self._CopyLearner(actor)

            self._actors.append(actor)

        self._num_ep_last_save = 0
        self._checkpoint_freq = self._config['checkpoint_frequency'] * self._num_actors

        self._actor_finished = False
        self._learns = 0
        self._episode_infos = []

        print("Finished initialization")

    def Train(self):
        learner_thread = threading.Thread(target=self.TrainLearner, args=())
        learner_thread.daemon = True
        learner_thread.start()

        print("Finished starting learner")

        actors_per_thread = self._num_actors // self._num_threads
        extra_actors = self._num_actors % self._num_threads
        actor_groups = []
        actor_threads = []
        actor_idx = 0

        i = 0
        while i < self._num_threads and (actors_per_thread > 0 or extra_actors > 0):
            actor_groups.append([])
            for j in range(actors_per_thread):
                actor_groups[i].append(self._actors[actor_idx])
                actor_idx += 1
            if extra_actors > 0:
                actor_groups[i].append(self._actors[actor_idx])
                extra_actors -= 1
                actor_idx += 1

            i += 1

        print("Finished grouping actors")
        
        for group in actor_groups:
            thread = threading.Thread(target=self.TrainActors, args=(group))
            thread.daemon = True

            actor_threads.append(thread)

        print("Finished creating threads")

        for thread in actor_threads:
            thread.start()
        
        print("Finished starting threads")
        

        print("Beginning training")
        while not self._actor_finished:

            self._TransferMemories()

            # if self._learns > self._learner_sync_frequency:
                # self._learns = 0
            # for actor in self._actors:
            #     self._CopyLearner(actor)



            time.sleep(0.0001)
            while self._episode_infos:
                self._num_ep_last_save += 1
                yield self._episode_infos.pop(0)

            if self._num_ep_last_save > self._checkpoint_freq:
                self.Save()
                self._num_ep_last_save = 0
        
        self.Save()


    def TrainActors(self, *actors):
        trains = []
        if isinstance(actors, tuple):
            for actor in actors:
                trains.append(actor.Train())        
            zipped_trains = zip(*trains)

            chained_zipped_trains = itertools.chain(zipped_trains)
            for train in zipped_trains:
                for episode_info in train:
                    self._episode_infos.append(episode_info)
        else:
            for episode_info in actors.Train():
                self._episode_infos.append(episode_info)
        
        # using chain and zipping the trains when one finishes, all
        # trains will finish automatically.
        self._actor_finished = True

    def TrainLearner(self):
        while not self._actor_finished:
            if len(self._learner._memory) > self._batch_size and len(self._learner._memory) > self._learner._warm_up:
                self._learner.Learn()
                # time.sleep(0.0001)

                self._learns += 1

    def _TransferMemories(self):
        for actor in self._actors:
            if actor._memories_waiting:
                states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, _ = actor._memory.Pop(self._mini_batch_size)
                for i in range(self._mini_batch_size):
                    transition = (states_np[i], actions_np[i], next_states_np[i], rewards_np[i], dones_np[i], errors_np[i])
                    if self._hyperparameters['n_steps'] > 1:
                        transition = self._learner._memory_n_step.Append(*transition)
                    
                    if transition:
                        self._learner._memory.Append(*transition)

                actor._memories_waiting = False


    def _CopyLearner(self, actor, tau=1.0):
        if actor._request_update:
            self._learner._copying = True
            actor.UpdateNetwork(self._learner._net, actor._net, tau)
            actor.UpdateNetwork(self._learner._target_net, actor._target_net, tau)
            self._learner._copying = False
            actor._request_update = False

    def Save(self):
        best_mean_score = float('-inf')
        actor_idx = 0
        for i, actor in enumerate(self._actors):
            if actor._best_mean_score > best_mean_score:
                best_mean_score = actor._best_mean_score
                actor_idx = i 

        
        # now save the actor
        actor = self._actors[actor_idx]

        folderpath = self._config['checkpoint_root'] + "/" + self._config['name']
        filename = "episode_" + str(actor._episode) + "_score_" + str(round(actor._episode_mean_score, 2)) + ".pt"
        self._actors[actor_idx].Save(folderpath, filename)
            

    def Load(self, filepath):
        self._learner.Load(filepath)

        for actor in self._actors:
            actor.UpdateNetwork(self._learner._net, actor._net)
            actor.UpdateNetwork(self._learner._target_net, actor._target_net)