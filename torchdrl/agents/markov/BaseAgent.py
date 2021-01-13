import os
import math
import torch
import torch.optim as optim
import random
import numpy as np
import gym

from torchdrl.data_structures.ExperienceReplay import ExperienceReplay
from torchdrl.data_structures.UniformExperienceReplay import UniformExperienceReplay
from torchdrl.data_structures.PrioritizedExperienceReplay import PrioritizedExperienceReplay
from torchdrl.representations.Plotter import Plotter


# TODO clean things up. lets get this streamlined.
class BaseAgent(object):
    def __init__(self, env, **kwargs):
        self._env = env
        self._kwargs = kwargs
        self._hyperparameters = self._kwargs['hyperparameters']

        self._name = self._kwargs['name']

        self._checkpoint_episode_frequency = self._kwargs['checkpoint_episode_frequency']
        self._checkpoint_root = self._kwargs['checkpoint_root']
        self._checkpoint_max_num = self._kwargs['checkpoint_max_num']

        self._device = self._kwargs['device']

        if self._kwargs['log']:
            self._plotter = Plotter()
        self._show_log = self._kwargs['show_log']
        self._show_log_frequency = self._kwargs['show_log_frequency']

        self._reward_goal = self._kwargs['reward_goal']
        self._reward_window = self._kwargs['reward_window']

        seed = self._kwargs['seed']
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            self._env.seed(seed)

        self._visualize = self._kwargs['visualize']
        self._visualize_frequency = self._kwargs['visualize_frequency']

        self._warm_up = self._kwargs['warm_up']

        self._max_steps = self._kwargs['max_steps']
        self._max_episode = self._kwargs['max_episodes']

        self._action_type = "DISCRETE" if self._env.action_space.dtype == np.int64 else "CONTINUOUS"
        self._n_actions = self._env.action_space.n if self._action_type == "DISCRETE" else env.action_space.shape[0]
        
        self._episode_score = 0
        self._episode_scores = []
        self._episode_mean_score = 0
        self._episode = 0
        self._best_score = float("-inf")
        self._prev_best_mean_score = float("-inf")
        self._best_mean_score = float("-inf")

        # begin defining all shared hyperparamters
        self._gamma = self._hyperparameters['gamma']
        self._n_steps = self._hyperparameters['n_steps']
        self._batch_size = self._hyperparameters['batch_size']
        
        # assumed that a object will be input
        if isinstance(self._env.observation_space, (gym.spaces.Tuple, gym.spaces.Dict)):
            input_shape = []
            for space in self._env.observation_space:
                try:
                    input_shape.append(space.shape)
                except:
                    input_shape.append(space.n)
            self._state_is_tuple = True    
            # for sp in self._env.observation_space:
            #     if type(sp) is gym.spaces.Discrete:
            #         input_shape.append([sp.n,])
            #     else:
            #         input_shape.append(sp.shape)
        else:
            input_shape = self._env.observation_space.shape
            self._state_is_tuple = False    

        memory_type = self._hyperparameters['memory_type']
        memory_kwargs = self._hyperparameters['memory_kwargs']
        memory_size = memory_kwargs['memory_size']
        if memory_type == "PER":
            alpha = memory_kwargs['alpha'] if 'alpha' in memory_kwargs else 0.5
            beta = memory_kwargs['beta'] if 'beta' in memory_kwargs else 0.4
            self._memory = PrioritizedExperienceReplay(memory_size, input_shape, alpha=alpha, beta=beta)
        else:
            self._memory = ExperienceReplay(memory_size, input_shape)

        if self._n_steps > 1:
            self._memory_n_step = ExperienceReplay(memory_size, input_shape, self._n_steps, self._gamma)
        else:
            self._memory_n_step = None

        self._steps = 0
        self._total_steps = 0
        self._done_training = False
            
    def TrainNoYield(self, num_episodes=math.inf, num_steps=math.inf):
        for episode_info in self.Train(num_episodes, num_steps):
            # episode = episode_info['episode']
            # steps = episode_info['steps']
            # score = episode_info['episode_score']
            # mean_score = episode_info['mean_score']
            # total_steps = episode_info['total_steps']
            # print(("Episode: %d, steps: %d, total steps: %d, episode score: %.2f, mean score: %.2f ") % (self._episode, steps, self._total_steps, episode_score, mean_score))

            msg = ""
            for k in episode_info:
                msg += k + ": " + str(episode_info[k]) + ", "

            print(msg)

    def Train(self, num_episodes=math.inf, num_steps=math.inf):
        """
        [[Generator function]]
        Trains the agent n epsiodes or k steps, which ever comes first.
        num_episodes {int} -- Number of episodes to train the agent. Otherwise, will train until num_steps or hits the reward goal.
        num_steps {int} -- Number of total steps to train the agent. Otherwise, will train until num_episodes or hits the reward goal.

        Yields {dict}: Episode {int}, steps {int}, episode_score {float}, mean_score {float}, best_score {float}, best_mean_score {float}, total_steps {int}, gym env episode info.
        """
        self._episode = 0

        mean_score = 0
        while self._episode < num_episodes and self._total_steps < num_steps and not self._done_training:
            episode_score, steps, episode_loss, info = self.PlayEpisode(evaluate=False)

            self._episode_scores.append(episode_score)
            self._episode_scores = self._episode_scores[-self._reward_window:]
            self._episode_mean_score = np.mean(self._episode_scores)

            if episode_score > self._best_score:
                self._best_score = episode_score
            if self._episode_mean_score > self._best_mean_score:
                self._best_mean_score = self._episode_mean_score
            if self._episode_mean_score >= self._reward_goal:
                self._done_training = True
            
            self._episode += 1

            episode_info = {}
            episode_info['agent_name'] = self._name
            episode_info['episode'] = self._episode
            episode_info['loss'] = episode_loss
            episode_info['steps'] = steps
            episode_info['episode_score'] = round(episode_score, 2)
            episode_info['mean_score'] = round(self._episode_mean_score, 2)
            episode_info['best_score'] = round(self._best_score, 2)
            episode_info['total_steps'] = self._total_steps
            
            episode_info.update(info)

            yield episode_info
        
        # finished training save self.
        self.Checkpoint()
        self._plotter.ShowAll()

    def Evaluate(self, episodes=1000):
        episode_scores = []
        episode_mean_score = 0
        best_score = -math.inf
        total_steps = 0

        for i in range(1, episodes+1):        
            if self._enable_seed:
                self._env.seed(self._seed)

            episode_score, steps, info = self.PlayEpisode(evaluate=True)

            if episode_score > best_score:
                best_score = episode_score
            episode_scores.append(episode_score)
            episode_mean_score = np.mean(episode_scores)

            total_steps += steps

            episode_info = {"eval": True, "agent_name": self._name, "episode": i, "steps": steps, "episode_score": round(episode_score, 2), "mean_score": round(episode_mean_score, 2), "best_score": round(best_score, 2), "total_steps": total_steps}
            episode_info.update(info)
            
            if self._episode % self._checkpoint_episode_frequency == 0 and self._best_score > self._prev_best_mean_score:
                self.Checkpoint()

            yield episode_info

    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size):
        raise NotImplementedError("Error must implement the Calculate Loss function")

    def PlayEpisode(self):
        raise NotImplementedError("Agent must implement the PlayEpisode function")

    def Act(self):
        raise NotImplementedError("Agent must implement the Act function")

    def Learn(self):
        raise NotImplementedError("Error must implement Learn function")

    def Load(self, filepath):
        raise NotImplementedError("Error must implement load function")

    def Save(self, folderpath, filename):
        # move into utility class
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)

        list_of_files = os.listdir(folderpath)
        if len(list_of_files) >= self._checkpoint_max_num:
            full_path = [folderpath + "/{0}".format(x) for x in list_of_files]

            oldest_file = min(full_path, key=os.path.getctime)
            os.remove(oldest_file)
    
    def Checkpoint(self):
        folderpath = self._checkpoint_root + "/" + self._name
        filename = "episode_" + str(self._episode) + "_score_" + str(round(self._episode_mean_score, 2)) + ".pt"
        self.Save(folderpath, filename)

    def OptimizationStep(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm)

        optimizer.step()

    def SampleMemoryT(self, batch_size):
        states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np, indices_np = self._memory.Sample(batch_size)
        states_t, actions_t, next_states_t, rewards_t, dones_t, weights_t = self.ConvertNPWeightedMemoryToTensor(states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np)
        return states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t

    def GenericConvertNPMemoryToTensor(self, *arrays_np):
        tensors = ()

        for array_np in arrays_np:
            x = (torch.tensor(array_np, device=self._device),)
            tensors = tensors + (torch.tensor(array_np, device=self._device),)
        return tensors

    # TODO converting objects into tensors... (this is gonna be a tough one)
    # Maybe the same way I am doing for acting???
    def ConvertNPMemoryToTensor(self, states_np, actions_np, next_states_np, rewards_np, dones_np):
        if self._state_is_tuple:
            states_t = []
            for state in states_np:
                state_t = torch.tensor(state, dtype=torch.float32, device=self._device)
                states_t.append(state_t)

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

    def ConvertNPWeightedMemoryToTensor(self, states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np):
        weights_t = torch.tensor(weights_np, dtype=torch.float32, device=self._device)
        states_t, actions_t, next_states_t, rewards_t, dones_t = self.ConvertNPMemoryToTensor(states_np, actions_np, next_states_np, rewards_np, dones_np)
        return states_t, actions_t, next_states_t, rewards_t, dones_t, weights_t

    def UpdateNetwork(self, from_net, to_net, tau=1.0):
        if tau == 1.0:
            to_net.load_state_dict(from_net.state_dict())
        else:
            for from_param, to_param in zip(from_net.parameters(), to_net.parameters()):
                to_param.data.copy_(tau*from_param.data + (1.0-tau) * to_param.data)
        
    def CreateOptimizer(self, optimizer, network_params, optimizer_args):
        if optimizer.lower() == "adam":
            return optim.Adam(network_params, **optimizer_args)
