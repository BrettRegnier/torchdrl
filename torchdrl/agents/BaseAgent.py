import os
import math
import torch
import numpy as np

from ..data_structures import Shared

from ..data_structures.ExperienceReplay import ExperienceReplay
from ..data_structures.UniformExperienceReplay import UniformExperienceReplay
from ..data_structures.PrioritizedExperienceReplay import PrioritizedExperienceReplay
from ..data_structures.NStepPrioritizedExperienceReplay import NStepPrioritizedExperienceReplay
from ..data_structures.PER import PER
from ..representations.Plotter import Plotter

class BaseAgent(object):
    def __init__(self, config):
        self._config = config

        self._name = config['name']
        # self._env = config['env']
        self._env = config['env'](**self._config['env_kwargs'])
        self._seed = config['seed']
        self._enable_seed = config['enable_seed']

        self._action_type = "DISCRETE" if self._env.action_space.dtype == np.int64 else "CONTINUOUS"
        self._n_actions = self._env.action_space.n if self._action_type == "DISCRETE" else env.action_space.shape[0]
        self._input_shape = self._env.observation_space.shape
        self._checkpoint_frequency = config['checkpoint_frequency']

        self._warm_up = config['warm_up']
        self._max_steps = config['max_steps']
        self._reward_goal = config["reward_goal"]
        self._reward_window = config["reward_window"] # how much to average the score over
        self._episode_score = 0
        self._episode_scores = [0 for i in range(100)]
        self._episode_mean_score = 0
        self._episode = 0
        self._best_score = float("-inf")
        self._prev_best_mean_score = float("-inf")
        self._best_mean_score = float("-inf")
        self._visualize = config["visualize"]
        self._visualize_frequency = config['visualize_frequency'] # how many episodes

        # begin defining all shared hyperparamters
        self._hyperparameters = config['hyperparameters']
        self._gamma = self._hyperparameters['gamma']
        self._n_steps = self._hyperparameters['n_steps']
        self._batch_size = self._hyperparameters['batch_size']


        # TODO streamline this from the config and remove the 
        # TODO memory declaration from the apex version
        # if this is an Ape-X agent, don't do this
        # because Ape-X will use a shared memory
        memory_size = self._hyperparameters['memory_size']
        if 'apex_parameters' not in self._config:
            self._device = config['device']
            memory_type = self._hyperparameters['memory_type']
            # memory
            if memory_type == "PER":
                alpha = self._hyperparameters['alpha']
                beta = self._hyperparameters['beta']
                priority_epsilon = self._hyperparameters['priority_epsilon']
                self._memory = PER(memory_size, self._input_shape, alpha, beta, priority_epsilon)
            else:
                self._memory = UniformExperienceReplay(memory_size, self._input_shape)

        if self._n_steps > 1:
            self._memory_n_step = ExperienceReplay(memory_size, self._input_shape, self._n_steps, self._gamma)

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
            if self._enable_seed:
                self._env.seed(self._seed)

            episode_score, steps, info = self.PlayEpisode(evaluate=False)

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

            episode_info = {"agent_name": self._name, "episode": self._episode, "steps": steps, "episode_score": round(episode_score, 2), "mean_score": round(self._episode_mean_score, 2), "best_score": round(self._best_score, 2), "total_steps": self._total_steps}
            episode_info.update(info)

            if self._episode % self._checkpoint_frequency == 0 and self._best_score > self._prev_best_mean_score:
                self.Checkpoint()

            yield episode_info
        
        # finished training save self.
        self.Checkpoint()

    def Evaluate(self, episodes=1000):
        episode = 0
        episode_scores = []
        episode_mean_score = 0
        best_score = -math.inf
        total_steps = 0

        for i in range(episodes):        
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

            yield episode_info

    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size):
        raise NotImplementedError("Error must implement the Calculate Loss function")

    def PlayEpisode(self):
        raise NotImplementedError("Agent must implement the Step function")

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
        if len(list_of_files) >= self._config['checkpoint_max_num']:
            full_path = [folderpath + "/{0}".format(x) for x in list_of_files]

            oldest_file = min(full_path, key=os.path.getctime)
            os.remove(oldest_file)
    
    def Checkpoint(self):
        folderpath = self._config['checkpoint_root'] + "/" + self._name
        filename = "episode_" + str(self._episode) + "_score_" + str(round(self._episode_mean_score, 2)) + ".pt"
        self.Save(folderpath, filename)

    def OptimizationStep(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm)

        optimizer.step()

    def SampleMemoryT(self, batch_size):
        states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np = self._memory.Sample(batch_size)
        states_t, actions_t, next_states_t, rewards_t, dones_t, weights_t = self.ConvertNPWeightedMemoryToTensor(states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np)
        return states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t
        

    def GenericConvertNPMemoryToTensor(self, *arrays_np):
        tensors = ()

        for array_np in arrays_np:
            x = (torch.tensor(array_np, device=self._device),)
            tensors = tensors + (torch.tensor(array_np, device=self._device),)
        return tensors

    def ConvertNPMemoryToTensor(self, states_np, actions_np, next_states_np, rewards_np, dones_np):
        states_t = torch.tensor(states_np, dtype=torch.float32, device=self._device)
        actions_t = torch.tensor(actions_np, dtype=torch.int64, device=self._device)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32, device=self._device)
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
        

            
