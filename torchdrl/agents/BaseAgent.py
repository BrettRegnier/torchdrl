import torch
import numpy as np
import math

from ..data_structures import Shared

from ..data_structures.UniformExperienceReplay import UniformExperienceReplay as UER
from ..representations.Plotter import Plotter

class BaseAgent(object):
    def __init__(self, config):
        self._config = config
        self._hyperparameters = config['hyperparameters']

        self._env = config['env']
        self._device = config['device']
        self._seed = config['seed']
        self._enable_seed = config['enable_seed']

        self._action_type = "DISCRETE" if self._env.action_space.dtype == np.int64 else "CONTINUOUS"
        self._n_actions = self._env.action_space.n if self._action_type == "DISCRETE" else env.action_space.shape[0]
        self._input_shape = self._env.observation_space.shape

        # TODO generalize how to choose memory.
        self._memory = UER(config['memory_size'])
        self._batch_size = config['batch_size']
        self._warm_up = config['warm_up']
        self._max_steps = config['num_steps']
        
        self._reward_goal = config["reward_goal"]
        self._reward_window = config["reward_window"] # how much to average the score over
        self._episode_score = 0
        self._episode_scores = []
        self._mean_episode_score = []
        self._episode = 0
        self._best_score = float("-inf")
        self._best_mean_score = float("-inf")
        self._visualize = config["visualize"]
        self._visualize_frequency = config['visualize_frequency'] # how many episodes
        self._total_steps = 0
            
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

        Yields {dict}: Epsisode {int}, steps {int}, episode_score {float}, mean_score {float}, best_score {float}, best_mean_score {float}, total_steps {int}, gym env episode info.
        """
        self._episode = 0

        done_training = False
        mean_score = 0
        while self._episode < num_episodes and self._total_steps < num_steps and not done_training and not Shared.stop_training:
            if self._enable_seed:
                self._env.seed(self._seed)

            episode_score, steps, info = self.PlayEpisode(evaluate=False)

            self._episode_scores.append(episode_score)
            self._episode_scores = self._episode_scores[-self._reward_window:]
            mean_score = np.mean(self._episode_scores)

            if episode_score > self._best_score:
                self._best_score = episode_score
            if mean_score > self._best_mean_score:
                self._best_mean_score = mean_score
                if mean_score > self._reward_goal:
                    done_training = True
            
            self._episode += 1

            episode_info = {"episode": self._episode, "steps": steps, "episode_score": episode_score, "mean_score": mean_score, "best_score": self._best_score, "best_mean_score": self._best_mean_score, "total_steps": self._total_steps}
            episode_info.update(info)
            yield episode_info


    def Evaluate(self):
        raise NotImplementedError("Error must implenet the Evaluate function")

    def PlayEpisode(self):
        raise NotImplementedError("Agent must implement the Step function")

    def Act(self):
        raise NotImplementedError("Agent must implement the Act function")

    def Learn(self):
        raise NotImplementedError("Error must implement Learn function")

    def Save(self, filepath):
        raise NotImplementedError("Error must implement save function")

    def Load(self, filepath):
        raise NotImplementedError("Error must implement load function")

    def OptimizationStep(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm)

        optimizer.step()

    def SampleMemoryT(self, batch_size):
        states_np, actions_np, next_states_np, rewards_np, dones_np = self._memory.Sample(batch_size)
        
        states_t = torch.tensor(states_np, dtype=torch.float32).to(self._device)
        actions_t = torch.tensor(actions_np, dtype=torch.int64).to(self._device)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32).to(self._device)
        rewards_t = torch.tensor(rewards_np, dtype=torch.float32).to(self._device)
        dones_t = torch.tensor(dones_np, dtype=torch.int64).to(self._device)

        return states_t, actions_t, next_states_t, rewards_t, dones_t

    def CopyNetwork(self, net, target_net, tau=1.0):
        for target_param, local_param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)
