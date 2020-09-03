import torch
import numpy as np

from data_structures.UniformExperienceReplay import UniformExperienceReplay as UER
from representations.Plotter import Plotter

class BaseAgent(object):
    def __init__(self, config):
        # TODO logger
        # self._logger = self.SetupLogger()

        self._config = config
        self._hyperparameters = config['hyperparameters']

        self._env = config['env']
        self._device = config['device']
        self._seed = config['seed']
        self._enable_seed = config['enable_seed']

        self._log = None
        if config['log']:
            self._log = Plotter()
            self._log.AddFigure("Score", "Episode Score", "green")
            self._log.AddFigure("Score", "Mean Score", "blue")
        
        self._show_log = config['show_log']
        self._show_log_frequency = config['show_log_frequency']

        # TODO setup environment title
        self._action_type = "DISCRETE" if self._env.action_space.dtype == np.int64 else "CONTINUOUS"
        self._n_actions = self._env.action_space.n if self._action_type == "DISCRETE" else env.action_space.shape[0]
        self._input_shape = self._env.observation_space.shape

        # TODO generalize how to choose memory.
        self._memory = UER(config['memory_size'])
        self._batch_size = config['batch_size']
        self._warm_up = config['warm_up']
        self._max_steps = config['max_steps']
        
        self._reward_goal = config["reward_goal"]
        self._reward_window = config["reward_window"] # how much to average the score over
        self._episode_score = 0 # TODO deprecate?
        self._episode_scores = []
        self._mean_episode_score = []
        self._episode = 0
        self._best_episode_score = float("-inf")
        self._best_mean_episode_score = float("-inf")
        self._visualize = config["visualize"]
        self._visualize_frequency = config["visualize_frequency"] # how many episodes
        self._total_steps = 0
    
    def Train(self, num_episodes=-1):
        self._episode = 0

        done_training = False
        mean_score = 0
        while self._episode != num_episodes and not done_training:
            if self._enable_seed:
                self._env.seed(self._seed)

            episode_reward, steps, info = self.PlayEpisode(evaluate=False)

            self._episode_scores.append(episode_reward)
            mean_score = np.mean(self._episode_scores[-self._reward_window:])

            if episode_reward > self._best_episode_score:
                self._best_episode_score = episode_reward
            if mean_score > self._best_mean_episode_score:
                self._best_mean_episode_score = mean_score
                if mean_score > self._reward_goal:
                    done_training = True
            
            self._episode += 1

            msg = ""
            for k in info:
                msg += k + ": " + str(info[k]) + ", "
            # if 'win' in info:
            #     win = "win: " + str(info['win'])

            # TODO visualization
            print(("Episode: %d, steps: %d, episode reward: %.2f, mean reward: %.2f " + msg) % (self._episode, steps, episode_reward, mean_score))

            # show/add to log
            if self._log != None:
                self._log.AddPoint("Score", "Episode Score", (self._episode, episode_reward))
                self._log.AddPoint("Score", "Mean Score", (self._episode, mean_score))
                if self._show_log and self._episode % self._show_log_frequency == 0:
                    self._log.ShowAll()


        # finished training
        # TODO add saving of plot and model and info

    def Evaluate(self):
        raise NotImplementedError("Error must implenet the Evaluate function")

    def PlayEpisode(self):
        raise NotImplementedError("Agent must implement the Step function")

    def Act(self):
        raise NotImplementedError("Agent must implement the Act function")

    def Learn(self):
        raise NotImplementedError("Error must implement Learn function")

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

    def CopyNetwork(self, net, target_net):
        target_net.load_state_dict(net.state_dict())
        
# TODO somehow many it generic enough to have a network buildable based on inputs