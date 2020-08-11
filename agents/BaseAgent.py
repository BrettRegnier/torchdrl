import numpy as np
import torch

# TODO maybe remove?
from data_structures.UniformExperienceReplay import UniformExperienceReplay as UER

class BaseAgent(object):
    def __init__(self, config):
        # TODO logger
        # self._logger = self.SetupLogger()

        self._config = config
        self._hyperparameters = config['hyperparameters']

        #TODO setup seeding
        self._env = config['env']
        self._device = config['device']
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
        self._episode = 1

        done_training = False
        mean_score = 0
        while self._episode != num_episodes and not done_training:
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
            win = ""
            if 'win' in info:
                win = "win: " + info['win']

            # TODO visualization
            print(("Episode: %d, steps: %d, episode reward: %.2f, mean reward: %.2f " + win) % (self._episode, steps, episode_reward, mean_score))

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


        
# TODO somehow many it generic enough to have a network buildable based on inputs