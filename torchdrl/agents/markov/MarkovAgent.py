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

from torchdrl.data_structures.ExperienceReplay import ExperienceReplay
from torchdrl.data_structures.UniformExperienceReplay import UniformExperienceReplay
from torchdrl.data_structures.PrioritizedExperienceReplay import PrioritizedExperienceReplay

class MarkovAgent(Agent):
    def __init__(self, 
            name, 
            env, 
            model, 
            optimizer,
            batch_size,
            memory,
            memory_n_step=None,
            scheduler=None,
            clip_grad=-1,
            gamma=0.99,
            target_update_frequency=100,
            tau=1,
            step_window=10,
            reward_window=100,
            reward_goal=None,
            max_steps_per_episode=-1,
            warm_up=-1,
            evaluate_episodes=100,
            evaluate_frequency=100,
            train_checkpoint=False,
            evaluate_checkpoint=False,
            checkpoint_root="./",
            checkpoint_frequency=-1,
            checkpoint_max_count=-1,
            visualize=False,
            visualize_frequency=-1,
            seed=-1,
            oracle=None,
            device="cpu"):

        self._name = name
        self._env = env
        self._model = model
        self._target_model = copy.deepcopy(self._model)
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._memory = memory
        self._memory_n_step = memory_n_step
        self._scheduler = scheduler # TODO implement

        # hyperparameters
        self._clip_grad = clip_grad
        self._gamma = gamma
        self._target_update_frequency = target_update_frequency
        self._target_update_steps = 0
        self._tau = tau

        self._step_window = step_window
        self._reward_goal = reward_goal
        self._reward_window = reward_window
        self._max_steps_per_episode = max_steps_per_episode
        self._warm_up = warm_up

        self._evaluate_episodes = evaluate_episodes
        self._evaluate_frequency = evaluate_frequency

        self._train_checkpoint = train_checkpoint if train_checkpoint == True else False
        self._evaluate_checkpoint = evaluate_checkpoint if evaluate_checkpoint == True else False
        self._checkpoint_root = checkpoint_root
        self._checkpoint_frequency = checkpoint_frequency
        self._checkpoint_max_num = checkpoint_max_count

        self._visualize = visualize
        self._visualize_frequency = visualize_frequency

        self._oracle = oracle
        self._device = device 

        # set the seed
        self._seed = seed
        if self._seed >= 0:
            np.random.seed(self._seed)
            random.seed(self._seed)
            torch.manual_seed(self._seed)
            self._env.seed(self._seed)
            

        self._episode_score = 0
        self._episode_scores = []
        self._episode_mean_score = 0
        self._episode = 0
        self._best_score = float("-inf")
        self._prev_best_mean_score = float("-inf")
        self._best_mean_score = float("-inf")
        self._avg_loss = 0
        self._test_scores = []
        self._avg_test_score = 0

        self._train_last_checkpoint = 0
        self._evaluate_last_checkpoint = 0
        
        # assumed that a object will be input
        if isinstance(self._env.observation_space, (gym.spaces.Tuple, gym.spaces.Dict)):
            self._state_is_tuple = True
        else:
            self._state_is_tuple = False    

        self._steps_history = []
        self._steps = 0
        self._total_steps = 0
        self._done_training = False

        # after variable init
        self._model.train()
        self._target_model.eval()
        Helper.UpdateNetwork(self._model, self._target_model)

        print(self._model)

    #                               #
    # Start super class functions   #
    #                               #

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
        avg_episode_loss = 0
        wins = 0 
        loses = 0
        eval_count = 0
        while self._episode < num_episodes and self._total_steps < num_steps and not self._done_training:
            self._episode += 1
            episode_score, steps, episode_loss, info = self.PlayEpisode(evaluate=False)

            self._episode_scores.append(episode_score)
            self._episode_scores = self._episode_scores[-self._reward_window:]
            self._episode_mean_score = np.mean(self._episode_scores)

            self._steps_history.append(steps)
            self._steps_history = self._steps_history[-self._reward_window:]
            mean_steps = round(np.mean(self._steps_history),0)

            self._avg_loss -= self._avg_loss / self._episode
            self._avg_loss += episode_loss / self._episode 

            avg_episode_loss += episode_loss

            if episode_score > self._best_score:
                self._best_score = episode_score
            if self._episode_mean_score > self._best_mean_score:
                self._best_mean_score = self._episode_mean_score
            if self._episode_mean_score >= self._reward_goal and self._episode > 10:
                self._done_training = True

            if 'win' in info:
                wins += 1 if info['win'] else 0
                loses += 0 if info['win'] else 1

            if self._episode % self._step_window == 0 or self._done_training: 
                avg_episode_loss /= self._step_window

                train_info = {}
                train_msg = ""
                test_info = {}
                test_msg = ""

                train_info['agent_name'] = self._name
                train_info['episode'] = self._episode
                train_info['episodes'] = num_episodes
                train_info['loss'] = avg_episode_loss
                train_info['avg_loss'] = self._avg_loss
                train_info['steps'] = steps
                train_info['total_steps'] = self._total_steps
                train_info['mean_steps'] = mean_steps
                train_info['episode_score'] = round(episode_score, 2)
                train_info['mean_score'] = round(self._episode_mean_score, 2)
                train_info['best_score'] = round(self._best_score, 2)
                train_info.update(info)

                train_msg = self.TrainMessage(train_info)

                if 'win' in info:
                    train_info['wins'] = wins
                    train_info['loses'] = loses
                    train_msg += f" | W: {wins} L: {loses}"
                    wins = 0
                    loses = 0

                if eval_count > self._evaluate_frequency:
                    test_info, test_msg = self.Evaluate(self._evaluate_episodes)
                    eval_count = 0
                
                avg_episode_loss = 0

                yield train_info, train_msg, test_info, test_msg

            if self._train_checkpoint:
                self._train_last_checkpoint += 1
                if self._train_last_checkpoint == self._checkpoint_frequency:
                    folderpath = f"{self._checkpoint_root}/{self._name}/checkpoint"
                    filename = f"episode_{self._episode}_score_{round(self._episode_mean_score, 2)}.pt"

                    self.Checkpoint(folderpath, filename)
                    self._train_last_checkpoint = 0

            eval_count += 1

        # finished training save self.
        folderpath = f"{self._checkpoint_root}/{self._name}/final"
        filename = f"episode_{self._episode}_score_{round(self._episode_mean_score, 2)}.pt"

        self.Save(folderpath, filename)
            
    def TrainNoYield(self, num_episodes=math.inf, num_steps=math.inf):
        for _, train_msg, _, test_msg in self.Train(num_episodes, num_steps):
            print(train_msg, test_msg)

    def TrainMessage(self, train_info):
        msg = ""
        episode = train_info['episode']
        episodes = train_info['episodes']
        total_steps = train_info['total_steps']
        mean_steps = train_info['mean_steps']
        loss = train_info['loss']
        avg_loss = train_info['avg_loss']
        avg_score = train_info['mean_score']

        out_of = episode
        if episodes != math.inf:
            out_of = str(episode) + "/" + str(episodes)

        msg = f"[Episode {out_of:>8}] Total Steps: {total_steps:>7}, Mean Steps: {mean_steps:>3} -> Loss {loss:.4f} Avg Loss: {avg_loss:.4f} Avg Score {avg_score:.4f}"
        return msg

    def Evaluate(self, episodes=100):
        test_info = {}

        total_steps = 0
        total_rewards = 0

        wins = 0
        loses = 0
        info = {}
        for i in range(1, episodes+1):

            episode_score, steps, episode_loss, info = self.PlayEpisode(evaluate=True)

            total_rewards += episode_score
            total_steps += steps

            if 'win' in info:
                win = info['win']
                wins += 1 if win == 1 else 0
                loses += 1 if win == 0 else 0
            
        test_info['eval'] = True
        test_info['agent_name'] = self._name
        test_info['episodes'] = episodes
        test_info['avg_steps'] = round(total_steps/episodes, 2)
        test_info['avg_reward'] = round(total_rewards/episodes, 2)

        self._test_scores.append(total_rewards/episodes)
        # TODO change this to episodes
        self._test_scores = self._test_scores[-self._reward_window:]

        if 'win' in info:
            test_info['wins'] = wins
            test_info['loses'] = loses

            acc = round(wins / (wins+loses)*100)
            test_info['accuracy'] = acc

            # # TODO fix this to grab the newest entry...
            self._test_scores[len(self._test_scores) - 1] = acc
        
        test_msg = self._TestMessage(test_info)

        avg_test_score = np.sum(self._test_scores)

        if self._evaluate_checkpoint:
            self._evaluate_last_checkpoint += 1
            if self._evaluate_last_checkpoint == self._checkpoint_frequency and avg_test_score > self._avg_test_score:
                folderpath = f"{self._checkpoint_root}/{self._name}/evaluation"
                filename = f"evaluation_score_{avg_test_score}.pt"

                self.Checkpoint(folderpath, filename)
                self._avg_test_score = avg_test_score
                self._evaluate_last_checkpoint = 0

        return test_info, test_msg

    def _TestMessage(self, test_info):
        avg_steps = test_info['avg_steps']

        msg = f"Test: Average Steps: {avg_steps}"
        if 'accuracy' in test_info:
            msg += f", Accuracy: {test_info['accuracy']}%"
        if 'wins' in test_info:
            msg += f" | W:{test_info['wins']}"
        if 'loses' in test_info:
            msg += f" L:{test_info['loses']}"
        
        return msg

    def GetAction(self, state, evaluate=False):
        if self._oracle and not evaluate:
            return self._oracle.Act(state)
        else:
            return self.Act(state, evaluate)

    #                               #
    #   End super class functions   #
    #                               #

    
    #                               #
    #   Begin abstract functions    #
    #                               #

    def Learn(self):
        raise NotImplementedError("Error. Markov agent must implement Learn function")

    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size):
        raise NotImplementedError("Error must implement the Calculate Loss function")
    
    #                           #
    #   End abstract functions  #
    #                           #

    def PlayEpisode(self, evaluate=False):
        self._steps = 0
        done = False
        episode_reward = 0
        episode_loss = 0

        state = self._env.reset()
        while self._steps != self._max_steps_per_episode and not done:
            action = self.GetAction(state, evaluate)

            next_state, reward, done, info = self._env.step(action)

            if not evaluate:
                transition = (state, action, next_state, reward, done)
                self.SaveMemory(transition)

                episode_loss += self.Learn()
                    
            if self._visualize:
                if self._episode % self._visualize_frequency == 0:
                    self._env.render()
                    
            episode_reward += reward
            state = next_state

            self._steps += 1
            self._total_steps += 1

        self._env.close()
        return episode_reward, self._steps, round(episode_loss,2), info

    def SaveMemory(self, transition):
        if self._memory_n_step:
            transition = self._memory_n_step.Append(*transition)

        if transition:
            self._memory.Append(*transition)
            
    def Update(self):
        states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t = self.SampleMemoryT(
            self._batch_size)

        # get errors
        errors = self.CalculateErrors(states_t, actions_t, next_states_t, rewards_t,
                                      dones_t, indices_np, weights_t, self._batch_size, self._gamma)

        weights_t = weights_t.reshape(-1, 1)
        # Prioritized Experience Replay weight importancing
        loss = torch.mean(errors * weights_t)

        # n-step learning with one-step to prevent high-variance
        if self._memory_n_step:
            gamma = self._gamma ** self._memory_n_step.GetNStep()
            states_np, actions_np, next_states_np, rewards_np, dones_np, _ = self._memory_n_step.SampleBatchFromIndices(
                indices_np)
            states_t, actions_t, next_states_t, rewards_t, dones_t = self.ConvertNPMemoryToTensor(
                states_np, actions_np, next_states_np, rewards_np, dones_np)
            errors_n = self.CalculateErrors(
                states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, self._batch_size, gamma)
            errors += errors_n

            loss = torch.mean(errors * weights_t)

        self._optimizer.zero_grad()
        loss.backward()
        if self._clip_grad > -1:
            clip_grad_norm_(self._model.parameters(), self._clip_grad)
        self._optimizer.step()
        
        if self._target_update_steps % self._target_update_frequency == 0:
            self._target_model.load_state_dict(self._model.state_dict())
            self._target_update_steps = 0

        self._target_update_steps += 1
        
        # for PER
        updated_priorities = errors.detach().cpu().numpy()
        self._memory.BatchUpdate(indices_np, updated_priorities)
        
        return loss

    def Checkpoint(self, folderpath, filename):
        if os.path.exists(folderpath):
            list_of_files = os.listdir(folderpath)
            if len(list_of_files) >= self._checkpoint_max_num:
                full_path = [folderpath + "/{0}".format(x) for x in list_of_files]

                oldest_file = min(full_path, key=os.path.getctime)
                os.remove(oldest_file)
        
        self.Save(folderpath, filename)

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

    def LoadStateDict(self, state_dict):
        self._model.load_state_dict(state_dict)
        self._target_model.load_state_dict(state_dict)
