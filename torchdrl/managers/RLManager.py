import os 
import math
import numpy as np

import torchdrl.tools.Helper as Helper

# TODO replace all self._agent._var references with getters.

class RLManager:
    def __init__(
            self, 
            agent, 
            metrics:list=[], 
            step_window=10,
            reward_window=100,
            reward_goal=None,
            evaluate_episodes=100,
            evaluate_frequency=100,
            train_checkpoint=False,
            evaluate_checkpoint=False,
            checkpoint_root="./",
            checkpoint_frequency=-1,
            checkpoint_max_count=-1,
            visualize=False,
            visualize_frequency=-1,
            device="cpu"):
        # Parameters
        # TODO check instance is that of a AgentBaseClass
        self._agent = agent
        self._step_window = step_window

        self._reward_window = reward_window
        self._reward_goal = reward_goal

        self._evaluate_episodes = evaluate_episodes
        self._evaluate_frequency = evaluate_frequency
        
        self._train_checkpoint = train_checkpoint if train_checkpoint == True else False
        self._evaluate_checkpoint = evaluate_checkpoint if evaluate_checkpoint == True else False

        self._checkpoint_root = checkpoint_root
        self._checkpoint_frequency = checkpoint_frequency
        self._checkpoint_max_num = checkpoint_max_count

        self._visualize = visualize
        self._visualize_frequency = visualize_frequency

        self._device = device

        # TODO maybe everything relating to the state
        # should be stored in this dictionary obj?
        # could be impossible...
        # Thought specific keys could be watched for in the ep info
        self._metrics = metrics

        self._episode = 0
        self._total_steps = 0

        # Episode metrics
        self._episode_scores = []
        self._epsiode_avg_score = 0       
        self._steps_history = []
        self._avg_loss = 0
        self._best_avg_score = 0
        self._best_score = 0
        
        # Training variables
        self._train_last_checkpoint = 0

        # Evaluation variables
        self._avg_test_score = 0
        self._test_scores = []
        self._evaluate_checkpoint_count = 0

    def Train(self, num_episodes=math.inf, num_steps=math.inf):
        """
        [[Generator function]]
        Trains the agent n epsiodes or k steps, which ever comes first.
        num_episodes {int} -- Number of episodes to train the agent. Otherwise, will train until num_steps or hits the reward goal.
        num_steps {int} -- Number of total steps to train the agent. Otherwise, will train until num_episodes or hits the reward goal.

        Yields {dict}: Episode {int}, steps {int}, episode_score {float}, avg_score {float}, best_score {float}, best_avg_score {float}, total_steps {int}, gym env episode info.
        """

        # Note this is in here for my project
        # but this could get moved to an inherited manager
        # maybe...?
        wins = 0 
        loses = 0 

        eval_count = 0
        mean_steps = 0
        avg_episode_loss = 0

        train_info = {}
        train_msg = ""
        test_info = {}
        test_msg = ""

        stop_training = False

        for (episode_score, steps, episode_loss, ep_info) in self._agent.PlayEpisode(evaluate=False):
            self._episode += 1
            self._total_steps += steps
            self._episode_scores.append(episode_score)
            self._episode_scores = self._episode_scores[-self._reward_window:]
            self._epsiode_avg_score = np.sum(self._episode_scores) / self._reward_window

            self._steps_history.append(steps)
            self._steps_history = self._steps_history[-self._reward_window:]
            mean_steps = round(np.mean(self._steps_history), 0)

            self._avg_loss -= self._avg_loss / self._episode
            self._avg_loss += episode_loss / self._episode 

            avg_episode_loss += episode_loss

            if episode_score > self._best_score:
                self._best_score = episode_score
            if self._epsiode_avg_score > self._best_avg_score:
                self._best_avg_score = self._epsiode_avg_score

            # TODO move into a possible metric??
            if 'win' in ep_info:
                wins += 1 if ep_info['win'] else 0
                loses += 0 if ep_info['win'] else 1
            # TODO add ep_info update hook

            if self._episode % self._step_window == 0 or stop_training:
                avg_episode_loss /= self._step_window

                train_info = {}
                train_msg = ""
                test_info = {}
                test_msg = ""

                train_info['agent_name'] = self._agent._name
                train_info['episode'] = self._episode
                train_info['episodes'] = num_episodes
                train_info['loss'] = avg_episode_loss
                train_info['avg_loss'] = self._avg_loss
                train_info['steps'] = steps
                train_info['total_steps'] = self._total_steps
                train_info['mean_steps'] = mean_steps
                train_info['episode_score'] = round(episode_score, 2)
                train_info['avg_score'] = round(self._epsiode_avg_score, 2)
                train_info['best_score'] = round(self._best_score, 2)
                train_info.update(ep_info)

                train_msg = self.TrainMessage(train_info)

                # TODO add a train info hook for subclasses
                if 'win' in ep_info:
                    train_info['wins'] = wins
                    train_info['loses'] = loses
                    train_msg += f" | W: {wins} L: {loses}"
                    wins = 0
                    loses = 0

                eval_count += 1
                if self._evaluate_frequency > 0 and eval_count % self._evaluate_frequency == 0:
                    test_info, test_msg = self.Evaluate(self._evaluate_episodes)
                    eval_count = 0
                
                avg_episode_loss = 0

                yield train_info, train_msg, test_info, test_msg

            if self._train_checkpoint:
                self._train_last_checkpoint += 1
                if self._train_last_checkpoint == self._checkpoint_frequency:
                    folderpath = f"{self._checkpoint_root}/{self._agent._name}/state_dict"
                    filename = f"episode_{self._episode}_score_{round(self._epsiode_avg_score, 2)}.pt"

                    self.Checkpoint(folderpath, filename)
                    self._train_last_checkpoint = 0

            # Done conditions
            if self._epsiode_avg_score >= self._reward_goal and self._episode > 10:
                self._agent.Stop()
                stop_training = True
            if self._episode >= num_episodes:
                self._agent.Stop()
                stop_training = True
            if self._total_steps >= num_steps:
                self._agent.Stop()
                stop_training = True


        # finished training save self.
        folderpath = f"{self._checkpoint_root}/{self._agent._name}/final"
        filename = f"episode_{self._episode}_score_{round(self._epsiode_avg_score, 2)}.pt"

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
        avg_score = train_info['avg_score']

        out_of = episode
        if episodes != math.inf:
            out_of = str(episode) + "/" + str(episodes)

        msg = f"[Episode {out_of:>8}] Total Steps: {total_steps:>7}, Mean Steps: {mean_steps:>3} -> Loss {loss:.4f} Avg Loss: {avg_loss:.4f} Avg Score {avg_score:.4f}"
        return msg

    def Evaluate(self, episodes=100):
        test_info = {}
        total_steps = 0
        total_rewards = 0

        # TODO move into a the dragonboat manager
        wins = 0
        loses = 0

        for steps, episode_reward, info in self._agent.EvaluateEpisode(episodes):
            total_steps += steps
            total_rewards += episode_reward

            # TODO Move into dragonboat manager
            if 'win' in info:
                win = info['win']
                wins += 1 if win == 1 else 0
                loses += 1 if win == 0 else 0

        test_info['eval'] = True
        test_info['agent_name'] = self._agent._name
        test_info['episodes'] = episodes
        test_info['avg_steps'] = round(total_steps/episodes, 2)
        test_info['avg_reward'] = round(total_rewards/episodes, 2)
        test_msg = self._TestMessage(test_info)

        self._test_scores.append(total_rewards/episodes)

        # TODO move into dragonboat manager
        # if 'win' in infos:
        #     test_info['wins'] = wins
        #     test_info['loses'] = loses

        #     acc = round(wins / (wins+loses)*100)
        #     test_info['accuracy'] = acc
        #     self._test_scores[len(self._test_scores) - 1] = acc

        self._test_scores = self._test_scores[-self._reward_window:]
        avg_test_score = np.sum(self._test_scores) / episodes

        if self._evaluate_checkpoint:
            self._evaluate_checkpoint_count = min(self._evaluate_checkpoint_count + 1, self._checkpoint_frequency)
            if self._evaluate_checkpoint_count == self._checkpoint_frequency and avg_test_score > self._avg_test_score:
                folderpath = f"{self._checkpoint_root}/{self._agent._name}/evaluation"
                filename = f"evaluation_score_{avg_test_score}.pt"

                self.Checkpoint(folderpath, filename)
                self._avg_test_score = avg_test_score
                self._evaluate_checkpoint_count = 0

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

    def Checkpoint(self, folderpath, filename):
        if os.path.exists(folderpath):
            list_of_files = os.listdir(folderpath)
            if len(list_of_files) >= self._checkpoint_max_num:
                full_path = [folderpath + "/{0}".format(x) for x in list_of_files]

                oldest_file = min(full_path, key=os.path.getctime)
                os.remove(oldest_file)
        
        self.Save(folderpath, filename)

    def Save(self, folderpath, filename):
        save_info = {
            'episode': self._episode,
            'total_steps': self._total_steps,
            'avg_loss': self._avg_loss,
            'avg_test_score': self._avg_test_score
        }

        save_info.update(self._agent.GetSaveInfo())

        Helper.SaveAgent(folderpath, filename, save_info)

    def Load(self, filepath):
        state_dict = Helper.LoadAgent(filepath)

        self.LoadStateDict(state_dict)

        self._agent.LoadStateDict(state_dict)

    def LoadStateDict(self, state_dict):
        self._episode = state_dict['episode']
        self._total_steps = state_dict['total_steps']
        self._avg_loss = state_dict['avg_loss']
        self._avg_test_score = state_dict['avg_test_score']

