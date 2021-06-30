import os 
import math
import numpy as np
from torch.serialization import save

import torchdrl.tools.Helper as Helper

from termcolor import colored

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
            record_chart_metrics=False,
            visualize=False,
            visualize_frequency=-1,
            milestone_checkpoints:list=[],
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
        self._record_chart_metrics = record_chart_metrics

        self._visualize = visualize
        self._visualize_frequency = visualize_frequency

        self._milestone_checkpoints = milestone_checkpoints

        self._device = device

        # TODO maybe everything relating to the state
        # should be stored in this dictionary obj?
        # could be impossible...
        # Thought specific keys could be watched for in the ep info
        self._metrics = metrics
        self._chart_metrics = {}

        self._episode = 0
        self._total_steps = 0

        # Episode metrics
        # score_history = [] # REMOVE Deprecated
        # avg_score = 0 # REMOVE deprecated
        # steps_history = [] # REMOVE deprecated
        self._total_avg_loss = 0 
        self._best_avg_score = 0
        self._best_score = 0
        
        # Training variables
        self._train_last_checkpoint = 0

        # Evaluation variables
        self._total_avg_test_score = -math.inf
        self._test_score_history = []
        self._evaluate_checkpoint_count = 0

    def Train(self, num_episodes=math.inf, num_steps=math.inf):
        """
        [[Generator function]]
        Trains the agent n epsiodes or k steps_history, which ever comes first.
        num_episodes {int} -- Number of episodes to train the agent. Otherwise, will train until num_steps or hits the reward goal.
        num_steps {int} -- Number of total steps_history to train the agent. Otherwise, will train until num_episodes or hits the reward goal.

        Yields {dict}: Episode {int}, steps_history {int}, episode_score {float}, avg_score {float}, best_score {float}, best_avg_score {float}, total_steps {int}, gym env episode info.
        """

        # Note this is in here for my project
        # but this could get moved to an inherited manager
        # maybe...?
        wins = 0 
        loses = 0 

        eval_count = 0

        score_history = []
        avg_score = 0

        steps_history = []
        avg_steps = 0

        loss_history = []
        avg_loss = 0

        window_score = 0
        window_steps = 0
        window_loss = 0

        train_info = {}
        train_msg = ""
        test_info = {}
        test_msg = ""

        self._is_training = True

        self._chart_metrics = {
            "train": {
                "episode": [],
                "score": [],
                "loss": [],
                "steps": [],
            },
            "test": {
                "episode": [],
                "score": [],
                "steps": [],
                "accuracy": [], # TODO move to dragon boat manager lol
            }
        }

        for (episode_score, episode_steps, episode_loss, ep_info) in self._agent.PlayEpisode(evaluate=False):
            self._episode += 1
            self._total_steps += episode_steps

            window_score += episode_score
            window_steps += episode_steps
            window_loss += episode_loss

            score_history.append(episode_score)
            score_history = score_history[-self._reward_window:]
            avg_score = np.average(score_history)

            steps_history.append(episode_steps)
            steps_history = steps_history[-self._reward_window:]
            avg_steps = round(np.average(steps_history), 0)

            loss_history.append(episode_loss)
            loss_history = loss_history[-self._reward_window:]
            avg_loss = np.average(loss_history)

            self._total_avg_loss -= self._total_avg_loss / self._episode
            self._total_avg_loss += episode_loss / self._episode 

            # TODO deprecate?
            if episode_score > self._best_score:
                self._best_score = episode_score
            if avg_score > self._best_avg_score:
                self._best_avg_score = avg_score

            # TODO move into a possible metric??
            if 'win' in ep_info:
                wins += 1 if ep_info['win'] else 0
                loses += 0 if ep_info['win'] else 1
            # TODO add ep_info update hook

            if self._episode % self._step_window == 0 or not self._is_training:
                window_score /= self._step_window
                window_steps /= self._step_window
                window_loss /= self._step_window                

                train_info = {}
                train_msg = ""
                test_info = {}
                test_msg = ""

                train_info['agent_name'] = self._agent._name
                train_info['episode'] = self._episode
                train_info['episodes'] = num_episodes
                train_info['total_steps'] = self._total_steps

                train_info['episode_score'] = window_score
                train_info['episode_steps'] = window_steps
                train_info['episode_loss'] = window_loss

                train_info['avg_score'] = avg_score
                train_info['avg_steps'] = avg_steps
                train_info['avg_loss'] = avg_loss

                train_info['total_avg_loss'] = self._total_avg_loss
                
                train_info['best_score'] = self._best_score
                
                train_info.update(ep_info)

                if self._record_chart_metrics:
                    self._chart_metrics["train"]['episode'].append(self._episode)
                    self._chart_metrics["train"]['score'].append(avg_score)
                    self._chart_metrics["train"]['loss'].append(avg_loss)
                    self._chart_metrics["train"]['steps'].append(avg_steps)

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
                
                window_score = 0
                window_steps = 0
                window_loss = 0

                yield train_info, train_msg, test_info, test_msg

            if self._train_checkpoint:
                self._train_last_checkpoint += 1
                if self._train_last_checkpoint == self._checkpoint_frequency:
                    folderpath = f"{self._checkpoint_root}/{self._agent._name}/checkpoint"
                    filename = f"episode_{self._episode}_score_{round(avg_score, 2)}.pt"

                    self.Checkpoint(folderpath, filename)
                    self._train_last_checkpoint = 0

            if self._milestone_checkpoints:
                if self._episode in self._milestone_checkpoints:
                    folderpath = f"{self._checkpoint_root}/{self._agent._name}/milestone"
                    filename = f"episode_{self._episode}_score_{round(avg_score, 2)}.pt"

                    self.Checkpoint(folderpath, filename)                 

            # Done conditions
            if avg_score >= self._reward_goal and self._episode > 10:
                self._agent.Stop()
                self._is_training = False
            if self._episode >= num_episodes:
                self._agent.Stop()
                self._is_training = False
            if self._total_steps >= num_steps:
                self._agent.Stop()
                self._is_training = False


        # finished training save self.
        folderpath = f"{self._checkpoint_root}/{self._agent._name}/final"
        filename = f"episode_{self._episode}_score_{round(avg_score, 2)}.pt"

        self.Save(folderpath, filename)

    def TrainNoYield(self, num_episodes=math.inf, num_steps=math.inf):
        for _, train_msg, _, test_msg in self.Train(num_episodes, num_steps):
            print(train_msg)
            if test_msg != "":
                print(test_msg)

    def TrainMessage(self, train_info):
        msg = ""
        episode = train_info['episode']
        episodes = train_info['episodes']
        total_steps = train_info['total_steps']

        episode_score = train_info['episode_score']
        episode_steps = int(train_info['episode_steps'])
        episode_loss = train_info['episode_loss']

        avg_score = train_info['avg_score']
        avg_steps = int(train_info['avg_steps'])
        avg_loss = train_info['avg_loss']

        total_avg_loss = round(train_info['total_avg_loss'], 4)

        out_of = episode
        if episodes != math.inf:
            out_of = str(episode) + "/" + str(episodes)

        #TODO use colorama
        msg = f"[Episode {out_of:>8}] T Steps: {total_steps:>7}, Ep Steps: {episode_steps:>3}, A Steps: {avg_steps:>3} | Ep Score: {episode_score:.2f}, A Score: {avg_score:.2f} | Ep Loss: {episode_loss:.4f}, A Loss {avg_loss:.4f}, T A Loss: {total_avg_loss:.4f}"
        return msg

    def Evaluate(self, episodes=100):
        test_info = {}
        total_steps = 0
        total_rewards = 0

        # TODO move into a the dragonboat manager
        wins = 0
        loses = 0

        for steps_history, episode_reward, info in self._agent.EvaluateEpisode(episodes):
            total_steps += steps_history
            total_rewards += episode_reward

            # TODO Move into dragonboat manager
            if 'win' in info:
                win = info['win']
                wins += 1 if win == 1 else 0
                loses += 1 if win == 0 else 0

        avg_score = total_rewards / episodes
        avg_steps = total_steps / episodes

        test_info['eval'] = True
        test_info['agent_name'] = self._agent._name
        test_info['episodes'] = episodes
        test_info['avg_steps'] = avg_steps
        test_info['avg_reward'] = avg_score

        self._test_score_history.append(total_rewards/episodes)

        # TODO move into dragonboat manager
        if 'win' in info:
            test_info['wins'] = wins
            test_info['loses'] = loses

            acc = round(wins / (wins+loses)*100)
            test_info['accuracy'] = acc
            self._test_score_history[len(self._test_score_history) - 1] = acc

            # TODO move into dragon boat manager lol
            if self._record_chart_metrics:
                self._chart_metrics['test']['accuracy'] = acc


        self._test_score_history = self._test_score_history[-self._reward_window:]
        total_avg_test_score = np.average(self._test_score_history)

        if self._record_chart_metrics and self._is_training:
            self._chart_metrics['test']['episode'] = self._episode
            self._chart_metrics['test']['score'] = avg_score
            self._chart_metrics['test']['steps'] = avg_steps

        if self._evaluate_checkpoint:
            self._evaluate_checkpoint_count = min(self._evaluate_checkpoint_count + 1, self._checkpoint_frequency)
            if self._evaluate_checkpoint_count == self._checkpoint_frequency and total_avg_test_score > self._total_avg_test_score:
                folderpath = f"{self._checkpoint_root}/{self._agent._name}/evaluation"
                filename = f"evaluation_score_{total_avg_test_score}.pt"

                self.Checkpoint(folderpath, filename)
                self._total_avg_test_score = total_avg_test_score
                self._evaluate_checkpoint_count = 0

        test_msg = self._TestMessage(test_info)
        return test_info, test_msg

    def _TestMessage(self, test_info):
        avg_steps = int(test_info['avg_steps'])

        msg = f"[--Test--] Avg Steps: {avg_steps:>3}"
        if 'accuracy' in test_info:
            msg += f", Acc: {test_info['accuracy']}%"
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
            'avg_loss': self._total_avg_loss,
            'total_avg_test_score': self._total_avg_test_score
        }

        if self._record_chart_metrics:
            save_info['chart_metrics'] = self._chart_metrics

        save_info.update(self._agent.GetSaveInfo())

        Helper.SaveAgent(folderpath, filename, save_info)

    def Load(self, filepath):
        state_dict = Helper.LoadAgent(filepath)

        self.LoadStateDict(state_dict)

        self._agent.LoadStateDict(state_dict)

    def LoadStateDict(self, state_dict):
        self._episode = state_dict['episode']
        self._total_steps = state_dict['total_steps']
        self._total_avg_loss = state_dict['avg_loss']
        self._total_avg_test_score = state_dict['total_avg_test_score']

