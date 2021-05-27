import math

from typing import Iterable


# TODO this will parallelize the the agents?
class RLManager:
    def __init__(self, envs:Iterable, agent, memory, n_step_memory):
        if isinstance(agent, (list, tuple)):
            self._agent = agent
        else:
            self._agent = [agent]

        if isinstance(self._envs, (list, tuple)):
            self._envs = self._envs
        else:
            self._envs = [self._envs]

    def Train(self, num_episodes=math.inf, num_steps=math.inf):
        """
        [[Generator function]]
        Trains the agent n epsiodes or k steps, which ever comes first.
        num_episodes {int} -- Number of episodes to train the agent. Otherwise, will train until num_steps or hits the reward goal.
        num_steps {int} -- Number of total steps to train the agent. Otherwise, will train until num_episodes or hits the reward goal.

        Yields {dict}: Episode {int}, steps {int}, episode_score {float}, mean_score {float}, best_score {float}, best_mean_score {float}, total_steps {int}, gym env episode info.
        """

        avg_episode_loss = 0

        # Note this is in here for my project
        # but this could get moved to an inherited manager
        wins = 0 
        loses = 0 
        eval_count = 0

        # TODO move into init
        self._episode = 0
        self._total_steps = 0
        self._done_training = False
        while self._episode < num_episodes and self._total_steps < num_steps and self._done_training:
            self._episode += 1
            episode_scores, steps, episode_losses, ep_infos = self.PlayEpisode(evaluate=False)


    def PlayEpisode(self, evaluate=False):
        # TODO move into init
        self._max_steps_per_episode = [10] * len(self._envs)
        self._total_steps = [0] * len(self._envs)

        # This will never play all agent without threading....
        # Which is an APEX job, so I should do a single agent
        # with many self._envs... and maybe the agent can keep the memory??
        states = [None] * len(self._envs)
        actions = [None] * len(self._envs)
        next_states = [None] * len(self._envs)
        rewards = [None] * len(self._envs)       
        dones = [True] * len(self._envs)
        info = [{} for _ in range(len(self._envs))]

        episode_rewards = [0] * len(self._envs)
        episode_loss = [0] * len(self._envs)

        steps = [0] * len(self._envs)            

        # when a game finishes, I should still collect info from all
        # other games, but reset that, however, how do I do that?
        # Maybe I should make this into a generator???
        # thus then each transition is instead yielded
        # and then the games can keep running??
        # Potentially change name to Play
        # or gather experience or something I dunno
        while True: 
            actions = self.GetActions(states)
            for idx, (env, action) in enumerate(zip(self._envs, actions)):

                next_state, reward, done, info = env.step(action)

                if not evaluate:
                    transition = (states[idx], action, next_state, reward, done)
                    self._agent.SaveMemory(transition)

                    episode_loss[idx] += self._agent.Learn()

                episode_rewards[idx] += reward

                steps[idx] += 1
                self._total_steps[idx] += 1

                if steps[idx] == self._max_steps_per_episode[idx] or done:
                    states[idx] = env.reset()
                    info[idx].update(info)
                    yield (episode_rewards[idx], steps[idx], round(episode_loss[idx], info[idx]))
                else:
                    states[idx] = next_state

    
    def GetActions(self, states, evaluate=False):
        # TODO move into init
        self._oracle = None
        if self._oracle and not evaluate:
            return self._oracle.Act(states)
        else:
            return self._agent.Act(states, evaluate)

    def SaveMemory(self, state, action, next_state, reward, done):
        transition = (state, action, next_state, reward, done)

        if self._n_step_memory:
            transition = self._n_step_memory.Append(*transition)
        
        if transition:
            self._memory.Append(*transition)
        

