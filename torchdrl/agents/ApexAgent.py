import time
import threading

import torchdrl.tools.Helper as Helper
from torchdrl.agents.apex.ApexActor import ApexActor
from torchdrl.agents.Agent import Agent

class ApexAgent:
    def __init__(self, learner:Agent, actors:list, num_threads:int):
        #TODO error checking on learner
        self._learner = learner
        if not isinstance(actors, (list,)):
            assert issubclass(type(actors), ApexActor), "Actors must be an ApexActor or Inherited ApexActor"
            self._actors = [actors]
        else:
            for i, actor in enumerate(actors):
                assert issubclass(type(actor), ApexActor), "Actor at index " + str(i) + " is not an ApexActor or Inherited ApexActor"
            self._actors = actors
        self._num_actors = len(self._actors)
        self._num_threads = num_threads

        # params
        
        self._step_window=10 # TODO
        self._reward_window=100 # TODO
        self._reward_goal=None # TODO
        self._max_steps_per_episode=-1 # TODO
        self._warm_up=0 # TODO

        self._evaluate_episodes=100 # TODO
        self._checkpoint_freq = 100 # TODO 

        self._train_checkpoint=False # TODO
        self._evaluate_checkpoint=False # TODO

        # local vars
        self._stop = False
        self._pause = False
        self._num_ep_last_save = 0

        # learner metrics
        self._total_learns = 0

        # actor metrics
        self._episode_infos = []


    def Train(self):
        actors_per_thread = self._num_actors // self._num_threads
        extra_actors = self._num_actors % self._num_threads

        actor_groups = []
        actor_threads = []
        actor_idx = 0

        for i in range(self._num_threads):
            actor_groups.append([])
            for _ in range(actors_per_thread):
                actor_groups[i].append(self._actors[actor_idx])
                actor_idx += 1
        
        while extra_actors > 0:
            for group in actor_groups:
                group.append(self._actors[actor_idx])
                actor_idx += 1
                extra_actors -= 1
                if extra_actors == 0:
                    break

        for group in actor_groups:
            thread = threading.Thread(target=self.TrainActors, args=(group,))
            thread.daemon = True

            actor_threads.append(thread)

        for thread in actor_threads:
            thread.start()

        
        learner_thread = threading.Thread(target=self.TrainLearner, args=())
        learner_thread.daemon = True
        learner_thread.start()

        while not self._stop:
            self.TransferMemories()

            time.sleep(0.0001)
            while self._episode_infos:
                self._num_ep_last_save += 1
                # TODO find the best actor
                yield self._episode_infos.pop(0)

            if self._num_ep_last_save > self._checkpoint_freq:
                self.Save()
                self._num_ep_last_save = 0

        self.Save()

    def TrainNoYield(self):
        for info in self.Train():
            print(info);

    def Evaluate(self):
        pass

    def TrainLearner(self):
        while not self._stop:
            self._learner.Learn()
            self.UpdateActors()

            self._total_learns +=1

    def TrainActors(self, actors):
        train_calls = []
        for actor in actors:
            train_calls.append(actor.Train())
        zipped_train_calls = zip(*train_calls)

        for train_call in zipped_train_calls:
            for train_info, train_msg, test_info, test_msg in train_call:
                self._episode_infos.append((train_info, train_msg, test_info, test_msg))


        # If one of the actors finished, means that we hit a good point
        # all other trains will finish since they are yield
        self._stop = True

    def UpdateActors(self):
        # copy the learner to the actors
        for actor in self._actors:
            if actor.IsWaitingSync():
                Helper.UpdateNetwork(self._learner._network, self._actors._network)
                Helper.UpdateNetwork(self._learner._target_network, self._actors._target_network)
                actor.StopWaitingSync()
        
    
    def TransferMemories(self):
        for actor in self._actors:
            if actor.IsWaitingMemories():
                states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, _ = actor._memory.Pop(self._mini_batch_size)
                for i in range(self._mini_batch_size):
                    self._learner.StoreMemory(states_np[i], actions_np[i], next_states_np[i], rewards_np[i], dones_np[i], errors_np[i])

                actor.StopWaitingMemories()

