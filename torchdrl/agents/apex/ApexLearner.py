from torchdrl.agents.QLearningAgent import QLearningAgent
from torchdrl.data_structures.ApexExperienceReplay import ApexExperieceReplay

# TODO the learner should be its own agent as it only requires a memory, no nstep
# and the q_loss function?


class ApexLearner(QLearningAgent):
    def __init__(self, *args, **kwargs):
        super(ApexLearner, self).__init__(*args, **kwargs)
        assert issubclass(type(self._memory), (ApexExperieceReplay,)), "Memory should be a ApexExperieceReplay"

    def StoreMemory(self, state, action, next_state, reward, done, error):
        transition = (state, action, next_state, reward, done, error)

        if transition:
            self._memory.Append(*transition)