import time

from torchdrl.agents.q_learning.QLearningAgent import QLearningAgent
from torchdrl.data_structures.ApexExperienceReplay import ApexExperieceReplay

# TODO The actor's batch size is simply the the memory
class ApexActor(QLearningAgent):
    def __init__(self, id, internal_memory, *args, sync_frequency=10, **kwargs):
        super(ApexActor, self).__init__(*args, **kwargs)

        self._name = f"{self._name}_{id}"

        self._sync_frequency = sync_frequency

        # TODO each memory should instead be a PER
        # TODO except the internal memory
        self._internal_memory = internal_memory
        assert issubclass(type(self._internal_memory), (ApexExperieceReplay,)), "Internal Memory should be a ApexExperieceReplay"

        assert issubclass(type(self._memory), (ApexExperieceReplay,)), "Memory should be a ApexExperieceReplay"
        if self._memory_n_step is not None:
            assert issubclass(type(self._memory_n_step), (ApexExperieceReplay,)), "N step memory should be a ApexExperieceReplay or None"

        self._wait_memories = False

        self._sync_steps = 0
        self._wait_sync = False

    def MemoryLearn(self, states, action, next_state, reward, done):        
        self.StoreMemory(states, action, next_state, reward, done)
        self.WaitSync()
        self.PrepareMemories()

    def PrepareMemories(self):
        if len(self._memory) >= self._batch_size:
            while self._wait_memories:
                time.sleep(0.0001)

            states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np = self._memory.Pop(self._mini_batch)
            states_t, actions_t, next_states_t, rewards_t, dones_t, weights_t = self.ConvertNPWeightedMemoryToTensor(states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np)
            errors = self._loss_function(states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, self._mini_batch, self._gamma)
            errors_np = errors.cpu().detach().numpy()

            self._internal_memory.BatchAppend(states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, self._mini_batch)
            self._wait_memories = True

    def WaitSync(self):
        if self._sync_steps % self._sync_frequency == 0:
            self._sync_steps = 0
            self._wait_sync = True
            while self._wait_sync:
                time.sleep(0.0001)

    def IsWaitingMemories(self):
        return self._wait_memories

    def StopWaitingMemories(self):
        self._wait_memories = False

    def IsWaitingSync(self):
        return self._wait_sync

    def StopWaitingSync(self):
        self._wait_sync = False