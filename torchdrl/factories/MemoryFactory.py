from torchdrl.data_structures.ApexExperienceReplay import ApexExperieceReplay
from torchdrl.data_structures.ExperienceReplay import ExperienceReplay
from torchdrl.data_structures.UniformExperienceReplay import UniformExperienceReplay
from torchdrl.data_structures.PrioritizedExperienceReplay import PrioritizedExperienceReplay
from torchdrl.data_structures.ApexExperienceReplay import ApexExperieceReplay

import gym

def CreateMemory(name, observation_space, kwargs):
    if isinstance(observation_space, (gym.spaces.Tuple, gym.spaces.Dict)):
        input_shape = []
        for space in observation_space:
            try:
                input_shape.append(observation_space.shape)
            except:
                input_shape.append(observation_space.n)
    else:
        input_shape = observation_space.shape

    if name.lower() == "experiencereplay" or name.lower() == "er":
        return ExperienceReplay(input_shape, **kwargs)
    elif name.lower() == "uniformexperiencereplay" or name.lower() == "uer":
        return UniformExperienceReplay(input_shape)
    elif name.lower() == "prioritizedexperiencereplay" or name.lower() == "per":
        return PrioritizedExperienceReplay(input_shape, **kwargs)
    elif name.lower() == "apexexperiencereplay" or name.lower() == "aer":
        return ApexExperieceReplay(input_shape, **kwargs)