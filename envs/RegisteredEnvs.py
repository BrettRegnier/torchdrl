import gym
from envs.Minesweeper_Text_v0 import Minesweeper_Text_v0
from envs.FiveNumSort import FiveNumSort

def GymRegistry():
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids

def CustomRegistry():    
    custom_envs = ["Minesweeper_Text_v0", "DragonBoat_v0", "FiveNumSort"]
    return custom_envs

def GymEnv(env_id, args):
    return gym.make(env_id, **args)

def CustomEnv(env_id, args):
    if env_id == "Minesweeper_Text_v0":
        env = Minesweeper_Text_v0(**args)
    elif env_id == "DragonBoat_v0":
        env = Minesweeper_Text_v0(**args)
    elif env_id == "FiveNumSort":
        env = FiveNumSort(**args)

    return env

def BuildEnv(env_id, args):
    if env_id in GymRegistry():
        return GymEnv(env_id, args)
    elif env_id in CustomRegistry():
        return CustomEnv(env_id, args)
    else:
        raise AssertionError("Env_id not found")
