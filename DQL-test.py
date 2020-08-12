from envs import RegisteredEnvs
import gym
import torch
from data_structures.UniformExperienceReplay import UniformExperienceReplay
from agents.DQL import DQL
import Config

# device = torch.device('cuda') # move into config.
# env = gym.make("CartPole-v0") # move into config
# memory = UniformExperienceReplay(1000) # TODO generalize this. #move to config

config = {
    "env": "Minesweeper_Text_v0",
    "env_kwargs": {"difficulty": 1, "mode": "one-hot"},
    "device": "cuda",
    "reward_goal": 150,
    "reward_window": 100,
    "memory_size": 100000,
    "batch_size": 64,
    "warm_up": 1000,
    "n_step_update": 1,
    "n_updates_per_learn": 1,
    "max_steps": -1,
    "visualize": False,
    "visualize_frequency": 10,
    "enable_seed": False,
    "seed": 0,
    "hyperparameters": {
        "convo": {
            "filters": [80, 40, 40],
            "kernels": [5, 3, 3],
            "strides": [1, 1, 1],
            "paddings": [2, 2, 1],
            "activations": ["relu", "relu", "relu"],
            "pools": [],
            "flatten": True,
        },
        "fc": {

            "hidden_layers": [
                512,
                # 1024,
                # 1024,
            ],
            "activations": [
                "relu",
                # "relu",
                # "relu",
            ],
            "final_activation": None,
        },
        "lr": 0.0001,
        "epsilon": 0.5,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.001,
        "target_update": 10,
        "tau": 0.1,
        "gamma": 0.99,
        "soft_update": True
    }
}

# Config.Save("configs/DQL_Minesweeper-v0.txt", config)

# config = Config.Load("configs/DQL_Minesweeper-v0.txt")

# TODO make this modular?
config['env'] = RegisteredEnvs.BuildEnv(config['env'], config['env_kwargs'])

agent = DQL(config)
agent.Train()
