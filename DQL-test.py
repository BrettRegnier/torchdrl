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
    "env_kwargs": {"difficulty": 1, "flat": True},
    "device": "cuda",
    "reward_goal": 150,
    "reward_window": 100,
    "memory_size": 100000,
    "batch_size": 256,
    "warm_up": 1000,
    "n_step_update": 1,
    "n_updates_per_learn": 1,
    "max_steps": -1,
    "visualize": False,
    "visualize_frequency": 10,
    "hyperparameters": {
        "hidden_layers": [
            1024,
            1024,
            1024,
        ],
        "activations": [
            "relu",
            "relu",
            "relu",
        ],
        "final_activation": None,
        "lr": 0.0001,
        "epsilon": 0.3,
        "epsilon_decay": 5e-05,
        "epsilon_min": 0.001,
        "target_update": 10,
        "tau": 0.1,
        "gamma": 0.99,
        "soft_update": True
    }
}

# config = Config.Load("configs/DQL_CartPole-v0.txt")

# TODO make this modular?
from envs import RegisteredEnvs
config['env'] = RegisteredEnvs.BuildEnv(config['env'], config['env_kwargs'])

agent = DQL(config)
agent.Train()
