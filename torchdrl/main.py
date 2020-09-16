import argparse
import threading

import gym
import torch

import shared

from ui.Commandline import Commandline

from data_structures import Config
from envs import RegisteredEnvs

from agents.DQL import DQL
from agents.SAC import SAC
from agents.SACD import SACD

# TODO add visualization
# config = Config.Load("configs/Minesweeper-v0_SACD.txt")

# TODO make this modular? or move into the agent...

def Main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ui", default="commandline", action="store_true")
    args = parser.parse_args()

    shared.init()

    ui = None
    if args.ui == "commandline":
        ui = Commandline()

        config_path = ui.GetConfig()
        # config = Config.Load(config_path)

        # TODO dynamic shaping on convo?
        config = {
            "agent": "SACD",
            "env_name": "DragonBoat_v0",
            "env_kwargs": {"num_participants": 6},
            "device": "cuda",
            "reward_goal": 1000,
            "reward_window": 100,
            "memory_size": 100000,
            "batch_size": 64,
            "warm_up": 10000,
            "n_step_update": 1,
            "n_updates_per_learn": 1,
            "max_steps": 30,
            "visualize": False,
            "visualize_frequency": 10,
            "log": True,
            "show_log": True,
            "show_log_frequency": 1,
            "enable_seed": False,
            "seed": 0,
            "hyperparameters": {
                "convo": True,
                "actor_lr": 0.0001,
                "actor_convo": {
                    "filters": [
                        300
                    ],
                    "kernels": [
                        [2, 3]
                    ],
                    "strides": [
                        1
                    ],
                    "paddings": [
                        0
                    ],
                    "activations": [
                        "relu"
                    ],
                    "pools": [],
                    "flatten": True
                },
                "actor_fc": {
                    "hidden_layers": [
                        1024,
                        1024,
                        1024
                    ],
                    "activations": [
                        "relu",
                        "relu",
                        "relu"
                    ],
                    "dropouts": [
                        0.5,
                        0.5,
                        0.5
                    ],
                    "final_activation": "softmax"
                },
                "actor_gradient_clipping_norm": 5,
                "critic_lr": 0.0001,
                "critic_convo": {
                    "filters": [
                        300
                    ],
                    "kernels": [
                        [2, 3]
                    ],
                    "strides": [
                        1
                    ],
                    "paddings": [
                        0
                    ],
                    "activations": [
                        "relu"
                    ],
                    "pools": [],
                    "flatten": True
                },
                "critic_tau": 0.005,
                "critic_fc": {
                    "hidden_layers": [
                        1024,
                        1024,
                        1024
                    ],
                    "activations": [
                        "relu",
                        "relu",
                        "relu"
                    ],
                    "dropouts": [
                        0.5,
                        0.5,
                        0.5
                    ],
                    "final_activation": None
                },
                "critic_gradient_clipping_norm": 5,
                "alpha_lr": 0.0003
            },
            "gamma": 0.99
        }

        Config.Save("configs/DragonBoat_SACD", config)

        config['env'] = RegisteredEnvs.BuildEnv(config['env_name'], config['env_kwargs'])
        agent = MakeAgent(config['agent'], config)

        train_thread = threading.Thread(target=agent.Train)
        train_thread.daemon = True
        train_thread.start()
        
        ui.Begin()

    elif args.ui == "gui":
        raise NotImplementedError("GUI not implemented")

def MakeAgent(agent_name, config):
    agent = None
    if agent_name == "DQL":
        agent = DQL(config)
    elif agent_name == "SAC":
        agent = SAC(config)
    elif agent_name == "SACD":
        agent = SACD(config)
    else:
        raise AssertionError("No agent with that name")

    return agent

if __name__ == "__main__":
    Main()