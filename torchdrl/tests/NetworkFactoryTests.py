import unittest

import gym
from torchdrl.tools.NeuralNetworkFactory import *


class NetworkFactoryTests(unittest.TestCase):
    def test_CreateNetwork(self):
        network_kwargs = {
            "group": {
                "conv2d": {
                    "filters": [
                        128
                    ],
                    "kernels": [
                        [1, 1]
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
                "fullyconnected": {
                    "hidden_layers": [
                        256
                    ],
                    "activations": [
                        "relu"
                    ],
                    "dropouts": [],
                    "out_features": 32,
                    "final_activation": "relu"
                }
            },
            "sequential": {
                "fullyconnected": {
                    "hidden_layers": [
                    ],
                    "activations": [
                    ],
                    "dropouts": [],
                    "out_features": 1024,
                    "final_activation": "relu"
                }
            },
            "head": {
                "noisyduelingcategorical": {
                    "hidden_layers": [
                        512
                    ],
                    "activations": [
                        "relu"
                    ],
                    "dropouts": [],
                    "v_min": 0,
                    "v_max": 200,
                    "atom_size": 51,
                    "out_features": None,
                    "final_activation": None
                }
            }
        }

        observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low=0, high=300, shape=(4, 4, 2)),
                gym.spaces.Box(low=0, high=3000,
                               shape=(2,))
            )
        )
        out_features = 2
        network = CreateNetwork(
            network_kwargs, observation_space, out_features)
        # output = network.__str__()
        # print(output)

        self.assertTrue(False, "Finish the network factory tests")
