import unittest
import torchdrl.tools.Config as Config


class ConfigTests(unittest.TestCase):
    def test_Save(self):
        config = {
            "env": {
                "env_name": "Boat",
                "env_kwargs": {
                    "num_participants": 8,
                    "num_locked": 0,
                    "lr_goal": 0,
                    "lr_relax": 50,
                    "fb_goal": 30,
                    "fb_relax": 0,
                    "rower_state_mode": "2D",
                    "rowers_state_normalize": False,
                    "include_boat_state_weight": True,
                    "reward_mode": "sparse"
                }
            },
            "agent": {
            }
        }

        Config.Save("./test_files/", "config_save.json", config)

        correct = Config.Load("./test_files/config_save.json")

        self.assertEqual(config, correct)

    def test_Load(self):
        test = Config.Load("./test_files/config_load.json")

        correct = {
            "env": {
                "env_name": "BoatLeftRight_v0",
                "env_kwargs": {
                    "num_participants": 8,
                    "num_locked": 0,
                    "lr_goal": 0,
                    "lr_relax": 0,
                    "fb_goal": 30,
                    "fb_relax": 0,
                    "rower_state_mode": "1D",
                    "rowers_state_normalize": False,
                    "include_boat_state_weight": True,
                    "reward_mode": "sparse"
                }
            },
            "agent": {
            }
        }

        self.assertEqual(test, correct)
