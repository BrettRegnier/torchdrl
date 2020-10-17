from torchdrl.data_structures.PrioritizedExperienceReplayTorch import PrioritizedExperienceReplayTorch
from torchdrl.data_structures.PrioritizedExperienceReplay import PrioritizedExperienceReplay
from DragonFruit.env.BoatDiscrete_v0 import BoatDiscrete_v0

import numpy as np
import time

env_kwargs = {
        "num_participants": 4,
        "num_locked": 0,
        "lr_goal": 0,
        "fb_goal": 30
}

env = BoatDiscrete_v0(**env_kwargs)
device = 'cpu'
cpu_version = PrioritizedExperienceReplay(100000)
gpu_version = PrioritizedExperienceReplayTorch(100000, env.observation_space.shape, device)

num_times = 1000

t0 = time.time()
# populate first
state = env.reset()
done = False
for _ in range(num_times):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    cpu_version.Append(state, action, next_state, reward, done)

    state = next_state

# retrieve, populate and update
for _ in range(num_times):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    cpu_version.Append(state, action, next_state, reward, done)

    state = next_state

    s, a, n, r, d, indices, w = cpu_version.Sample(128)
    errors = np.random.rand(128)

    cpu_version.BatchUpdate(indices, errors)

t1 = time.time()

print("done cpu", t1-t0)

t0 = time.time()
state = env.reset()
done = False
for i in range(num_times):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    gpu_version.Append(state, action, next_state, reward, done)

    state = next_state

# retrieve, populate and update
for i in range(num_times):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    gpu_version.Append(state, action, next_state, reward, done)

    state = next_state

    s, a, n, r, d, indices, w = gpu_version.Sample(128)
    errors = np.random.rand(128)

    gpu_version.BatchUpdate(indices, errors)
t1 = time.time()

print("done torch", t1-t0)


