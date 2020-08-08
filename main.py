import gym
import torch
import numpy as np

# from Agent import AgentA2C
from SAC import AgentSAC
from SACD import AgentSACD
from Memory import UniformExperienceReplay


lr = 1e-3
memory_size = 10000


device = torch.device('cuda')

env = gym.make('CartPole-v0')
agent = AgentSACD(env.observation_space.shape, env.action_space.n, lr, device)
memory = UniformExperienceReplay(memory_size)

max_episodes = 10000
batch_size = 64

max_steps = 10000
total_steps = 0
for episode in range(max_episodes):
	episode_reward = 0
	steps = 0
	done = False

	# TODO do random exploring for awhile
	state = env.reset()
	while steps < max_steps and not done:
		action = agent.Act(state)
		next_state, reward, done, info = env.step(action)
		memory.Append(state, action, reward, next_state, done)

		episode_reward += reward
		state = next_state
		
		if len(memory) > batch_size:
			# TODO fix this later
			# assume not using PER
			batch = memory.Sample(batch_size)
			agent.Learn(batch, 1.0)

		steps += 1

	# done
	total_steps += steps
	print('Episode: %d, reward: %.3f, steps: %d, total steps: %d' % (episode, episode_reward, steps, total_steps))

	if episode_reward >= 200:
		exit()













# hyperparams
# gamma = 0.99
# tau = 1e-2
# alpha = 0.5 # epsilon
# noise_std = 0.2
# delay_step = 2
# memory_size = 1000

# critic_lr = 1e-3
# actor_lr = 1e-3
# alpha_lr = 1e-3

# batch_size = 64
# max_episodes = 10000
# num_steps = 200

# env = gym.make('CartPole-v0')

# device = torch.device('cuda')
# agent = AgentSAC(
# 	env.observation_space.shape, env.action_space.n,
# 	gamma, tau, alpha, critic_lr, actor_lr, alpha_lr, device
# 	)

# buffer = UniformExperienceReplay(memory_size)

# episodes = 0

# episode_rewards = []
# for episode in range(max_episodes):
# 	state = env.reset()
# 	done = False
# 	episode_reward = 0

# 	step = 0
# 	while step < num_steps and not done:
# 		action = agent.Act(state)
# 		next_state, reward, done, info = env.step(np.argmax(action))
# 		buffer.Append(state, action, reward, next_state, done)
# 		episode_reward += reward

# 		state = next_state
# 		step += 1

# 		# learn
# 		if len(buffer) > batch_size:
# 			batch = buffer.Sample(batch_size)
# 			agent.Learn(batch)

# 	# done
# 	episode_rewards.append(episode_reward)
# 	print("Episode: %d, reward: %.3f, mean reward: %.3f" % (episode, episode_reward, np.mean(episode_rewards[-100:])))

# 	if episode_reward == 200:
# 		break







# running_reward = 10
# while True:
# 	done = False
# 	is_rendering = False
# 	episode_reward = 0
# 	state = env.reset()
	
# 	step = 0
	
# 	episodes += 1
# 	while step < num_steps and not done: 
# 		if episodes % 500 == 0 and not is_rendering:
# 			# env.render()
# 			is_rendering = True
		
# 		action = agent.Act(state)
		
# 		next_state, reward, done, info = env.step(action)
# 		agent._rewards.append(reward)
		
# 		episode_reward += reward
		
# 		running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
# 		# agent.Learn_o(state, next_state, reward, done)
	
# 		state = next_state
		
# 		step += 1
# 	# now done
# 	if is_rendering:	
# 		env.close()
# 		is_rendering = False
	
# 	agent.Learn()

# 	if episodes % 1 == 0:
# 		print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
# 				episodes, episode_reward, running_reward))

# 	# check if we have "solved" the cart pole problem
# 	if running_reward > 50:
# 		print("Solved! Running reward is now {} and "
# 				"the last episode runs to {} time steps!".format(running_reward, step))