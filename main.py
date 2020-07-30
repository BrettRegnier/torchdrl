import gym
import torch
from Agent import AgentA2C

env = gym.make('CartPole-v0')
num_steps = 10000


learning_rate = 1e-3
gamma = 0.99
device = torch.device('cuda')
agent = AgentA2C(learning_rate, env.observation_space.shape, env.action_space.n, gamma, device)

episodes = 0

running_reward = 10
while True:
	done = False
	is_rendering = False
	episode_reward = 0
	state = env.reset()
	
	step = 0
	
	episodes += 1
	while step < num_steps and not done: 
		if episodes % 50 == 0 and not is_rendering:
			env.render()
			is_rendering = True
		
		action = agent.Act(state)
		
		next_state, reward, done, info = env.step(action)
		agent._rewards.append(reward)
		
		episode_reward += reward
		
		running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
		# agent.Learn_o(state, next_state, reward, done)
	
		state = next_state
		
		step += 1
	# now done
	if is_rendering:	
		env.close()
		is_rendering = False
	
	agent.Learn()

	if episodes % 1 == 0:
		print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
				episodes, episode_reward, running_reward))

	# check if we have "solved" the cart pole problem
	if running_reward > 50:
		print("Solved! Running reward is now {} and "
				"the last episode runs to {} time steps!".format(running_reward, step))