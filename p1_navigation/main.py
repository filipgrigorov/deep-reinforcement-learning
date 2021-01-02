from unityagents import UnityEnvironment
import numpy as np

from agent import Agent

def run(env, state_size, action_size):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score

    eps = 1.0
    min_eps = 0.1
    eps_decay = 0.995
    alpha = 2.5e-4
    gamma = 0.98
    batch_size = 64
    agent = Agent(eps, min_eps, eps_decay, alpha, gamma, batch_size, state_size, action_size)

    while True:
        action = np.random.randint(action_size)        # select an action

        env_info = env.step(action)[brain_name]        # send the action to the environment

        next_state = env_info.vector_observations[0]   # get the next state

        reward = env_info.rewards[0]                   # get the reward

        done = env_info.local_done[0]                  # see if episode has finished

        agent.step(state, action, reward, next_state)

        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step

        agent.act()

        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))

if __name__ == '__main__':
    env = UnityEnvironment(file_name="VisualBanana_Windows_x86_64/Banana.exe")

    brain_name = env.brain_names[0]
    print('brain_name: %s' % brain_name)
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)

    state_size = len(state)
    print('States have length:', state_size)

    run(env, state_size, action_size)
    env.close()
