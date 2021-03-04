from agent import MADDPG
from unityagents import UnityEnvironment

import matplotlib.pyplot as plt
import numpy as np

ENV_NAME = "Tennis_Windows_x86_64/Tennis.exe"

def plot(scores, moving_average_scores):
    len_scores = len(scores)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len_scores), scores, label='scores')
    plt.plot(np.arange(len_scores), moving_average_scores, c='r', label='moving average')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left')
    plt.show()

def run_training(config):
    env = config['env']
    brain_name = config['brain_name']
    num_agents = config['num_agents']

    state_size = config['state_size']
    action_size = config['action_size']

    print(f'Brain name ({brain_name})')
    print(f'States({state_size}), Actions({action_size})')

    num_episodes = config['episodes']

    ntimesteps = config['ntimesteps']

    config = {
        'seed': 1,
        'batch_size': 128,
        'memory_size': int(1e5),
        'gamma': 0.99,
        'tau': 1e-3,
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'update_every': 20,
        'update_iterations': 10,
        'noise_decay': 0.999
    }

    maddpg = MADDPG(state_size, action_size, num_agents, config)

    for episode_idx in range(1, num_episodes + 1):                                      # play game for 5 episodes

        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)

        maddpg.reset_noise()

        for t in range(ntimesteps):
            actions = maddpg.act(states)                             # select an action (for each agent)

            env_info = env.step(actions)[brain_name]           # send all actions to tne environment

            next_states = env_info.vector_observations         # get next state (for each agent)
            
            rewards = env_info.rewards                         # get reward (for each agent)
            
            dones = env_info.local_done                        # see if episode finished

            maddpg.remember(states, actions, rewards, next_states, dones)

            maddpg.step()

            raise('debug')
            
            scores += env_info.rewards                         # update the score (for each agent)
            
            states = next_states                               # roll over states to next time step
            
            if np.any(dones):                                  # exit loop if episode finished
                break
        
        print('Score (max over agents) from episode {}: {}'.format(episode_idx, np.max(scores)))



if __name__ == '__main__':
    env = UnityEnvironment(file_name=ENV_NAME)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)

    action_size = brain.vector_action_space_size

    states = env_info.vector_observations
    state_size = states.shape[1]

    root_weights_path = 'weights'
    config = {
        'env': env,
        'brain_name': brain_name,
        'episodes': 2000,
        'ntimesteps': 1000,
        'num_agents': num_agents, # 2 for each racket
        'state_size': state_size,
        'action_size': action_size,
        'weights_path': root_weights_path
    }

    scores, moving_average_scores = run_training(config)

    plot(scores, moving_average_scores)
