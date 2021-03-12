from agent import MADDPG
from collections import deque
from unityagents import UnityEnvironment

import matplotlib.pyplot as plt
import numpy as np
import os
import time

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
    target_score = config['target_score']

    average_over_episodes = config['average_over_episodes']

    actor_weights_path = config['actor_weights_path']
    critic_weights_path = config['critic_weights_path']

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
        'batch_size': 1000,
        'memory_size': int(1e6),
        'gamma': 0.95,
        'tau': 1e-3,
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'update_every': 2 * num_agents,
        'update_iterations': 3,
        'noise_decay': 2
    }

    maddpg = MADDPG(state_size, action_size, num_agents, config)

    # Lists for mean, low and high scores per episode
    mean_scores = []
    min_scores = []
    max_scores = []

    best_score = -np.inf

    # Mean scores for the last "average_over_episodes" episodes
    scores_over_range = deque(maxlen=average_over_episodes)

    # List of moving averages of scores for the last "average_over_episodes" episodes
    moving_avgs = []

    for episode_idx in range(1, num_episodes + 1):                                      # play game for 5 episodes

        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)

        maddpg.reset_noise()

        # Measure time per trajectory
        start_time = time.time()

        for t in range(ntimesteps):
            actions = maddpg.act(states)                             # select an action (for each agent)

            env_info = env.step(actions)[brain_name]           # send all actions to tne environment

            next_states = env_info.vector_observations         # get next state (for each agent)
            
            rewards = env_info.rewards                         # get reward (for each agent)
            
            dones = env_info.local_done                        # see if episode finished

            maddpg.remember(states, actions, rewards, next_states, dones)

            maddpg.step(t)
            
            scores += np.max(rewards)                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            
            if np.any(dones):                                  # exit loop if episode finished
                break
        
        elapsed_time = time.time() - start_time

        min_scores.append(np.min(scores))
        max_scores.append(np.max(scores))
        mean_scores.append(np.mean(scores))

        scores_over_range.append(mean_scores[-1])
        moving_avgs.append(np.mean(scores_over_range))

        print('\rEpisode {} ({} sec)  -- \tMin reward: {:.1f}\tMax reward: {:.1f}\tMean reward: {:.1f}\tMoving Average: {:.1f}'.format(
            episode_idx,
            round(elapsed_time),
            min_scores[-1],
            max_scores[-1],
            mean_scores[-1],
            moving_avgs[-1])
        )

        if mean_scores[-1] > best_score:
            # Note: Save every progress we make
            maddpg.save(actor_weights_path, critic_weights_path)

        if moving_avgs[-1] >= target_score and episode_idx >= average_over_episodes:
            print('\nEnvironment was solved in {} episodes!\tMoving Average ={:.1f} over last {} episodes'.format(
                episode_idx - average_over_episodes, 
                moving_avgs[-1], 
                average_over_episodes
            ))

            maddpg.save(actor_weights_path, critic_weights_path)
            break

    return mean_scores, moving_avgs


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
        'episodes': 9000,
        'ntimesteps': 300,
        'average_over_episodes': 100,
        'target_score': 0.5,
        'num_agents': num_agents, # 2 for each racket
        'state_size': state_size,
        'action_size': action_size,
        'actor_weights_path': os.path.join(root_weights_path, 'actor_weights_checkpoint'),
        'critic_weights_path': os.path.join(root_weights_path, 'critic_weights_checkpoint')
    }

    scores, moving_average_scores = run_training(config)

    plot(scores, moving_average_scores)

    print('Closing the environment!')
    env.close()
