import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from collections import deque
from unityagents import UnityEnvironment

if __name__ == '__main__':
    env = UnityEnvironment(file_name='Reacher_Windows_x86_64_many/Reacher.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print(f'Number of agents: {num_agents}')

    action_size = brain.vector_action_space_size

    states = env_info.vector_observations
    state_size = states.shape[1]

    series_scores = []
    average_scores = deque(maxlen=100)

    # Hyper-parameters:
    batch_size = 64
    memory_size = 1e5
    gamma = 0.99
    actor_lr = 1e-4
    critic_lr = 1e-4
    agent = Agent(gamma, actor_lr, critic_lr, batch_size, state_size, action_size, memory_size, num_agents)

    nepisodes = 600

    for episode_idx in range(nepisodes):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment

        states = env_info.vector_observations                  # get the current state (for each agent)

        episodic_scores = np.zeros(num_agents)                          # initialize the score (for each agent)

        agent.reset()

        while True:
            # (1) Generate action from actor
            actions = agent.step(states)                       # select an action (for each agent)

            env_info = env.step(actions)[brain_name]           # send all actions to tne environment

            next_states = env_info.vector_observations         # get next state (for each agent)

            # Normalize and clip rewards (future rewards)
            rewards = env_info.rewards                         # get reward (for each agent)

            dones = env_info.local_done                        # see if episode finished

            # (2) Collect experience
            agent.remember(states, actions, rewards, next_states, dones)

            episodic_scores += env_info.rewards                         # update the score (for each agent)

            # (3) Act upon experience (learn)
            agent.act()

            states = next_states                               # roll over states to next time step

            if np.any(dones):                                  # exit loop if episode finished
                break

        series_scores.append(np.mean(episodic_scores))
        average_scores.append(np.mean(episodic_scores))

        print('Total score (averaged over agents) for episode {}: {}'.format(episode_idx, np.mean(average_scores)))

        # Check for last 100 average reward
        if np.mean(average_scores) >= 30:
            print(f'Environment has been solved in {episode_idx - 100} with an average reward of {np.mean(average_scores)}')
            break

    print('End of training')
    env.close()

    # Plot the scores
    plt.plot(series_scores)
    plt.show()
