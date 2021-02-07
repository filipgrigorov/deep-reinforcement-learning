import matplotlib.pyplot as plt
import numpy as np

from agent import DDPGAgent
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

    scores = []
    average_scores = deque(maxlen=100)

    # Hyper-parameters:
    batch_size = 32
    memory_size = 1e6
    gamma = 0.99
    eps = 1.0
    min_eps = 0.01
    eps_decay = 0.99
    soft_update_steps = 10000
    agent = DDPGAgent(gamma, 1e-4, batch_size, soft_update_steps, state_size, action_size, memory_size, num_agents)

    nepisodes = 5000

    for episode_idx in range(nepisodes):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment

        states = env_info.vector_observations                  # get the current state (for each agent)

        scores = np.zeros(num_agents)                          # initialize the score (for each agent)

        episodic_scores = []

        while True:
            # (1) Generate action from actor
            actions = agent.step(states)                       # select an action (for each agent)

            # debug
            print(actions)

            env.close()
            raise('debug')

            env_info = env.step(actions)[brain_name]           # send all actions to tne environment

            next_states = env_info.vector_observations         # get next state (for each agent)

            # Normalize and clip rewards (future rewards)
            rewards = env_info.rewards                         # get reward (for each agent)
            episodic_scores.append(np.mean([ R for R in rewards ]))
            scores.append(np.mean([ R for R in rewards ]))

            dones = env_info.local_done                        # see if episode finished

            # (2) Collect experience
            agent.remember(states, actions, rewards, next_states)

            scores += env_info.rewards                         # update the score (for each agent)

            # (3) Act upon experience (learn)
            agent.act()

            states = next_states                               # roll over states to next time step

            if np.any(dones):                                  # exit loop if episode finished
                break

        average_scores.append(np.sum(episodic_scores))

        # Update eps
        eps *= eps_decay
        eps = min(eps, min_eps)

        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

        # Check for last 100 average reward
        if np.mean(average_scores) >= 30:
            print(f'Environment has been solved in {episode_idx - 100} with an average reward of {np.mean(average_scores)}')
            break

    print('End of training')
    env.close()

    # Plot the scores
    plt.plot(scores)
    plt.show()
