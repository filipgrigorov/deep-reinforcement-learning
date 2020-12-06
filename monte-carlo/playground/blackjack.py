import gym
import numpy as np
import sys

from collections import defaultdict

actions = ['stick', 'hit']

def count_successive_reward(residual_episodes, cummulative_reward):
    if len(residual_episodes) < 1:
        return cummulative_reward
    r = residual_episodes[0][2]
    return count_successive_reward(residual_episodes[1 : ], r + cummulative_reward)

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')

    print('Action space: %s\n' % env.action_space)
    
    # Initialize Q-table, rows=states, columns=actions

    # policy: sum > 18 => p = 0.8 for "stick" else p = 1 - 0.8
    p = 0.8

    nactions = env.action_space.n
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    nepisodes = 6000
    for i in range(0, nepisodes):
        state = env.reset()

        episode = []
        is_terminal = False
        while not is_terminal:
            # policy: 80% chance to stick if the sum > 18, 20% otherwise
            action = int(round((1 - p) * 1 if state[0] > 18 else p * 1, 0))
            print(f'From St={state}, At={actions[action]}')

            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state

            print(f' to St+1={state}, Rt+1={reward}')
            if done:
                print(6 * '\t' + 'Outcome: You {}'.format('won' if reward > 0 else 'lost'))
                is_terminal = True

        # MC First-visit
        visited = []
        for ii in range(0, len(episode)):
            s, a, r = episode[ii]
            # first-time in this (s, q, r)
            if s not in visited:
                visited.append(s)
                # For each state count the cummulative reward and add it to the q-table
                # By the law of large numbers, as n -> inf, mean_rewards = E[Gt | St, At]
                
                # N(s, a) += 1
                N[s][a] += 1

                # Q(s, a) += Gt
                q_table[s][a] += count_successive_reward(episode[ii + 1 :], 0.0)

    for (s, v) in q_table.items():
        q_table[s][a] /= N[s][a]
        q_table[s][a] = round(q_table[s][a], 4)

    print(q_table)
    print('End')
