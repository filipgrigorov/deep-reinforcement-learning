import gym
import matplotlib.pyplot as plt
import numpy as np
import sys

from agent import Agent

def run(is_prerained=False):
    env = gym.make('MountainCar-v0')
    env.seed(505)

    agent = Agent(env, alpha=0.02, gamma=0.99, eps=1.0, min_eps=0.01, eps_decay_rate=0.9995)

    if not is_prerained:
        scores = []
        max_avg_score = -np.inf

        nepisodes = 20000
        for idx in np.arange(1, nepisodes + 1):
            state = env.reset()
            action = agent.restart(state)

            total_reward = 0
            done = False

            while not done:
                next_state, reward, done, _ = agent.step(action)
                total_reward += reward
                action = agent.train(next_state, reward, done)

            scores.append(total_reward)

            if len(scores) > 100:
                # Note: Take the average score of the last 100 scores
                avg_score = np.mean(scores[-100 :])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score

            if idx % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(idx, nepisodes, max_avg_score), end="")
                sys.stdout.flush()

        agent.save()
    plt.plot(scores)
    plt.show()
    
    agent.load()

    # Test time
    state = env.reset()
    score = 0
    for t in range(200):
        action = agent.play(state)
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break
    print('\nFinal score:', score)
    env.close()

if __name__ == '__main__':
    run(False)
