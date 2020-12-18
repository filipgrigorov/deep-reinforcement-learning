import gym
import numpy as np
import pickle

def generate_grid(env, bins=[10, 10]):
    nbins = len(bins)
    low_extremes = env.observation_space.low
    high_extremes = env.observation_space.high

    print(f'LOW: {low_extremes} HIGH: {high_extremes}')

    # Note: Break the continuous range into discretized histogram (without the first and last value)
    world_grid = np.array([ np.linspace(low_extremes[idx], high_extremes[idx], bins[idx] + 1)[1 : -1] for idx in range(0, nbins) ])
    return world_grid

# Note: Convert sample from contiuous space into discretized one
def discretize_space(sample, grid):
    pairs = zip(sample, grid)
    return tuple(int(np.digitize(s, g)) for s, g in pairs)

class Agent():
    def __init__(self, env, alpha=0.1, gamma=0.9, eps=0.7, min_eps=0.4, eps_decay_rate=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.min_eps = min_eps
        self.eps_decay_rate = eps_decay_rate

        self.nA = env.action_space.n

        # [nstates, nbins]
        bins = [10, 10]
        self.grid = generate_grid(env, bins)
        print(f'Space: {self.grid.shape}')

        grid_size = tuple([side for side in bins])
        self.grid_size = tuple(grid_size + (self.nA,))
        print(f'Space size: {self.grid_size}')

        self.Q = np.ones(shape=(self.grid_size))
        print('Q shape: ', self.Q.shape)

    def restart(self, state):
        self.eps *= self.eps_decay_rate
        self.eps = max(self.eps, self.min_eps)

        self.last_state = discretize_space(state, self.grid)
        self.last_action = np.argmax(self.Q[self.last_state])
        return self.last_state, self.last_action

    def step(self, action):
        prob = np.random.uniform(0.0, 1.0)
        if prob < self.eps:
            self.last_action = np.random.randint(0, self.nA)
        else:
            self.last_action = np.argmax(self.Q[self.last_state])
        return self.env.step(self.last_action)

    def train(self, next_state, reward, done):
        next_state = discretize_space(next_state, self.grid)
        err = reward + self.gamma * max(self.Q[next_state]) - self.Q[self.last_state + (self.last_action,)]
        self.Q[self.last_state + (self.last_action,)] += self.alpha * err

        # Update
        self.last_state = next_state
        self.last_action = np.argmax(self.Q[self.last_state])

    def load(self):
        with open('q_table.pickle', 'rb') as fh:
            self.Q = pickle.load(fh)

    def save(self):
        with open('q_table.pickle', 'wb') as fh:
            pickle.dump(self.Q, fh)

    def play(self, state):
        state = discretize_space(state, self.grid)
        return np.argmax(self.Q[state])

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    state = env.reset()
    agent = Agent(env, alpha=0.1, gamma=0.9, eps=0.7, min_eps=0.4, eps_decay_rate=0.995)
    state, action = agent.restart(state)
    print('State: ', state)
    print('Action: ', action)
