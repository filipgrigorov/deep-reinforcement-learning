import numpy as np
from collections import defaultdict
class Agent:
    def __init__(self, nA=6):
        self.alpha = 0.12
        self.gamma = 1.0
        self.eps = 0.4
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, idx, state):
        # Given the state, select an action. Selects eps-greedy selection
        self.eps /= idx
        probs = self.define_eps_greedy_probs(state)
        return np.random.choice(self.nA, p=probs)

    # Q-learning (sarsamax)
    def step(self, state, action, reward, next_state, done):
        # Gt - Qt
        err = self.compute_err('sarsamax', state, action, reward, next_state)
        self.Q[state][action] += self.alpha * err

    def compute_err(self, mode, *args):
        if 'sarsamax' in mode:
            return self.sarsamax(*args)
        elif 'exp_sarsa' in mode:
            return self.expected_sarsa(*args)

    def define_eps_greedy_probs(self, state):
        eps_p = self.eps / self.nA
        probs = np.ones(self.nA) * eps_p
        probs[np.argmax(self.Q[state])] = (1 - self.eps) + eps_p
        return probs

    def sarsamax(self, state, action, reward, next_state):
        return reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]

    def expected_sarsa(self, state, action, reward, next_state):
        return reward + self.gamma * self.define_eps_greedy_probs(next_state).dot(self.Q[next_state].T) - self.Q[state][action]
