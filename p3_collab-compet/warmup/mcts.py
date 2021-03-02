# Monte-Carlo Tree Search for tic-tac-toe
import numpy as np

'''
x_i_bar: mean node value (reward received)
n_i: node visits
N: parent visits
c: 0.2
'''

# Utilities:
def UCB(rewards, n, N, c=0.2):
    return np.mean(rewards) + c * np.sqrt(np.log(N) / n)

class GameBoard:
    def __init__(self, rows, cols):
        self.rows = rows; self.cols = cols
        self.state = np.zeros(shape=(self.rows, self.cols))
        
class Env:
    def __init__(self, game):
        self.game = game


    def play(self, tree, state, action):
        pass

class MCTree:
    def __init__(self):
        self.v = 0
        self.n = 0
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def max_action(self):
        return max([ UCB(child.v) for child in self.children ])

    def is_visited(self):
        return self.n == 0

    def rollout(self):
        pass


if __name__ = '__main__':
    game_board = GameBoard(layout=(3, 3))
    state = game_board.state
    env = Env(state)

    tree = MCTree()

    action = (np.random.randint(0, state.rows), np.random.randint(0, state.rows))

    for _ in range(5000):
        # action = tuple(row, col)
        state, action = tree.playout(env, state, action)


