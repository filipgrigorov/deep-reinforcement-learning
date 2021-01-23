import copy
import numpy as np

from graphviz import Digraph

dot = Digraph()

class Tree:
    def __init__(self, idx, data):
        self.left = None
        self.right = None

        self.idx = idx
        self.data = data

        print(f'Created a node with data {self.data}')

    def is_empty(self):
        return self.data is None

    def insert(self, idx, data):
        if not self.left and data <= self.data:
            print(f'Left: {self.idx} -> {data}')
            self.left = Tree(idx, data)
            return
        elif not self.right and data > self.data:
            print(f'Right: {self.idx} -> {data}')
            self.right = Tree(idx, data)
            return
        
        if data <= self.data:
            print(f'Left: {self.idx} -> {self.left.data}')
            self.left.insert(idx, data)
        else:
            print(f'Right: {self.idx} -> {self.right.data}')
            self.right.insert(idx, data)

    def serialize(self, serialization):
        serialization += f'Idx={self.idx} Data={self.data}   '

        if self.left:
            print(f'Serialize {self.left.idx}')
            serialization += self.left.serialize(serialization)
        if self.right:
           print(f'Serialize {self.right.idx}')
           serialization += self.right.serialize(serialization)

        return serialization

    def __str__(self):
        serialization = self.serialize('')
        return serialization

if __name__ == '__main__':
    l = [ 3, 9, 2, 1, 4, 5 ]

    # Note: Return a tree with an initialized data
    tree = Tree(idx=0, data=l[0])

    for idx in range(1, len(l)):
        tree.insert(idx, l[idx])

    print('\n')
    print(tree)
