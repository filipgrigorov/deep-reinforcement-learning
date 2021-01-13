import copy
import numpy as np

from graphviz import Digraph

class Node:
    def __init__(self, data, left, right):
        self.data = 0
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.current_idx = -1
        self.root = None
        self.left = None
        self.right = None

        self.dot = Digraph()

    def is_empty(self):
        return self.root is None

    def add(self, data):
        if self.is_empty():
            print('Empty root')
            self.root = self._add(self.root, data)
        else:
            if data <= self.root.data:
                print('Going left of root')
                self.root.left = self._add(self.root.left, data)
            else:
                print('Going right of root')
                self.root.right = self._add(self.root.right, data)

    def _add(self, node, data):
        if node is None:
            print('Added a child\n')
            node = Node(data, None, None)
            return node

        if data <= node.data:
            print('Going left')
            return self._add(node.left, data)
        else:
            print('Going right')
            return self._add(node.right, data)

    def traverse(self):
        self.preorder(self.root)

        return self.dot.source

    def preorder(self, node):
        if node is None: return
        self.preorder(node.left)
        print(node)
        self.dot.node(str(node.data), str(node.data))
        self.dot.node(str(node.left.data), str(node.left.data))
        self.dot.node(str(node.left.data), str(node.left.data))
        self.dot.edge(node.data, node.left.data)
        self.dot.edge(node.data, node.right.data)
        self.preorder(node.right)

if __name__ == '__main__':
    l = [ 3, 9, 2, 1, 4, 5 ]

    heap = BinaryTree()

    for idx in range(0, len(l)):
        item = l[idx]
        print('{} Input item: {}'.format(idx, item))
        heap.add(item)

    print(heap.traverse())
