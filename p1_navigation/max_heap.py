import copy
import numpy as np

class Node:
    def __init__(self, data, left, right):
        self.data = 0
        self.left = None
        self.right = None

class MaxHeapTree:
    def __init__(self):
        self.current_idx = -1
        self.root = None

    def is_empty(self):
        return self.current_idx < 0

    def add(self, item):
        last_node = self.root
        while True:
            if last_node is None or last_node.item is None:
                self.current_idx += 1
                if self.root is None:
                    print('Root has been created')
                    self.root = Node(item, self.current_idx, self.current_idx - 1, [Node(), Node()])
                    last_node = self.root
                else:
                    print('Child has been created')
                    last_node = Node(item, self.current_idx, self.current_idx - 1, [Node(), Node()])
                self.tree.append(last_node)
                print(f'Added {item} to {last_node.idx} from {last_node.parent_idx}\n')
                break
            side = 'left' if item <= last_node.item else 'right'
            print('Going %s from %s' % (side, last_node.idx))
            last_node = last_node.children[0 if item <= last_node.item else 1]
            #print('After: ', hex(id(last_node)))

        # TODO
        self.balance()

    def balance(self):
        pass

    def dot_repr(self):
        from graphviz import Digraph

        dot = Digraph()

        # Create nodes first
        edges = {}
        for node in self.tree:
            name = str(node.idx)
            dot.node(name, str(node.item))

            if node.children[0] is not None and node.children[1] is not None:
                edges[name] = [ str(node.children[0].idx), str(node.children[1].idx) ]

        print(edges)

        # Create edges second
        for from_idx, to_indices in edges.items():
            dot.edge(from_idx, to_indices[0])
            dot.edge(from_idx, to_indices[1])

        print(dot.source)

if __name__ == '__main__':
    l = [ 3, 9, 2, 1, 4, 5 ]

    heap = MaxHeap()

    for idx in range(0, len(l)):
        item = l[idx]
        print('{} Input item: {}'.format(idx, item))
        heap.add(item)

    heap.dot_repr()
