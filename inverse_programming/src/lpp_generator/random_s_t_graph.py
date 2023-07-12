import random

import networkx as nx


class RandomSTGraph:
    def __init__(self, n, k, p):
        self.graph: nx.DiGraph = nx.connected_watts_strogatz_graph(n, k, p).to_directed()
        for u in range(n):
            for v in range(u + 1, n):
                if self.graph.has_edge(v, u):
                    if random.randint(0, 1):
                        self.graph.remove_edge(u, v)
                    else:
                        self.graph.remove_edge(v, u)

        self.n_edges = len(self.graph.edges)
        self.n_nodes = len(self.graph.nodes)

        self.s, self.t = 0, random.randint(1, self.n_nodes - 1)
