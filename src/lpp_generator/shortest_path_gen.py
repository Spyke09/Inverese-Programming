import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.structures import simple_instance


class LPPShortestPath:
    """
    Генератор инстансов ЗЛП на основе задачи о поиске кратчайшего пути в графе.
    Граф задается здесь же случайным образом.
    """

    def __init__(self, n, k):
        """
        :param n: число вершин графа
        :param k: кол-во итераций добавления ребер

        TODO: обобщить код и для мультиграфов.
        """
        self._graph: nx.DiGraph = nx.DiGraph()

        for i in range(k):
            for u in range(n):
                v = random.randint(0, n - 1)
                if not self._graph.has_edge(u, v) and u != v:
                    self._graph.add_edge(u, v)
                    self._graph[u][v]["weight"] = random.randint(0, 10*n)

        self._n_edges = len(self._graph.edges)
        self._n_nodes = len(self._graph.nodes)

        self._s, self._t = 0, random.randint(1, self._n_nodes - 1)

        self._edge_encoder = dict()
        c = 0
        for edge in self._graph.edges:
            self._edge_encoder[edge] = c
            c += 1

        self._edge_decoder = {j: i for i, j in self._edge_encoder.items()}

        self.lpp = self.init_lpp()

    def init_lpp(self):
        s, t = self._s, self._t
        a = np.full((2 * self._n_nodes, self._n_edges), 0.0)

        for u, v in self._graph.edges:
            a[u][self._edge_encoder[u, v]] = 1.0
            a[self._n_nodes + u][self._edge_encoder[u, v]] = -1.0

        for v, u in self._graph.edges:
            a[u][self._edge_encoder[v, u]] = -1.0
            a[self._n_nodes + u][self._edge_encoder[v, u]] = 1.0

        b = np.full(2 * self._n_nodes, 0.0)
        b[s] = 1.0
        b[t] = -1.0
        b[self._n_nodes + s] = -1.0
        b[self._n_nodes + t] = 1.0

        c = np.full(self._n_edges, 0.0)
        for v, u in self._graph.edges:
            c[self._edge_encoder[v, u]] = self._graph[v][u]["weight"]

        return simple_instance.LpInstance(a, b, c, np.full(self._n_edges, 0.0), np.full(self._n_edges, 1.0))

    def print_graph(self):
        pos = nx.random_layout(self._graph)
        plt.figure()
        nx.draw(
            self._graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color="pink",
            labels={node: node for node in self._graph.nodes()},
        )
        nx.draw_networkx_edge_labels(
            self._graph,
            pos,
            edge_labels={(u, v): self._graph[u][v]["weight"] for u, v in self._graph.edges},
        )
        plt.show()

    @property
    def edge_decoder(self):
        return self._edge_decoder
