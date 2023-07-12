import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from inverse_programming.src.lpp_generator.random_s_t_graph import RandomSTGraph
from inverse_programming.src.structures import simple_instance


class LPPShortestPath:
    """
    Генератор инстансов ЗЛП на основе задачи о поиске кратчайшего пути в графе.
    Граф задается здесь же случайным образом.
    """

    def __init__(self, n, k):
        """
        :param n: число вершин графа
        :param k: каждый узел соединяется со своими k ближайшими соседями в кольцевой топологии.
        """
        rstg = RandomSTGraph(n, k, 0.1)

        self._graph: nx.DiGraph = rstg.graph
        self._n_nodes, self._n_edges = rstg.n_nodes, rstg.n_edges
        self._s, self._t = rstg.s, rstg.t

        for v in range(n):
            for u in range(n):
                if self._graph.has_edge(v, u):
                    self._graph[v][u]["weight"] = random.randint(0, 10 * n)

        # кодирование ребер
        self._edge_encoder = dict()
        c = 0
        for edge in self._graph.edges:
            self._edge_encoder[edge] = c
            c += 1

        self._edge_decoder = {j: i for i, j in self._edge_encoder.items()}

        self.lpp = self._init_lpp()

    def _init_lpp(self) -> simple_instance.InvLpInstance:
        s, t = self._s, self._t
        n, m = self._n_nodes, self._n_edges
        a = np.full((n, m), 0.0)

        # ограничения равенства из матрицы А
        for u, v in self._graph.edges:
            a[u][self._edge_encoder[u, v]] = 1.0

        for v, u in self._graph.edges:
            a[u][self._edge_encoder[v, u]] = -1.0

        # правые части ограничений в матрице А
        b = np.full(self._n_nodes, 0.0)
        b[s] = 1.0
        b[t] = -1.0

        # вектор стоимостей
        c = np.full(m, 0.0)
        for v, u in self._graph.edges:
            c[self._edge_encoder[v, u]] = self._graph[v][u]["weight"]

        return simple_instance.InvLpInstance(a, b, c, simple_instance.LpSign.Equal, np.full(m, 0.0), np.full(m, 1.0))

    @property
    def start(self):
        return self._s

    @property
    def finish(self):
        return self._t

    def draw_graph(self):
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
