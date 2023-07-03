import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.lpp_generator.random_s_t_graph import RandomSTGraph
from src.structures import simple_instance


class LPPMinCostFlow:
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
                    self._graph[v][u]["capacity"] = random.randint(0, 10 * n)
                    self._graph[v][u]["cost"] = random.randint(-10 * n, 10 * n)

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
        a = np.full((self._n_nodes, self._n_edges), 0.0)

        for v in range(self._n_nodes):
            if v != t:
                for w in self._graph.nodes:
                    if self._graph.has_edge(v, w):
                        a[v][self._edge_encoder[v, w]] = 1.0
                    if self._graph.has_edge(w, v):
                        a[v][self._edge_encoder[w, v]] = -1.0

        b = np.full(self._n_nodes, 0.0)
        b[s] = random.randint(0, 10 * self._n_nodes)

        # вектор стоимостей ребер
        c = np.full(self._n_edges, 0.0)
        for v, u in self._graph.edges:
            c[self._edge_encoder[v, u]] = self._graph[v][u]["cost"]

        # нижние и верхние границы переменных
        low_b = np.full(self._n_edges, 0.0)
        up_b = np.full(self._n_edges, 0.0)
        for v, u in self._graph.edges:
            up_b[self._edge_encoder[v, u]] = self._graph[v][u]["capacity"]

        return simple_instance.InvLpInstance(a, b, c, simple_instance.LpSign.Equal, low_b, up_b)

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
            edge_labels={(u, v): self._graph[u][v]["capacity"] for u, v in self._graph.edges},
        )
        plt.show()

    @property
    def edge_decoder(self):
        return self._edge_decoder

    @property
    def source(self):
        return self._s

    @property
    def sink(self):
        return self._t
