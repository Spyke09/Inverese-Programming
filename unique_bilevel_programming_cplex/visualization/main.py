import json
from datetime import datetime

import networkx as nx

import unique_bilevel_programming_cplex.src.egm.data_parser as data_parser
import matplotlib.pyplot as plt

import os


os.chdir("../")


if __name__ == "__main__":
    def visual(year, json_path, cc):
        dates = [datetime(year, i, 1) for i in range(1, 13)]
        parser = data_parser.DataParser(dates)
        data = parser.get_data()

        all_nodes = set(j for c_1, ccc in data.export_assoc.items() for j in ccc)
        result = dict()
        edges = set()
        nodes = set()
        with open(json_path, "r") as f:
            result_data = json.load(f)
            for edge_name, flow in result_data["x"].items():
                if "flowMid" in edge_name:
                    edge_name_split = edge_name.split("_")
                    if len(edge_name_split) == 5:
                        edge_name_split[3] = "_".join(edge_name_split[3:])
                        edge_name_split.pop(4)
                    elif len(edge_name_split) == 6 and (edge_name_split[2] == "export EX" or edge_name_split[2] == "export ASI"):
                        edge_name_split[2] = "_".join(edge_name_split[2:-1])
                        edge_name_split.pop(4)
                        edge_name_split.pop(3)
                    elif len(edge_name_split) == 7 and (edge_name_split[2] == "export EX" or edge_name_split[2] == "export ASI"):
                        edge_name_split[2] = "_".join(edge_name_split[2:-1])
                        edge_name_split.pop(4)
                        edge_name_split.pop(3)
                        edge_name_split[3] = "_".join(edge_name_split[3:])
                        edge_name_split.pop(4)

                    date = datetime.strptime(edge_name_split[1], "%Y-%m-%d %H:%M:%S")

                    c_1, c_2 = edge_name_split[2:]
                    c_1 = c_1.replace("export ", "")
                    cc_12 = c_1, c_2
                    if c_1 in cc and c_2 in all_nodes:
                        if date not in result:
                            result[date] = dict()
                        result[date][cc_12] = flow
                        nodes.update(cc_12)
                        edges.add(cc_12)

        for date in sorted(result.keys()):
            flow = result[date]
            graph = nx.DiGraph()

            edge_labels = dict()
            true_edge = set()
            true_nodes = set()
            for edge in edges:
                a = round(flow[edge] / 1000, 2)
                b = round(data.export_assoc[edge[0]][edge[1]][date] / 1000, 2)
                if a > 0 or b > 0:
                    true_edge.add(edge)
                    true_nodes.update(edge)
                    edge_labels[edge] = f"{a} <-> {b}"

            graph.add_nodes_from(true_nodes)
            graph.add_edges_from(true_edge)

            print(date)

            plt.figure()
            nx.draw(
                graph, pos=nx.circular_layout(graph), edge_color='black', width=1, linewidths=1,
                node_size=500, node_color="pink", labels={node: node for node in graph.nodes()}
            )

            nx.draw_networkx_edge_labels(
                graph, nx.circular_layout(graph), edge_labels=edge_labels, font_color="red", font_size=8
            )

            plt.show()


    visual(2019, "out/res_2019_mode_1_6m.json", {"DZ", "RU", "NO"})
