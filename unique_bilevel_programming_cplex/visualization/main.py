import json
from datetime import datetime

import networkx as nx

import unique_bilevel_programming_cplex.src.egm.data_parser as data_parser
from unique_bilevel_programming_cplex.src.base.common import is_lp_nan
import matplotlib.pyplot as plt
import numpy as np
import os



if __name__ == "__main__":
    def visual(year, json_path, countries):
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
                    _, date, c_1, c_2 = edge_name.split("_")
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    c_1 = c_1.replace("export ", "")
                    cc_12 = c_1, c_2
                    if date not in result:
                        result[date] = dict()
                    result[date][cc_12] = flow
                    nodes.update(cc_12)
                    edges.add(cc_12)

        cc_list = list(countries)
        for_plot_1_true = [
            sum(c[d] for d in result for c in data.export_assoc[cc].values() if not is_lp_nan(c[d]))
            for cc in cc_list
        ] + [
            sum(tt["MonthData"][d]["sendOut"] for d in result for tt in data.terminal_db.values())
        ]
        for_plot_1_pred = [
            sum(result[d][cc, c] for d in result for c in nodes if (cc, c) in result[d])
            for cc in cc_list
        ] + [
            sum(result[d][rd] for d in result for rd in result[d] if rd[0] in data.terminal_db)
        ]

        print(for_plot_1_true[-1])
        print(for_plot_1_pred[-1])

        cc_list += ["СПГ"]
        barWidth = 0.25
        br1 = np.arange(len(cc_list))
        br2 = [x + barWidth for x in br1]
        plt.bar(br1, for_plot_1_true, width=barWidth, label="Данные")
        plt.bar(br2, for_plot_1_pred, width=barWidth, label="Рассчет")
        plt.xticks([r + barWidth for r in range(len(cc_list))], cc_list)

        for i, cc in enumerate(br1):
            per = 200 * (for_plot_1_true[i] - for_plot_1_pred[i]) / (for_plot_1_true[i] + for_plot_1_pred[i])
            plt.text(cc, barWidth + for_plot_1_pred[i], s=f'{per:.1f}%')

        plt.legend()
        plt.show()

        dates = [i for i in dates if i in result]
        for_plot_2_pred = {
            cc: [
                sum(result[d][cc, c] for c in nodes if (cc, c) in result[d])
                for d in dates
            ]
            for cc in cc_list[:-1]
        }
        for_plot_2_true = {
            cc: [
                sum(c[d] for c in data.export_assoc[cc].values() if not is_lp_nan(c[d]))
                for d in dates
            ]
            for cc in cc_list[:-1]
        }

        for cc_i in cc_list[:-1]:
            month_pred = for_plot_2_pred[cc_i]
            month_true = for_plot_2_true[cc_i]
            plt.plot(dates, month_pred, label="Данные")
            plt.plot(dates, month_true, label="Рассчет")

            plt.legend()
            plt.show()




    visual(2019, "out/res_2019_mode_1.json", {"DZ", "RU", "NO"})
