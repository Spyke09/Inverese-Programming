import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import unique_bilevel_programming_cplex.src.egm.data_parser as data_parser
from unique_bilevel_programming_cplex.src.base.common import is_lp_nan

if __name__ == "__main__":
    def visual(year, json_path, countries):
        dates = [datetime(year, i, 1) for i in range(1, 13)]
        parser = data_parser.DataParser(dates)
        data = parser.get_data()

        all_countries = data.cc_list_full
        all_lng = data.graph_db["lngList"]
        export_result = {"flow_pred": dict(), "flow_true": dict()}
        lng_result = {"flow_pred": dict(), "flow_true": dict()}
        with open(json_path, "r") as f:
            result_data = json.load(f)
            for edge_name, flow in result_data["x"].items():
                if "flowMid" in edge_name:
                    _, date, c_1, c_2 = edge_name.split("_")
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    if "sosExport" in c_1 and "sosGetExport" in c_2:
                        c_1 = c_1.replace(" sosExport", "")
                        c_2 = c_2.replace(" sosGetExport", "")
                        export_result["flow_pred"][date, c_1, c_2] = flow
                        export_result["flow_true"][date, c_1, c_2] = data.export_assoc[c_1][c_2][date]

                    if c_1 in all_lng and c_2 == f"{c_1} sos":
                        lng_result["flow_pred"][date, c_1] = flow
                        lng_result["flow_true"][date, c_1] = data.terminal_db[c_1]["MonthData"][date]["sendOut"]

        export_df = pd.DataFrame(export_result)
        export_df["flow_true"] /= 1e5
        export_df["flow_pred"] /= 1e5
        export_df.reset_index(names=["date", "c_1", "c_2"], inplace=True)
        export_df.sort_values("date", inplace=True)
        export_df.to_csv("export_df.csv", sep=";")

        lng_df = pd.DataFrame(lng_result)
        lng_df["flow_true"] /= 1e5
        lng_df["flow_pred"] /= 1e5
        lng_df.reset_index(names=["date", "c_1"], inplace=True)
        lng_df.sort_values("date", inplace=True)
        lng_df.to_csv("lng_df.csv", sep=";")

        cc_list = list(countries)
        for_plot_1_true = [
            export_df["flow_true"][export_df["c_1"] == c_1].sum()
            for c_1 in cc_list
        ] + [
            lng_df["flow_true"].sum()
        ]
        for_plot_1_pred = [
            export_df["flow_pred"][export_df["c_1"] == c_1].sum()
            for c_1 in cc_list
        ] + [
            lng_df["flow_pred"].sum()
        ]
        print(for_plot_1_true)
        print(for_plot_1_pred)


        cc_list += ["СПГ"]
        barWidth = 0.25
        br1 = np.arange(len(cc_list))
        br2 = [x + barWidth for x in br1]
        plt.bar(br1, for_plot_1_true, width=barWidth, label="Данные")
        plt.bar(br2, for_plot_1_pred, width=barWidth, label="Рассчет")
        plt.xticks([r + barWidth for r in range(len(cc_list))], cc_list)

        for i, cc in enumerate(br1):
            per = 200 * (for_plot_1_true[i] - for_plot_1_pred[i]) / (for_plot_1_true[i] + for_plot_1_pred[i])
            h = max(for_plot_1_pred[i], for_plot_1_true[i])
            plt.text(cc, barWidth + h, s=f'{per:.1f}%')

        plt.legend()
        plt.show()

        dates = [i for i in dates if i in export_df["date"].unique()]

        for_plot_2_true = {
            c_1: [export_df["flow_true"][(export_df["c_1"] == c_1) & (export_df["date"] == d)].sum() for d in dates]
            for c_1 in cc_list
        }
        for_plot_2_true.update({
            "СПГ": [lng_df["flow_true"][lng_df["date"] == d].sum() for d in dates]
        })
        for_plot_2_pred = {
            c_1: [export_df["flow_pred"][(export_df["c_1"] == c_1) & (export_df["date"] == d)].sum() for d in dates]
            for c_1 in cc_list
        }
        for_plot_2_pred.update({
            "СПГ": [lng_df["flow_pred"][lng_df["date"] == d].sum() for d in dates]
        })

        for cc_i in cc_list:

            plt.plot(dates, for_plot_2_true[cc_i], label="Данные")
            plt.plot(dates, for_plot_2_pred[cc_i], label="Рассчет")
            plt.title(cc_i)
            plt.legend()
            plt.show()


    year = 2019
    visual(year, f"out/res_{year}_mode_0_2m.json", {"DZ", "RU", "NO", "AZ", "LY", "IR"})
