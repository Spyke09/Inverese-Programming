import json
import typing as tp
from dataclasses import dataclass
from datetime import datetime

import numpy
import numpy as np

from unique_bilevel_programming_cplex.src.base.model import Model
from unique_bilevel_programming_cplex.src.base.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.base.var_expr_con import Var

LPNan = numpy.nan

@dataclass
class EGMData:
    cc_list_full: tp.Any
    consumption_production_assoc: tp.Any
    export_assoc: tp.Any
    graph_db: tp.Any
    prices_assoc: tp.Any
    storage_db: tp.Any
    terminal_db: tp.Any


class DataParser:
    @staticmethod
    def _process_num(num):
        return LPNan if num == "Missing" else np.float64(num)

    @staticmethod
    def _process_date(date):
        return date if isinstance(date, datetime) else datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')

    @staticmethod
    def get_data():
        with open("../../data/ccListFull.json", "r") as f:
            cc_list_full = set(json.load(f))
        with open("../../data/consumptionProductionAssoc.json", "r") as f:
            consumption_production_assoc = json.load(f)
            consumption_production_assoc['consumption'] = consumption_production_assoc['consumption']["bcm"]
            consumption_production_assoc['production'] = consumption_production_assoc['production']["bcm"]
            consumption_production_assoc = {
                name: {
                    cou: {DataParser._process_date(d): DataParser._process_num(c) for d, c in dtc.items()}
                    for cou, dtc in pc.items()
                }
                for name, pc in consumption_production_assoc.items()
            }
        with open("../../data/exportAssoc.json", "r") as f:
            export_assoc = json.load(f)
            export_assoc = export_assoc["bcm"]
            export_assoc = {
                c1: {
                    c2: {
                        DataParser._process_date(d): DataParser._process_num(c) for d, c in expo.items()
                    }
                    for c2, expo in assoc.items()
                }
                for c1, assoc in export_assoc.items()
            }
        with open("../../data/graphDB.json", "r") as f:
            graph_db = json.load(f)
            graph_db['arcCapTimeAssoc'] = {
                DataParser._process_date(d): {(edge[0], edge[1]): DataParser._process_num(edge[2]) for edge in edges}
                for d, edges in graph_db['arcCapTimeAssoc'].items()
            }
            graph_db['arcList'] = set(tuple(i) for i in graph_db['arcList'])
            graph_db['tsoList'] = set(graph_db['tsoList'])
            graph_db['lngList'] = set(graph_db['lngList'])
            graph_db['storList'] = set(graph_db['storList'])
            graph_db['consumVertexList'] = set(graph_db['consumVertexList'])
            graph_db['consumList'] = set(graph_db['consumList'])
            graph_db['prodVertexList'] = set(graph_db['prodVertexList'])
            graph_db['prodList'] = set(graph_db['prodList'])
            graph_db['exporterVertexList'] = set(graph_db['exporterVertexList'])
            graph_db['exporterList'] = set(graph_db['exporterList'])
            graph_db['exportDirections'] = {i: set(j) for i, j in graph_db['exportDirections'].items()}
        with open("../../data/priceAssoc.json", "r") as f:
            prices_assoc = json.load(f)
            prices_assoc = {
                name: {
                    DataParser._process_date(d): DataParser._process_num(n) for d, n in pc.items()
                }
                for name, pc in prices_assoc.items()
            }
        with open("../../data/storageDB.json", "r") as f:
            storage_db = json.load(f)
            storage_db = storage_db["aggregated"]
            storage_db = {
                name: {
                    "CC": st["CC"],
                    "DayData": {DataParser._process_date(d): {c: DataParser._process_num(n) for c, n in ns.items()}
                                for d, ns in st["DayData"].items()},
                    "MonthData": {DataParser._process_date(d): {c: DataParser._process_num(n) for c, n in ns.items()}
                                  for d, ns in st["MonthData"].items()}
                }
                for name, st in storage_db.items()
            }
        with open("../../data/terminalDB.json", "r") as f:
            terminal_db = json.load(f)
            terminal_db = {
                name: {
                    "CC": st["CC"],
                    "DayData": {DataParser._process_date(d): {c: DataParser._process_num(n) for c, n in ns.items()}
                                for d, ns in st["DayData"].items()},
                    "MonthData": {DataParser._process_date(d): {c: DataParser._process_num(n) for c, n in ns.items()}
                                  for d, ns in st["MonthData"].items()}
                }
                for name, st in terminal_db.items()
            }

        return EGMData(
            cc_list_full,
            consumption_production_assoc,
            export_assoc,
            graph_db,
            prices_assoc,
            storage_db,
            terminal_db
        )


class EGRMinCostFlowModel:
    def __init__(self, data: EGMData, dates: tp.List[datetime]):
        self._data = data
        self._dates = dates

        self._model = None
        self._ubmodel = None

        self._f_T_pi = dict()
        self._f_pi_prod = dict()
        self._f_arc = dict()
        self._f_cons_c = dict()
        self._f_c_D = dict()
        self._f_ugs = dict()

        # верхние границы
        self._known_ub = list()
        self._unknown_ub = list()

        self._init_model()
        self._init_ub_model()

    def _init_model(self):
        graph = self._data.graph_db
        m = self._model = Model()

        sos_lng = {f"{i} sos": i for i in graph["lngList"]}
        sos_stor_in = {f"{i} sos in": i for i in graph["storList"]}
        sos_stor_out = {f"{i} sos out": i for i in graph["storList"]}
        vertex_in = set.union(graph["prodVertexList"], graph["exporterVertexList"], graph["lngList"])

        self._arcs_fan_in = arcs_fan_in = {i[1]: set() for i in graph["arcList"]}
        self._arcs_fan_out = arcs_fan_out = {i[0]: set() for i in graph["arcList"]}
        _update_fan_in = (lambda x: None if x in arcs_fan_in else arcs_fan_in.update({x: set()}))
        _update_fan_out = (lambda x: None if x in arcs_fan_out else arcs_fan_out.update({x: set()}))
        update_fan = (lambda x: (_update_fan_in(x), _update_fan_out(x)))
        for v, w in graph["arcList"]:
            if (v in graph["lngList"] or v in graph["prodVertexList"] or v in graph["storList"] or
                    w in graph["consumVertexList"] or w in graph["storList"]):
                if w in graph["consumVertexList"]:
                    s = f"{w} sos"
                elif v in graph["lngList"] or v in graph["prodVertexList"]:
                    s = f"{v} sos"
                elif v in graph["storList"]:
                    s = f"{v} sos out"
                elif w in graph["storList"]:
                    s = f"{w} sos in"
                else:
                    raise ValueError
                update_fan(s)
                arcs_fan_out[v].add(s)
                arcs_fan_out[s].add(w)
                arcs_fan_in[s].add(v)
                arcs_fan_in[w].add(s)
            else:
                arcs_fan_out[v].add(w)
                arcs_fan_in[w].add(v)

        for c1, c_out in graph['exportDirections'].items():
            c1 = f"export {c1}"
            for c2 in c_out:
                update_fan(c2)
                arcs_fan_out[c1].add(c2)
                arcs_fan_in[c2].add(c1)

        cc_tso = {i: set() for i in graph["vertexCountryAssoc"].values()}
        for i, j in graph["vertexCountryAssoc"].items():
            cc_tso[j].add(i)

        vertex_out = graph["consumVertexList"]

        self._f_T_pi = f_T_pi = {v: Var(f"flow_Theta_pi_({v})") for v in vertex_in}
        self._f_pi_prod = f_pi_prod = {d: {v: Var(f"flow_({d})_pi_({v})") for v in vertex_in} for d in self._dates}
        self._f_arc = f_arc = {
            d: {(i, j): Var(f"flow_({d})_({i},{j})") for i in arcs_fan_out for j in arcs_fan_out[i]}
            for d in self._dates
        }
        self._f_cons_c = f_cons_c = {d: {v: Var(f"flow_({d})_({v})_c") for v in vertex_out} for d in self._dates}
        self._f_c_D = f_c_D = {v: Var(f"flow_c_D_({v})") for v in vertex_out}

        delta = self._dates[1] - self._dates[0]
        exp_dates = [self._dates[0] - delta] + self._dates + [self._dates[-1] + delta]
        self._f_ugs = f_ugs = {
            u: {
                di: Var(f"flow_ugs_({u})_({di})_({exp_dates[i + 1]})") for i, di in enumerate(exp_dates[:-1])
            } for u in graph["storList"]
        }

        # верхние границы
        self._known_ub = known_ub = []
        self._unknown_ub = unknown_ub = []

        # супер-вершина -> pi
        for v in vertex_in:
            unknown_ub.append(m.add_constr(f_T_pi[v].e <= LPNan))

        for d in self._dates:
            # {страна экспортер, страна производитель, пхг} :> {tso, ugs}
            for v in vertex_in:
                unknown_ub.append(m.add_constr(f_pi_prod[d][v].e <= LPNan))
            for v1, fanout in arcs_fan_out.items():
                for v2 in fanout:
                    edge = v1, v2
                    cap = LPNan
                    if v1 in sos_lng and v2 in graph["tsoList"]:
                        cap = graph["arcCapTimeAssoc"][d][sos_lng[v1], v2]
                    if v1 in graph["lngList"] and v2 in sos_lng:
                        cap = self._data.terminal_db[v1]["MonthData"][d]["dtrs"]
                    if v1 in graph["tsoList"] and v2 in graph["tsoList"]:
                        cap = graph["arcCapTimeAssoc"][d][edge]
                    if v1 in sos_stor_in and v2 in graph["storList"]:
                        cap = self._data.storage_db[v2]["MonthData"][d]["injectionCapacity"]
                    if v1 in graph["storList"] and v2 in sos_stor_out:
                        cap = self._data.storage_db[v1]["MonthData"][d]["withdrawalCapacity"]
                    temp = m.add_constr(
                        f_arc[d][edge].e <= cap
                    )
                    (unknown_ub if cap is LPNan else known_ub).append(temp)

            for v in vertex_out:
                unknown_ub.append(m.add_constr(f_cons_c[d][v].e <= LPNan))

        for v in vertex_out:
            unknown_ub.append(m.add_constr(f_c_D[v].e <= LPNan))

        for d in exp_dates[:-1]:
            for v in graph["storList"]:
                cap = self._data.storage_db[v]["MonthData"][d]["workingGasVolume"]
                temp = m.add_constr(
                    f_ugs[v][d].e <= cap
                )
                (unknown_ub if cap is LPNan else known_ub).append(temp)

        # балансы
        for v in vertex_in:
            m.add_constr(sum(f_pi_prod[d][v] for d in self._dates) == f_T_pi[v])
            for d in self._dates:
                m.add_constr(sum(f_arc[d][v, w] for w in arcs_fan_out[v]) == f_pi_prod[d][v])
        for v in graph["lngList"]:
            m.add_constr(sum(f_pi_prod[d][v] for d in self._dates) == f_T_pi[v])
            for d in self._dates:
                m.add_constr(sum(f_arc[d][v, w] for w in arcs_fan_out[v]) == f_pi_prod[d][v])
        for v in vertex_out:
            m.add_constr(sum(f_cons_c[d][v] for d in self._dates) == f_c_D[v])
            for d in self._dates:
                m.add_constr(sum(f_arc[d][w, v] for w in arcs_fan_in[v]) == f_cons_c[d][v])
        for i, d in enumerate(self._dates):
            for v in set(arcs_fan_out.keys()).intersection(arcs_fan_in.keys()):
                in_ = sum(f_arc[d][w, v] for w in arcs_fan_in[v])
                out_ = sum(f_arc[d][v, w] for w in arcs_fan_out[v])
                if v in graph["storList"]:
                    out_ += f_ugs[v][d]
                    in_ += f_ugs[v][exp_dates[i]]
                m.add_constr(in_ == out_)

        # from_A_to_B
        for d in self._dates:
            for c1, c_out in graph['exportDirections'].items():
                for c2 in c_out:
                    r_h = sum(
                        f_arc[d][tso_1, tso_2]
                        for tso_1 in cc_tso[c1] for tso_2 in cc_tso[c2] if tso_2 in arcs_fan_out[tso_1]
                    )
                    m.add_constr(f_arc[d][f"export {c1}", c2].e == r_h)

        # естественные ограничения
        m.add_constrs(v.e >= 0 for v in m.vars)

        print(len(m.vars))
        print(len(m.constraints))

    def _init_ub_model(self):
        m = self._model
        ub_m = UBModel(self._model)
        c = ub_m.init_c_as_var()
        b = ub_m.init_b_as_var(self._unknown_ub)

        x_0 = self._get_x_0()
        print(x_0)

    def _get_x_0(self):
        graph = self._data.graph_db

        x_0 = dict()

        sos_lng = {f"{i} sos": i for i in graph["lngList"]}
        sos_stor_in = {f"{i} sos in": i for i in graph["storList"]}
        sos_stor_out = {f"{i} sos out": i for i in graph["storList"]}

        cc_tso = {i: set() for i in graph["vertexCountryAssoc"].values()}

        delta = self._dates[1] - self._dates[0]
        exp_dates = [self._dates[0] - delta] + self._dates + [self._dates[-1] + delta]

        for i, j in graph["vertexCountryAssoc"].items():
            cc_tso[j].add(i)

        for d in self._dates:
            # {страна экспортер, страна производитель, пхг} :> {tso, ugs}
            for v1, fanout in self._arcs_fan_out.items():
                for v2 in fanout:
                    edge = v1, v2
                    if v1 in graph["lngList"] and v2 in sos_lng:
                        x_0[self._f_arc[d][v1, v2]] = self._data.terminal_db[v1]["MonthData"][d]["sendOut"]
                    if v1 in sos_stor_in and v2 in graph["storList"]:
                        x_0[self._f_arc[d][v1, v2]] = self._data.storage_db[v2]["MonthData"][d]["injection"]
                    if v1 in graph["storList"] and v2 in sos_stor_out:
                        x_0[self._f_arc[d][v1, v2]] = self._data.storage_db[v1]["MonthData"][d]["withdrawal"]

            for c1, c_out in graph['exportDirections'].items():
                for c2 in c_out:
                    x_0[self._f_arc[d][c1, c2]] = self._data.export_assoc[c1][c2][d]

            for v1 in graph["prodVertexList"]:
                x_0[self._f_arc[d][v1, f"{v1} sos"]] = self._data.consumption_production_assoc["production"][v1][d]

            for v2 in graph["consumVertexList"]:
                x_0[self._f_arc[d][f"{v2} sos", v2]] = self._data.consumption_production_assoc["consumption"][v2][d]

        for d in exp_dates[:-1]:
            for v in graph["storList"]:
                x_0[self._f_ugs[v][d]] = self._data.storage_db[v]["MonthData"][d]["gasInStorage"]

        return x_0


if __name__ == "__main__":
    EGRMinCostFlowModel(
        DataParser.get_data(),
        [datetime(2019, i, 1) for i in range(1, 13)]
    )