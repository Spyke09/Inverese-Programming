import itertools
import logging
import typing as tp
from datetime import datetime

from unique_bilevel_programming_cplex.src.base.common import LPNan, Sense
from unique_bilevel_programming_cplex.src.base.model import Model
from unique_bilevel_programming_cplex.src.base.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.base.var_expr_con import Var, LinExpr, Constraint
from unique_bilevel_programming_cplex.src.egm.data_parser import EGMData


class EGRMinCostFlowModel:
    def __init__(self, data: EGMData, dates: tp.List[datetime], lag=12):
        self._data: EGMData = data
        self._dates: tp.List[datetime] = dates
        self._lag = lag

        self._model: Model = Model()
        self._ub_model = UBModel(self._model)

        _arc = tp.Tuple[str, str]
        self._f_pi_prod: tp.Dict[datetime, tp.Dict[_arc, Var]] = dict()
        self._f_arc: tp.Dict[datetime, tp.Dict[_arc, Var]] = dict()
        self._f_cons_c: tp.Dict[datetime, tp.Dict[_arc, Var]] = dict()
        self._f_ugs: tp.Dict[datetime, tp.Dict[_arc, Var]] = dict()

        # верхние границы
        self._known_ub: tp.List[Constraint] = list()
        self._unknown_ub: tp.List[Constraint] = list()

        self._var_obj: tp.List[Var] = list()

        self._logger = logging.getLogger("EGRMinCostFlowModel")

    def setup(self):
        self._logger.info("Starting the Min-cost-flow model initialization")
        self._init_model()
        self._logger.info(
            f"Initialization is finished. Model with {len(self._model.vars)} vars, {len(self._model.constraints)} constraints"
        )
        self._logger.info("Starting the UB-Inv model initialization")
        self._init_ub_model()
        self._logger.info("UB-Inv model initialization is finished")

    def solve(self):
        self._ub_model.init()
        return self._ub_model.solve()

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

        f_T_pi = {v: Var(f"flow_Theta_pi_({v})") for v in vertex_in}
        self._f_pi_prod = f_pi_prod = {d: {v: Var(f"flow_({d})_pi_({v})") for v in vertex_in} for d in self._dates}
        self._f_arc = f_arc = {
            d: {(i, j): Var(f"flow_({d})_({i},{j})") for i in arcs_fan_out for j in arcs_fan_out[i]}
            for d in self._dates
        }
        self._f_cons_c = f_cons_c = {d: {v: Var(f"flow_({d})_({v})_c") for v in vertex_out} for d in self._dates}
        f_c_D = {v: Var(f"flow_c_D_({v})") for v in vertex_out}

        delta = self._dates[1] - self._dates[0]
        exp_dates = [self._dates[0] - delta] + self._dates + [self._dates[-1] + delta]
        self._f_ugs = f_ugs = {
            di: {
                u: Var(f"flow_ugs_({u})_({di})_({exp_dates[i + 1]})") for u in graph["storList"]
            } for i, di in enumerate(exp_dates[:-1])
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
                    f_ugs[d][v].e <= cap
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
                    out_ += f_ugs[d][v]
                    in_ += f_ugs[exp_dates[i]][v]
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

        # целевая функция
        self._var_obj = [v for v in m.vars if v not in f_T_pi and v not in f_c_D]
        m.add_obj(
            LinExpr(0) + sum(self._var_obj),
            Sense.MIN
        )

    def _init_ub_model(self):
        ub_m = self._ub_model = UBModel(self._model, big_m=1e13, eps=1e-4)
        ub_m.set_x0(self._get_x_0())

        # b = ub_m.init_b_as_var()
        # ub_m.set_b0(self._known_ub)
        b = ub_m.init_b_as_var(self._unknown_ub)
        ub_m.add_constrs(bi.e >= 0 for bi in b.values())

        self._init_price_constrs()

    def _init_price_constrs(self):
        pa = self._data.prices_assoc
        vertex_in = self._f_pi_prod[self._dates[0]].keys()
        c = self._ub_model.init_c_as_var(self._var_obj)
        self._ub_model.add_constrs(ci.e >= 0 for ci in c.values())

        # alpha = {v: Var(f"alpha_({v.name})") for v in self._var_obj}
        # beta = {v: Var(f"beta_({v.name})") for v in self._var_obj}
        # for v in vertex_in:
        #     alpha = {v: Var(f"alpha_({v.name})") for v in self._var_obj}
        #     beta = {v: Var(f"beta_({v.name})") for v in self._var_obj}
        #     for d in self._f_pi_prod:
        #
        #
        # for v in itertools.chain(, self._f_cons_c[d].values(), self._f_arc[d].values()):
        #     self._ub_model.add_constr(
        #         alpha[v] * pa["TTFG1MON Index"][d] + beta[v] * pa["CO1 Comdty"][d] == c[v]
        #     )
        #
        # for d in itertools.chain([self._dates[0] - (self._dates[1] - self._dates[0])], self._dates):
        #     for v in self._f_ugs[d].values():
        #         self._ub_model.add_constr(alpha[v] * pa["TTFG1MON Index"][d] + beta[v] * pa["CO1 Comdty"][d] == c[v])

    def _get_x_0(self):
        graph = self._data.graph_db

        x_0 = dict()

        sos_lng = {f"{i} sos": i for i in graph["lngList"]}
        sos_stor_in = {f"{i} sos in": i for i in graph["storList"]}
        sos_stor_out = {f"{i} sos out": i for i in graph["storList"]}

        cc_tso = {i: set() for i in graph["vertexCountryAssoc"].values()}

        delta = self._dates[1] - self._dates[0]
        exp_dates = [self._dates[0] - delta] + self._dates

        for i, j in graph["vertexCountryAssoc"].items():
            cc_tso[j].add(i)

        for d in self._dates:
            # {страна экспортер, страна производитель, пхг} :> {tso, ugs}
            for v1, fanout in self._arcs_fan_out.items():
                for v2 in fanout:
                    edge = v1, v2
                    if v1 in graph["lngList"] and v2 in sos_lng:
                        if (x0 := self._data.terminal_db[v1]["MonthData"][d]["sendOut"]) is not LPNan:
                            x_0[self._f_arc[d][edge]] = x0
                    if v1 in sos_stor_in and v2 in graph["storList"]:
                        if (x0 := self._data.storage_db[v2]["MonthData"][d]["injection"]) is not LPNan:
                            x_0[self._f_arc[d][edge]] = x0
                    if v1 in graph["storList"] and v2 in sos_stor_out:
                        if (x0 := self._data.storage_db[v1]["MonthData"][d]["withdrawal"]) is not LPNan:
                            x_0[self._f_arc[d][edge]] = x0

            for c1, c_out in graph['exportDirections'].items():
                for c2 in c_out:
                    if (x0 := self._data.export_assoc[c1][c2][d]) is not LPNan:
                        x_0[self._f_arc[d][f"export {c1}", c2]] = x0

            for v1 in graph["prodVertexList"]:
                v2 = v1.replace("prod ", "")
                if (x0 := self._data.consumption_production_assoc["production"][v2][d]) is not LPNan:
                    x_0[self._f_arc[d][v1, f"{v1} sos"]] = x0

            for v2 in graph["consumVertexList"]:
                v1 = v2.replace("consum ", "")
                if (x0 := self._data.consumption_production_assoc["consumption"][v1][d]) is not LPNan:
                    x_0[self._f_arc[d][f"{v2} sos", v2]] = x0

        for d in exp_dates:
            for v in graph["storList"]:
                if (x0 := self._data.storage_db[v]["MonthData"][d]["gasInStorage"]) is not LPNan:
                    x_0[self._f_ugs[d][v]] = x0

        return x_0
