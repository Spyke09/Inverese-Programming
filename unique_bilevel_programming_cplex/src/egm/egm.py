import itertools
import json
import logging
import typing as tp
from collections import defaultdict
from datetime import datetime

import numpy as np
from dateutil import relativedelta

from unique_bilevel_programming_cplex.src.base.common import LPNan, is_lp_nan, Sense, LPFloat
from unique_bilevel_programming_cplex.src.base.model import Model
from unique_bilevel_programming_cplex.src.base.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.base.var_expr_con import Var, LinExpr, Constraint


class EGRMinCostFlowModel:
    def __init__(
            self,
            price_lag=12,
            big_m=1e6,
            eps=1e-3,
            first_unique=False
    ):
        self._lag = price_lag
        self._big_m = big_m
        self._eps = eps
        self._first_unique = first_unique

        self._model: Model = Model()
        self._ub_model = UBModel(self._model)

        _arc = tp.Tuple[str, str]
        self._f_t_pi: tp.Dict[str, Var] = dict()
        self._f_pi_prod: tp.Dict[datetime, tp.Dict[_arc, Var]] = dict()
        self._f_arc: tp.Dict[datetime, tp.Dict[_arc, Var]] = dict()
        self._f_cons_c: tp.Dict[datetime, tp.Dict[_arc, tp.Union[LPFloat, Var]]] = dict()
        self._f_c_d: tp.Dict[str, Var] = dict()
        self._f_ugs: tp.Dict[datetime, tp.Dict[_arc, Var]] = dict()

        # верхние границы
        self._known_ub: tp.List[Constraint] = list()
        self._unknown_ub: tp.List[Constraint] = list()
        self._pi_ub: tp.Dict[str, tp.List[Constraint]] = dict()
        # ребра с ненулевыми ценами (все внутренние без фиктивных ребер)
        self._non_zero_costs: tp.Set[_arc] = set()

        self._var_obj: tp.List[Var] = list()

        self._logger = logging.getLogger("EGRMinCostFlowModel")

        self._solution = None

    def fit(self, train_data, dates):
        self._logger.info("Starting the EGM model initialization.")
        self._init_model(train_data, dates)

        self._prepare_ub_model(train_data, dates)
        self._logger.info(
            f"Initialization is finished. "
            f"Model with {len(self._model.vars)} vars, {len(self._model.constraints)} constraints."
        )

        self._ub_model.init()
        self._solution = self._ub_model.solve(first_unique=self._first_unique)

    def write_results(self, path):
        res = {"x": dict(), "u": dict(), "c": {}}
        for x_i in self._model.vars:
            res["x"][x_i.name] = self._solution[x_i]

        for x_i in self._var_obj:
            res["c"][x_i.name] = self._solution[Var(f"c_{x_i.name}")]

        for u in itertools.chain(self._known_ub + self._unknown_ub):
            assert len(u.vars) == 1
            var_ = list(u.vars)[0]
            res["u"][var_.name] = self._solution[Var(f"b_{u.name}")]

        with open(path, "w") as f:
            json.dump(res, f)

    def predict_x(self, x):
        return {x_i: self._solution[x_i] for x_i in x}

    def predict_b(self, cons):
        return {con: self._solution[Var(f"b_{con.name}")] for con in cons}

    def _init_model(self, train_data, dates):
        graph = train_data.graph_db
        m = self._model = Model()

        vertex_in = set.union(graph["prodVertexList"], graph["exporterVertexList"], graph["lngList"])

        self._arcs_fan_in = arcs_fan_in = defaultdict(set)
        self._arcs_fan_out = arcs_fan_out = defaultdict(set)
        zero_costs = set()
        for v, w in graph["arcList"]:
            if (v in graph["lngList"] or v in graph["prodVertexList"] or v in graph["storList"] or
                    w in graph["consumVertexList"] or w in graph["storList"]):
                if w in graph["consumVertexList"]:
                    s = f"{w} sos"
                    zero_costs.add((s, w))
                elif v in graph["lngList"] or v in graph["prodVertexList"]:
                    s = f"{v} sos"
                    zero_costs.add((v, s))
                elif v in graph["storList"]:
                    s = f"{v} sos out"
                    zero_costs.add((v, s))
                elif w in graph["storList"]:
                    s = f"{w} sos in"
                    zero_costs.add((s, w))
                else:
                    raise ValueError
                arcs_fan_out[v].add(s)
                arcs_fan_out[s].add(w)
                arcs_fan_in[s].add(v)
                arcs_fan_in[w].add(s)
            else:
                arcs_fan_out[v].add(w)
                arcs_fan_in[w].add(v)

        for c1, out in train_data.export_assoc.items():
            c1 = f"export {c1}"
            for c2 in out:
                arcs_fan_out[c1].add(c2)
                arcs_fan_in[c2].add(c1)

        cc_tso = defaultdict(set)
        for v, cc in graph["vertexCountryAssoc"].items():
            if v in graph["tsoList"]:
                cc_tso[cc].add(v)

        vertex_out = graph["consumVertexList"]
        pac = train_data.cp_assoc["consumption"]

        self._f_t_pi = f_t_pi = {v: Var(f"flowThetaPi_theta_pi {v}") for v in vertex_in}
        self._f_pi_prod = f_pi_prod = {d: {v: Var(f"flowPiMid_{d}_pi {v}_{v}") for v in vertex_in} for d in dates}
        self._f_arc = f_arc = {
            d: {(i, j): Var(f"flowMid_{d}_{i}_{j}") for i in arcs_fan_out for j in arcs_fan_out[i]}
            for d in dates
        }
        self._f_cons_c = f_cons_c = {
            d: {
                full_v: Var(f"flowMidC_{d}_{full_v}_c {full_v}") if is_lp_nan(pac[v][d]) else pac[v][d]
                for v, full_v in ((i.replace("consum ", ""), i) for i in vertex_out)
            }
            for d in dates
        }
        self._f_c_d = f_c_d = {v: Var(f"flowCDelta_c {v}_delta") for v in vertex_out}

        delta = relativedelta.relativedelta(months=1)
        exp_dates = [dates[0] - delta] + dates + [dates[-1] + delta]
        self._f_ugs = f_ugs = {
            di: {
                u: Var(f"flowUgs_{di}_{exp_dates[i + 1]}_{u}") for u in graph["storList"]
            } for i, di in enumerate(exp_dates[:-1])
        }

        # ограничения на потоки сверху
        self._known_ub, self._unknown_ub, self._pi_ub = self._get_upper_bound_constrs(train_data, dates)
        m.add_constrs(con for con in self._known_ub)
        m.add_constrs(con for con in self._unknown_ub)

        # балансы
        for v in vertex_in:
            m.add_constr(sum(f_pi_prod[d][v] for d in dates) - f_t_pi[v] == 0)
            for d in dates:
                m.add_constr(sum(f_arc[d][v, w] for w in arcs_fan_out[v]) - f_pi_prod[d][v] == 0)
        for v in graph["lngList"]:
            m.add_constr(sum(f_pi_prod[d][v] for d in dates) - f_t_pi[v] == 0)
            for d in dates:
                m.add_constr(sum(f_arc[d][v, w] for w in arcs_fan_out[v]) - f_pi_prod[d][v] == 0)
        for v in vertex_out:
            m.add_constr(sum(f_cons_c[d][v] for d in dates) - f_c_d[v] == 0)
            for d in dates:
                m.add_constr(sum(f_arc[d][w, v] for w in arcs_fan_in[v]) - f_cons_c[d][v] == 0)
        for i, d in enumerate(dates):
            for v in set(arcs_fan_out.keys()).intersection(arcs_fan_in.keys()):
                in_ = sum(f_arc[d][w, v] for w in arcs_fan_in[v])
                out_ = sum(f_arc[d][v, w] for w in arcs_fan_out[v])
                if v in graph["storList"]:
                    out_ += f_ugs[d][v]
                    in_ += f_ugs[exp_dates[i]][v]
                m.add_constr(in_ - out_ == 0)

        # from_A_to_B
        for d in dates:
            for c1, out in train_data.export_assoc.items():
                for c2 in out:
                    r_h = sum(
                        f_arc[d][tso_1, tso_2]
                        for tso_1 in cc_tso[c1] for tso_2 in cc_tso[c2] if tso_2 in arcs_fan_out[tso_1]
                    )
                    if r_h != 0:
                        m.add_constr(f_arc[d][f"export {c1}", c2].e - r_h == 0)
                    else:
                        f_arc[d].pop((f"export {c1}", c2))

        # естественные ограничения
        m.add_constrs(v.e >= 0 for v in m.vars)

        # целевая функция
        self._non_zero_costs = set(self._f_arc[dates[0]].keys()).difference(zero_costs)
        self._var_obj = set(self._f_arc[d][arc] for d in dates for arc in self._non_zero_costs)
        self._var_obj.update(set(v for di in self._f_ugs.values() for v in di.values()))
        m.add_obj(
            LinExpr(0) + sum(self._var_obj),
            Sense.MIN
        )

    def _prepare_ub_model(self, train_data, dates):
        ub_m = self._ub_model = UBModel(self._model, big_m=self._big_m, eps=self._eps)
        ub_m.set_x0(self.get_x_0(train_data, dates))

        self._init_b()
        self._init_c(train_data, dates)

    def _init_b(self):
        ub_m = self._ub_model
        b = ub_m.init_b_as_var(itertools.chain(self._known_ub, self._unknown_ub))
        ub_m.set_b0(self._known_ub)
        ub_m.add_constrs(bi.e >= 0 for bi in b.values())

        for cons in self._pi_ub.values():
            con_1 = cons[0]
            ub_m.add_constrs(b[con_i] - b[con_1] == 0 for con_i in cons[1:])

    def _init_c(self, data, dates):
        c = self._ub_model.init_c_as_var(self._var_obj)
        self._ub_model.add_constrs(ci.e >= 0 for ci in c.values())

        _pa = data.prices_assoc
        pa_t, pa_c = _pa["TTFG1MON Index"], _pa["CO1 Comdty"]
        graph = data.graph_db

        delta = relativedelta.relativedelta(months=1)

        def make_one(vertex_or_arc_, dates_, vars_):
            for v_ in vertex_or_arc_:
                alpha_ = {lag: Var(f"alpha_({v_})_{lag}") for lag in range(1, self._lag + 1)}
                beta_ = {lag: Var(f"beta_({v_})_{lag}") for lag in range(1, self._lag + 1)}
                for d_ in dates_:
                    w_sum_ = sum(
                        alpha_[la] * pa_t[d_ - delta * la] + beta_[la] * pa_c[d_ - delta * la]
                        for la in range(1, self._lag + 1)
                    )
                    self._ub_model.add_constr(c[vars_[d_][v_]].e == w_sum_)

        make_one(graph["storList"], itertools.chain((dates[0] - delta,), dates), self._f_ugs)

        arcs = self._non_zero_costs
        usual_arcs = set(i for i in arcs if not (i[0] in graph["tsoList"] and i[1] in graph["tsoList"]))
        make_one(usual_arcs, dates, self._f_arc)

        tso_arcs = arcs.difference(usual_arcs)
        tso_groups = dict()
        for v, w in tso_arcs:
            cv = graph["vertexCountryAssoc"][v]
            cw = graph["vertexCountryAssoc"][w]
            if (cv, cw) not in tso_groups:
                tso_groups[cv, cw] = set()
            tso_groups[cv, cw].add((v, w))

        for cc, group in tso_groups.items():
            alpha = {lag: Var(f"alpha_({cc})_{lag}") for lag in range(1, self._lag + 1)}
            beta = {lag: Var(f"beta_({cc})_{lag}") for lag in range(1, self._lag + 1)}
            for arc in group:
                for d in dates:
                    w_sum = sum(
                        alpha[la] * pa_t[d - delta * la] + beta[la] * pa_c[d - delta * la]
                        for la in range(1, self._lag + 1)
                    )
                    self._ub_model.add_constr(w_sum == c[self._f_arc[d][arc]])

    def _get_upper_bound_constrs(self, train_data, dates) -> tp.Tuple[
        tp.List[Constraint],tp.List[Constraint], tp.Dict[str, tp.List[Constraint]]
    ]:
        graph = train_data.graph_db

        known_ub = []
        unknown_ub = []
        pi_ub = defaultdict(list)

        vertex_in = set.union(graph["prodVertexList"], graph["exporterVertexList"], graph["lngList"])

        delta = relativedelta.relativedelta(months=1)
        exp_dates = [dates[0] - delta] + dates + [dates[-1] + delta]

        sos_lng = {f"{i} sos": i for i in graph["lngList"]}
        sos_stor_in = {f"{i} sos in": i for i in graph["storList"]}
        sos_stor_out = {f"{i} sos out": i for i in graph["storList"]}

        # супер-вершина -> pi
        for v in vertex_in:
            unknown_ub.append(self._f_t_pi[v].e <= np.nan)

        for d in dates:
            # {страна экспортер, страна производитель, пхг} :> {tso, ugs}
            for v in vertex_in:
                con = (self._f_pi_prod[d][v].e <= np.nan)
                unknown_ub.append(con)
                pi_ub[v].append(con)
            for v1, fanout in self._arcs_fan_out.items():
                for v2 in fanout:
                    edge = v1, v2
                    cap = LPNan
                    if v1 in sos_lng and v2 in graph["tsoList"]:
                        cap = graph["arcCapTimeAssoc"][d][sos_lng[v1], v2]
                    if v1 in graph["lngList"] and v2 in sos_lng:
                        cap = train_data.terminal_db[v1]["MonthData"][d]["dtrs"]
                    if v1 in graph["tsoList"] and v2 in graph["tsoList"]:
                        cap = graph["arcCapTimeAssoc"][d][edge]
                    if v1 in sos_stor_in and v2 in graph["storList"]:
                        cap = train_data.storage_db[v2]["MonthData"][d]["injectionCapacity"]
                    if v1 in graph["storList"] and v2 in sos_stor_out:
                        cap = train_data.storage_db[v1]["MonthData"][d]["withdrawalCapacity"]
                    (unknown_ub if is_lp_nan(cap) else known_ub).append(self._f_arc[d][edge].e <= cap)

            for v in graph["consumVertexList"]:
                if isinstance(self._f_cons_c[d][v], Var):
                    unknown_ub.append(self._f_cons_c[d][v].e <= LPNan)

        for v in graph["consumVertexList"]:
            unknown_ub.append(self._f_c_d[v].e <= LPNan)

        for d in exp_dates[:-1]:
            for v in graph["storList"]:
                cap = train_data.storage_db[v]["MonthData"][d]["workingGasVolume"]
                (unknown_ub if is_lp_nan(cap) else known_ub).append(self._f_ugs[d][v].e <= cap)

        return known_ub, unknown_ub, pi_ub

    def get_x_0(self, data, dates):
        graph = data.graph_db

        x_0 = dict()

        sos_lng = {f"{i} sos": i for i in graph["lngList"]}
        sos_stor_in = {f"{i} sos in": i for i in graph["storList"]}
        sos_stor_out = {f"{i} sos out": i for i in graph["storList"]}

        delta = relativedelta.relativedelta(months=1)
        exp_dates = [dates[0] - delta] + dates

        for d in dates:
            # {страна экспортер, страна производитель, пхг} :> {tso, ugs}
            for v1, fanout in self._arcs_fan_out.items():
                for v2 in fanout:
                    edge = v1, v2
                    if v1 in graph["lngList"] and v2 in sos_lng:
                        if not is_lp_nan(x0 := data.terminal_db[v1]["MonthData"][d]["sendOut"]):
                            x_0[self._f_arc[d][edge]] = x0
                    if v1 in sos_stor_in and v2 in graph["storList"]:
                        if not is_lp_nan(x0 := data.storage_db[v2]["MonthData"][d]["injection"]):
                            x_0[self._f_arc[d][edge]] = x0
                    if v1 in graph["storList"] and v2 in sos_stor_out:
                        if not is_lp_nan(x0 := data.storage_db[v1]["MonthData"][d]["withdrawal"]):
                            x_0[self._f_arc[d][edge]] = x0

            for c1, out in data.export_assoc.items():
                for c2 in out:
                    if not is_lp_nan(x0 := data.export_assoc[c1][c2][d]):
                        if (f"export {c1}", c2) in self._f_arc[d]:
                            x_0[self._f_arc[d][f"export {c1}", c2]] = x0

            for v1 in graph["prodVertexList"]:
                v2 = v1.replace("prod ", "")
                if not is_lp_nan(x0 := data.cp_assoc["production"][v2][d]):
                    x_0[self._f_arc[d][v1, f"{v1} sos"]] = x0

            # consumption is const
            # for v2 in graph["consumVertexList"]:
            #     v1 = v2.replace("consum ", "")
            #     if not is_lp_nan(x0 := data.cp_assoc["consumption"][v1][d]):
            #         x_0[self._f_arc[d][f"{v2} sos", v2]] = x0

        for d in exp_dates:
            for v in graph["storList"]:
                if not is_lp_nan(x0 := data.storage_db[v]["MonthData"][d]["gasInStorage"]):
                    x_0[self._f_ugs[d][v]] = x0

        return x_0

    @property
    def known_ub(self):
        return self._known_ub