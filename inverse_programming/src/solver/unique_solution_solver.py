import logging

import coptpy
import numpy as np
import inverse_programming.src.config.config as config
from inverse_programming.src.structures import inv_instance


class UniqueSolutionSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("UniqueSolutionSolver")

    def solve(self, inst: inv_instance.InvLpInstance, x0, obj, big_m=1000000, eps=10e-6):
        if inst.sign != inv_instance.LpSign.Equal:
            raise ValueError("Sign should be 'equal'.")

        if inst.upper_bounds is None and inst.lower_bounds is None:
            obj[1] = 0
            inst = inv_instance.InvLpInstance(inst.a, inst.b, inst.c, inst.sign, np.full(inst.a.shape[1], 0))
            model = self._create_model_lower_bounds(inst, x0, obj, big_m, eps)
        elif inst.upper_bounds is not None and inst.lower_bounds is not None:
            model = self._create_model_bounds(inst, x0, obj, big_m, eps)
        elif inst.upper_bounds is None and inst.lower_bounds is not None:
            model = self._create_model_lower_bounds(inst, x0, obj, big_m, eps)
        else:
            model = self._create_model_upper_bounds(inst, x0, obj, big_m, eps)

        model.solve()
        if model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Master problem is infeasible.")
            raise ValueError("Master problem should be feasible")

        res = dict()
        x = list()
        for i in range(inst.a.shape[1]):
            x.append(model.getVarByName(f"x({i})").getInfo("value"))
        res["x"] = np.array(x)

        y1 = list()
        for i in range(inst.a.shape[0]):
            y1.append(model.getVarByName(f"y1({i})").getInfo("value"))
        res["y1"] = np.array(y1)

        if obj[0]:
            c = list()
            for i in range(inst.a.shape[1]):
                c.append(model.getVarByName(f"c({i})").getInfo("value"))
            res["c"] = np.array(c)

        if inst.lower_bounds is not None and obj[1]:
            l = list()
            for i in range(inst.a.shape[1]):
                l.append(model.getVarByName(f"l({i})").getInfo("value"))
            res["l"] = np.array(l)

        if inst.upper_bounds is not None and obj[2]:
            u = list()
            for i in range(inst.a.shape[1]):
                u.append(model.getVarByName(f"u({i})").getInfo("value"))
            res["u"] = np.array(u)

        if inst.upper_bounds is not None and inst.lower_bounds is not None:
            y2 = list()
            for i in range(inst.a.shape[1]):
                y2.append(model.getVarByName(f"y2({i})").getInfo("value"))
            res["y2"] = np.array(y2)
            # inst.a.T @ y1 + y2 - true_c
            res["y3"] = inst.a.T @ y1 + res["y2"] - (res["c"] if "c" in res else inst.c)
        else:
            res["y2"] = inst.a.T @ y1 - (res["c"] if "c" in res else inst.c)

        return res

    def _create_model_bounds(self, inst: inv_instance.InvLpInstance, x0, obj, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Model")
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x")
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1")
        y2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y2", lb=0.0)
        ome0 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome0")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        lam3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam3")
        kkt1 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt1")
        kkt2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")
        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")

        true_c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c") if obj[0] else inst.c
        true_l = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l") if obj[1] else inst.lower_bounds
        true_u = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u") if obj[2] else inst.upper_bounds

        # primal constraints
        model.addConstrs(inst.a @ x == inst.b)
        model.addConstrs(x >= true_l)
        model.addConstrs(x <= true_u)

        # dual constraints
        model.addConstrs(inst.a.T @ y1 + y2 - true_c >= 0)

        # kkt
        model.addConstrs(y2 <= kkt1 * big_m)
        model.addConstrs(x - true_l <= (1 - kkt1) * big_m)
        model.addConstrs(inst.a.T @ y1 + y2 - true_c <= kkt2 * big_m)
        model.addConstrs(-x + true_u <= (1 - kkt2) * big_m)

        # counting
        model.addConstrs(y1 >= eps * lam1 - big_m * gam)
        model.addConstrs(y1 <= -eps * lam1 + big_m * (1 - gam))
        model.addConstrs(y2 >= eps * lam2)
        model.addConstrs(inst.a.T @ y1 + y2 - true_c >= eps * lam3)
        model.addConstrs(lam1.sum() + lam2.sum() + lam3.sum() == m)

        # for objective
        model.addConstrs(-ome0 <= x - x0)
        model.addConstrs(x - x0 <= ome0)
        sum_ = ome0.sum()
        if obj[0]:
            ome1 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome1")
            model.addConstrs(-ome1 <= inst.c - true_c)
            model.addConstrs(inst.c - true_c <= ome1)
            sum_ += ome1.sum() * obj[0]

        if obj[1]:
            ome2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome2")
            model.addConstrs(-ome2 <= inst.lower_bounds - true_l)
            model.addConstrs(inst.lower_bounds - true_l <= ome2)
            sum_ += ome2.sum() * obj[1]

        if obj[2]:
            ome3 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome3")
            model.addConstrs(-ome3 <= inst.upper_bounds - true_u)
            model.addConstrs(inst.upper_bounds - true_u <= ome3)
            sum_ += ome3.sum() * obj[2]

        model.setObjective(sum_, coptpy.COPT.MINIMIZE)

        self._logger.info("Model is created.")
        return model

    def _create_model_lower_bounds(self, inst: inv_instance.InvLpInstance, x0, obj, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Model")
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x")
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1")
        ome0 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome0")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        kkt1 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt1")
        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")

        true_c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c") if obj[0] else inst.c
        true_l = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l") if obj[1] else inst.lower_bounds

        # primal constraints
        model.addConstrs(inst.a @ x == inst.b)
        model.addConstrs(x >= true_l)

        # dual constraints
        model.addConstrs(-inst.a.T @ y1 + true_c >= 0)

        # kkt
        model.addConstrs(x - true_l <= kkt1 * big_m)
        model.addConstrs(-inst.a.T @ y1 + true_c <= (1 - kkt1) * big_m)

        # counting
        model.addConstrs(y1 >= eps * lam1 - big_m * gam)
        model.addConstrs(y1 <= -eps * lam1 + big_m * (1 - gam))
        model.addConstrs(-inst.a.T @ y1 + true_c >= eps * lam2)
        model.addConstrs(lam1.sum() + lam2.sum() == m)

        # for objective
        model.addConstrs(-ome0 <= x - x0)
        model.addConstrs(x - x0 <= ome0)
        sum_ = ome0.sum()
        if obj[0]:
            ome1 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome1")
            model.addConstrs(-ome1 <= inst.c - true_c)
            model.addConstrs(inst.c - true_c <= ome1)
            sum_ += ome1.sum() * obj[0]

        if obj[1]:
            ome2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome2")
            model.addConstrs(-ome2 <= inst.lower_bounds - true_l)
            model.addConstrs(inst.lower_bounds - true_l <= ome2)
            sum_ += ome2.sum() * obj[1]

        model.setObjective(sum_, coptpy.COPT.MINIMIZE)

        self._logger.info("Model is created.")
        return model

    def _create_model_upper_bounds(self, inst: inv_instance.InvLpInstance, x0, obj, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Model")
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x")
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1")
        ome0 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome0")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        kkt1 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt1")
        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")

        true_c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c") if obj[0] else inst.c
        true_u = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u") if obj[2] else inst.upper_bounds

        # primal constraints
        model.addConstrs(inst.a @ x == inst.b)
        model.addConstrs(x <= true_u)

        # dual constraints
        model.addConstrs(inst.a.T @ y1 - true_c >= 0)

        # kkt
        model.addConstrs(-x + true_u <= kkt1 * big_m)
        model.addConstrs(inst.a.T @ y1 - true_c <= (1 - kkt1) * big_m)

        # counting
        model.addConstrs(y1 >= eps * lam1 - big_m * gam)
        model.addConstrs(y1 <= -eps * lam1 + big_m * (1 - gam))
        model.addConstrs(inst.a.T @ y1 - true_c >= eps * lam2)
        model.addConstrs(lam1.sum() + lam2.sum() == m)

        # for objective
        model.addConstrs(-ome0 <= x - x0)
        model.addConstrs(x - x0 <= ome0)
        sum_ = ome0.sum()
        if obj[0]:
            ome1 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome1")
            model.addConstrs(-ome1 <= inst.c - true_c)
            model.addConstrs(inst.c - true_c <= ome1)
            sum_ += ome1.sum() * obj[0]

        if obj[2]:
            ome3 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome3")
            model.addConstrs(-ome3 <= inst.upper_bounds - true_u)
            model.addConstrs(inst.upper_bounds - true_u <= ome3)
            sum_ += ome3.sum() * obj[2]

        model.setObjective(sum_, coptpy.COPT.MINIMIZE)

        self._logger.info("Model is created.")
        return model