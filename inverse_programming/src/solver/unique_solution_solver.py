import logging

import coptpy
import numpy as np
import inverse_programming.src.config.config as config
from inverse_programming.src.structures import inv_instance


class UniqueSolutionSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("UniqueSolutionSolver")

    def solve(self, inst: inv_instance.InvLpInstance, x0, obj, mask, big_m=1000000, eps=10e-6):
        if inst.sign != inv_instance.LpSign.Equal:
            raise ValueError("Sign should be 'equal'.")

        if inst.upper_bounds is None and inst.lower_bounds is None:
            obj[1] = 0
            inst = inv_instance.InvLpInstance(inst.a, inst.b, inst.c, inst.sign, np.full(inst.a.shape[1], 0))
            model = self._create_model_lower_bounds(inst, x0, obj, mask, big_m, eps)
        elif inst.upper_bounds is not None and inst.lower_bounds is not None:
            model = self._create_model_bounds(inst, x0, obj, mask, big_m, eps)
        elif inst.upper_bounds is None and inst.lower_bounds is not None:
            model = self._create_model_lower_bounds(inst, x0, obj, mask, big_m, eps)
        else:
            model = self._create_model_upper_bounds(inst, x0, obj, mask, big_m, eps)

        model.solve()
        if model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Master problem is infeasible.")
            raise ValueError("Master problem should be feasible")

    def _vars_number(self, name):
        return sum([i.getName().split("(")[0] == name for i in self.model.getVars().getAll()])

    def get_values_by_names(self, names):
        return {name: self.get_values_by_name(name) for name in names if self._vars_number(name) > 0}

    def get_values_by_name(self, name):
        n = self._vars_number(name)
        return np.array([self.model.getVarByName(f"{name}({i})").getInfo("value") for i in range(n)])

    def _create_model_bounds(self, inst: inv_instance.InvLpInstance, x0, obj, mask, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Model")
        self.model = model
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape
        k = sum(mask)

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x")
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1")
        y2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y2", lb=0.0)
        y3 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y3", lb=0.0)
        ome0 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome0")
        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        lam3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam3")
        lam4 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam4")
        kkt1 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt1")
        kkt2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")
        kkt3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")

        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")

        true_c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c") if obj[0] else inst.c
        true_l = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l") if obj[1] else inst.lower_bounds
        true_u = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u") if obj[2] else inst.upper_bounds

        # primal constraints
        model.addConstrs(inst.a @ x == inst.b)
        model.addConstrs(x >= true_l)
        model.addConstrs(x <= true_u)

        # dual constraints
        model.addConstrs(true_c - inst.a.T @ y1 - y2 + y3 >= 0)

        # kkt
        model.addConstrs(true_c - inst.a.T @ y1 - y2 + y3 <= kkt1 * big_m)
        model.addConstrs(x <= (1 - kkt1) * big_m)
        model.addConstrs(y2 <= kkt2 * big_m)
        model.addConstrs(x - true_l <= (1 - kkt2) * big_m)
        model.addConstrs(y3 <= kkt3 * big_m)
        model.addConstrs(true_u - x <= (1 - kkt3) * big_m)

        # counting
        model.addConstrs(y1 >= eps * lam1 - big_m * gam)
        model.addConstrs(y1 <= -eps * lam1 + big_m * (1 - gam))
        model.addConstrs(true_c - inst.a.T @ y1 - y2 + y3 >= eps * lam2)
        model.addConstrs(y2 >= eps * lam3)
        model.addConstrs(y3 >= eps * lam4)

        model.addConstrs(lam1.sum() + lam2.sum() + lam3.sum() + lam4.sum() == m)

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

        self._logger.info("Model b is created.")
        return model

    def _create_model_lower_bounds(self, inst: inv_instance.InvLpInstance, x0, obj, mask, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Model")
        self.model = model
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape
        k = sum(mask)

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x")
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1")
        y2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y2", lb=0.0)
        ome0 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome0")
        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        lam3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam3")
        kkt1 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt1")
        kkt2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")
        kkt3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")

        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")

        true_c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c") if obj[0] else inst.c
        true_l = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l") if obj[1] else inst.lower_bounds
        true_u = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u") if obj[2] else inst.upper_bounds

        # primal constraints
        model.addConstrs(inst.a @ x == inst.b)
        model.addConstrs(x >= true_l)
        model.addConstrs(x <= true_u)

        # dual constraints
        model.addConstrs(true_c - inst.a.T @ y1 - y2 >= 0)

        # kkt
        model.addConstrs(true_c - inst.a.T @ y1 - y2 <= kkt1 * big_m)
        model.addConstrs(x <= (1 - kkt1) * big_m)
        model.addConstrs(y2 <= kkt2 * big_m)
        model.addConstrs(x - true_l <= (1 - kkt2) * big_m)

        # counting
        model.addConstrs(y1 >= eps * lam1 - big_m * gam)
        model.addConstrs(y1 <= -eps * lam1 + big_m * (1 - gam))
        model.addConstrs(true_c - inst.a.T @ y1 - y2 >= eps * lam2)
        model.addConstrs(y2 >= eps * lam3)

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

        model.setObjective(sum_, coptpy.COPT.MINIMIZE)

        self._logger.info("Model lb is created.")
        return model

    def _create_model_upper_bounds(self, inst: inv_instance.InvLpInstance, x0, obj, mask, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Model")
        self.model = model
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape
        k = sum(mask)

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x")
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1")
        y3 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y3", lb=0.0)
        ome0 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome0")
        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        lam4 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam4")
        kkt1 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt1")
        kkt3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")

        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")

        true_c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c") if obj[0] else inst.c
        true_l = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l") if obj[1] else inst.lower_bounds
        true_u = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u") if obj[2] else inst.upper_bounds

        # primal constraints
        model.addConstrs(inst.a @ x == inst.b)
        model.addConstrs(x >= true_l)
        model.addConstrs(x <= true_u)

        # dual constraints
        model.addConstrs(true_c - inst.a.T @ y1 + y3 >= 0)

        # kkt
        model.addConstrs(true_c - inst.a.T @ y1 + y3 <= kkt1 * big_m)
        model.addConstrs(x <= (1 - kkt1) * big_m)
        model.addConstrs(y3 <= kkt3 * big_m)
        model.addConstrs(true_u - x <= (1 - kkt3) * big_m)

        # counting
        model.addConstrs(y1 >= eps * lam1 - big_m * gam)
        model.addConstrs(y1 <= -eps * lam1 + big_m * (1 - gam))
        model.addConstrs(true_c - inst.a.T @ y1 + y3 >= eps * lam2)
        model.addConstrs(y3 >= eps * lam4)

        model.addConstrs(lam1.sum() + lam2.sum() + lam4.sum() == m)

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

        self._logger.info("Model ub is created.")
        return model
