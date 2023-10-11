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

        model = self._create_model_bounds(inst, x0, obj, mask, big_m, eps)

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
        o = [None, inst.lower_bounds is not None, inst.upper_bounds is not None]
        model: coptpy.Model = self._envr.createModel(name="Model")
        self.model = model
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape
        k = sum(mask)

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x")
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1")
        y2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y2", lb=0.0) if o[1] else np.full(m, 0.0)
        y3 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y3", lb=0.0) if o[2] else np.full(m, 0.0)

        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        lam3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam3") if o[1] else np.full(m, 0.0)
        lam4 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam4") if o[2] else np.full(m, 0.0)
        kkt1 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt1")

        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")

        true_c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c") if obj[0] else inst.c
        true_l = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l") if obj[1] else inst.lower_bounds
        true_u = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u") if obj[2] else inst.upper_bounds

        # primal constraints
        model.addConstrs(inst.a @ x == inst.b)

        # dual constraints
        model.addConstrs(true_c - inst.a.T @ y1 - y2 + y3 >= 0)

        # kkt
        model.addConstrs(true_c - inst.a.T @ y1 - y2 + y3 <= kkt1 * big_m)
        model.addConstrs(x <= (1 - kkt1) * big_m)

        # counting
        model.addConstrs(y1 >= eps * lam1 - big_m * gam)  # !!!
        model.addConstrs(y1 <= -eps * lam1 + big_m * (1 - gam))  # !!!
        model.addConstrs(true_c - inst.a.T @ y1 - y2 + y3 >= eps * lam2)  # !!!
        model.addConstrs(lam1.sum() + lam2.sum() + lam3.sum() + lam4.sum() == m)

        # for cases with boundaries
        if o[1]:
            model.addConstrs(x >= true_l)

            kkt2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")
            model.addConstrs(y2 <= kkt2 * big_m)
            model.addConstrs(x - true_l <= (1 - kkt2) * big_m)

            model.addConstrs(y2 >= eps * lam3)  # !!!
        if o[2]:
            model.addConstrs(x <= true_u)

            kkt3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt3")
            model.addConstrs(y3 <= kkt3 * big_m)
            model.addConstrs(true_u - x <= (1 - kkt3) * big_m)

            model.addConstrs(y3 >= eps * lam4)  # !!!

        # for objective
        sum_ = self._create_abs_constraint(x - x0, "ome_x").sum()
        if obj[0]:
            sum_ += self._create_abs_constraint(inst.c - true_c, "ome_c").sum() * obj[0]
        if obj[1]:
            sum_ += self._create_abs_constraint(inst.lower_bounds - true_l, "ome_lb").sum() * obj[1]
        if obj[2]:
            sum_ += self._create_abs_constraint(inst.upper_bounds - true_u, "ome_ub").sum() * obj[2]

        model.setObjective(sum_, coptpy.COPT.MINIMIZE)

        self._logger.info("Model is created.")
        return model

    def _create_abs_constraint(self, x, name):
        omega = self.model.addMVar(x.shape[0], vtype=coptpy.COPT.CONTINUOUS, nameprefix=name)
        self.model.addConstrs(-omega <= x)
        self.model.addConstrs(x <= omega)
        return omega
