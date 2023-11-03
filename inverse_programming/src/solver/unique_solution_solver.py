import logging

import coptpy
import numpy as np
from collections import defaultdict
import MIBLP.src.tools
import inverse_programming.src.config.config as config
from inverse_programming.src.structures import inv_instance


class UniqueSolutionSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("UniqueSolutionSolver")

    def solve(self, inst: inv_instance.InvLpInstance, x0, obj, mask=None, c_b_cons=None, big_m=1000000, eps=10e-6):
        if inst.sign != inv_instance.LpSign.Equal:
            raise ValueError("Sign should be 'equal'.")

        model = self._create_model_bounds(inst, x0, obj, c_b_cons, mask, big_m, eps)

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

    @staticmethod
    def _get_mask_k_q(inst: inv_instance.InvLpInstance, mask):
        x, y = np.where(inst.a != 0)
        d = defaultdict(set)
        for i in range(x.shape[0]):
            d[x[i]].add(y[i])

        old_mask = set(mask)
        new_mask = set()
        q = list()
        for i, j in d.items():
            if j.intersection(old_mask):
                new_mask.update(j)
                q.append(i)
        return np.array(sorted(new_mask)), len(new_mask), np.array(q)

    def _create_model_bounds(
            self,
            inst: inv_instance.InvLpInstance,
            x0,
            obj,
            mask=None,
            c_b_cons=None,  # TODO: refactor this!!!
            big_m=10e6,
            eps=10e-6
    ):
        n, m = inst.a.shape
        o = [None, inst.lower_bounds is not None, inst.upper_bounds is not None]
        mask = np.array(mask) if mask else np.arange(m)
        mask, k, q = self._get_mask_k_q(inst, mask)

        model: coptpy.Model = self._envr.createModel(name="Model")
        self.model = model
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x", lb=-coptpy.COPT.INFINITY)
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1", lb=-coptpy.COPT.INFINITY)
        y2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y2", lb=0.0) if o[1] else np.full(m, 0.0)
        # y3 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y3", lb=0.0) if o[2] else np.full(m, 0.0)

        lam1 = model.addMVar(q.shape[0], vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        lam2 = model.addMVar(k, vtype=coptpy.COPT.BINARY, nameprefix="lam2") if o[1] else np.full(m, 0.0)
        lam3 = model.addMVar(k, vtype=coptpy.COPT.BINARY, nameprefix="lam3") if o[2] else np.full(m, 0.0)

        gam = model.addMVar(q.shape[0], vtype=coptpy.COPT.BINARY, nameprefix="gam")

        true_c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c", lb=-coptpy.COPT.INFINITY) if obj[0] else inst.c
        true_l = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l", lb=-coptpy.COPT.INFINITY) if obj[1] else inst.lower_bounds
        true_u = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u", lb=-coptpy.COPT.INFINITY) if obj[2] else inst.upper_bounds

        # primal constraints
        model.addConstrs(inst.a @ x == inst.b)

        # dual constraints
        if o[2]:
            model.addConstrs(inst.a.T @ y1 + y2 - true_c >= 0)
        else:
            model.addConstrs(inst.a.T @ y1 + y2 == true_c)


        # counting
        model.addConstrs(y1[q] >= eps * lam1 - big_m * gam)
        model.addConstrs(y1[q] <= -eps * lam1 + big_m * (1 - gam))
        model.addConstrs(lam1.sum() + lam2.sum() + lam3.sum() == k)

        # for cases with boundaries
        if o[1]:
            model.addConstrs(x >= true_l)

            kkt2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")
            model.addConstrs(y2 <= kkt2 * big_m)
            model.addConstrs(x - true_l <= (1 - kkt2) * big_m)

            model.addConstrs(y2[mask] >= eps * lam2)
        if o[2]:
            model.addConstrs(x <= true_u)

            kkt3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt3")
            # model.addConstrs(y3 <= kkt3 * big_m)
            model.addConstrs(inst.a.T @ y1 + y2 - true_c <= kkt3 * big_m)
            model.addConstrs(true_u - x <= (1 - kkt3) * big_m)

            # model.addConstrs(y3[mask] >= eps * lam3)
            model.addConstrs((inst.a.T @ y1 + y2 - true_c)[mask] >= eps * lam3)

        # for objective
        sum_ = self._create_abs_constraint(x - x0, "ome_x").sum()
        if obj[0]:
            sum_ += self._create_abs_constraint(inst.c - true_c, "ome_c").sum() * obj[0]
            if c_b_cons is not None and "c" in c_b_cons:
                model.addConstrs(c_b_cons["c"][0] @ true_c == c_b_cons["c"][1])
        if obj[1]:
            sum_ += self._create_abs_constraint(inst.lower_bounds - true_l, "ome_lb").sum() * obj[1]
            if c_b_cons is not None and  "l" in c_b_cons:
                model.addConstrs(c_b_cons["l"][0] @ true_c == c_b_cons["l"][1])
        if obj[2]:
            sum_ += self._create_abs_constraint(inst.upper_bounds - true_u, "ome_ub").sum() * obj[2]
            if c_b_cons is not None and "u" in c_b_cons:
                model.addConstrs(c_b_cons["u"][0] @ true_c == c_b_cons["u"][1])

        model.setObjective(sum_, coptpy.COPT.MINIMIZE)

        self._logger.info("Model is created.")
        return model

    def _create_abs_constraint(self, x, name):
        omega = self.model.addMVar(x.shape[0], vtype=coptpy.COPT.CONTINUOUS, nameprefix=name)
        self.model.addConstrs(-omega <= x)
        self.model.addConstrs(x <= omega)
        return omega
