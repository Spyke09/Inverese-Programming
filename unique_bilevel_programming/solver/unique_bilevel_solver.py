import logging

import coptpy

from unique_bilevel_programming import config
from unique_bilevel_programming.structures.unique_bilevel_instance import UBInstance


class UniqueBilevelSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("UniqueBilevelSolver")

    def solve(
            self,
            inst: UBInstance,
            weights,
            big_m=1000000,
            eps=10e-6
    ):
        model = self._create_model(inst, weights, big_m, eps)

        model.solve()
        if model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Master problem is infeasible.")
            raise ValueError("Master problem should be feasible")

    def _create_abs_constraint(self, x, name):
        omega = self.model.addMVar(x.shape[0], vtype=coptpy.COPT.CONTINUOUS, nameprefix=name)
        self.model.addConstrs(-omega <= x)
        self.model.addConstrs(x <= omega)
        return omega

    def _create_model(self, inst: UBInstance, weights, big_m, eps):
        n, m = inst.A.shape
        model: coptpy.Model = self._envr.createModel(name="UniqueBilevelModel")
        self.model = model
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x", lb=-coptpy.COPT.INFINITY)
        y1 = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y1", lb=-coptpy.COPT.INFINITY)
        y2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y2", lb=0.0)
        # y3 = inst.A.T @ y1 + y2 - c >= 0

        l1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="l1")
        l2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="l2")
        l3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="l3")

        g = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="g")

        c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c", lb=-coptpy.COPT.INFINITY)
        b = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="b", lb=-coptpy.COPT.INFINITY)
        l = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l", lb=-coptpy.COPT.INFINITY)
        u = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u", lb=-coptpy.COPT.INFINITY)

        # primal constraints
        model.addConstrs(inst.A @ x == b)
        model.addConstrs(inst.A.T @ y1 + y2 - c >= 0)

        # counting
        model.addConstrs(y1 >= eps * l1 - big_m * g)
        model.addConstrs(y1 <= -eps * l1 + big_m * (1 - g))
        model.addConstrs(l1.sum() + l2.sum() + l3.sum() == m)

        # lower bounds constraints
        model.addConstrs(x >= l)

        kkt2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")
        model.addConstrs(y2 <= kkt2 * big_m)
        model.addConstrs(x - l <= (1 - kkt2) * big_m)

        model.addConstrs(y2 >= eps * l2)

        # upper bounds constraints
        model.addConstrs(x <= u)

        kkt3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt3")
        model.addConstrs(inst.A.T @ y1 + y2 - c <= kkt3 * big_m)
        model.addConstrs(u - x <= (1 - kkt3) * big_m)

        model.addConstrs((inst.A.T @ y1 + y2 - c) >= eps * l3)

        # Constraints for c, b, l, u
        model.addConstrs(inst.C @ c == inst.c_hat)
        model.addConstrs(inst.B @ b == inst.b_hat)
        model.addConstrs(inst.L @ l == inst.l_hat)
        model.addConstrs(inst.U @ u == inst.u_hat)

        # for objective
        weighed_obj_sum_ = self._create_abs_constraint(x - inst.x0, "ome_x").sum() * weights["x"]
        weighed_obj_sum_ += self._create_abs_constraint(b - inst.b0, "ome_b").sum() * weights["b"]
        weighed_obj_sum_ += self._create_abs_constraint(c - inst.c0, "ome_c").sum() * weights["c"]
        weighed_obj_sum_ += self._create_abs_constraint(l - inst.l0, "ome_l").sum() * weights["l"]
        weighed_obj_sum_ += self._create_abs_constraint(u - inst.u0, "ome_u").sum() * weights["u"]

        model.setObjective(weighed_obj_sum_, coptpy.COPT.MINIMIZE)

        return model
