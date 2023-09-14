import logging

import coptpy

import MIBLP.src.tools
import inverse_programming.src.config.config as config
from inverse_programming.src.structures import inv_instance


class UniqueSoluteionSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("MIBLPSolver")

    def solve(self, inst: inv_instance.InvLpInstance, x0, big_m=1000000, eps=10e-6):
        if inst.upper_bounds is not None:
            raise NotImplementedError("Algorithm with upper bounds not implemented.")

        if not (inst.lower_bounds == 0.0).all():
            raise ValueError("Lower bounds should be zero.")

        if inst.sign != inv_instance.LpSign.Equal:
            raise ValueError("Sign should be 'equal'.")

        model = self._create_model(inst, x0, big_m, eps)
        model.solve()
        if model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Master problem is infeasible.")
            raise ValueError("Master problem should be feasible")

        x = list()
        c = list()
        for i in range(inst.a.shape[1]):
            x.append(model.getVarByName(f"x({i})").getInfo("value"))
        for i in range(inst.a.shape[1]):
            c.append(model.getVarByName(f"c({i})").getInfo("value"))

        return x, c

    def _create_model(self, inst: inv_instance.InvLpInstance, x0, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Master")
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape

        x = model.addVars(range(m), vtype=coptpy.COPT.CONTINUOUS, nameprefix="x", lb=0.0)
        y = model.addVars(range(n), vtype=coptpy.COPT.CONTINUOUS, nameprefix="y")
        c = model.addVars(range(m), vtype=coptpy.COPT.CONTINUOUS, nameprefix="c")
        ome1 = model.addVars(range(m), vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome1")
        ome2 = model.addVars(range(m), vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome2")
        psi = model.addVars(range(m), vtype=coptpy.COPT.CONTINUOUS, nameprefix="psi", lb=0.0)
        lam = model.addVars(range(n + m), vtype=coptpy.COPT.BINARY, nameprefix="lam")
        gam = model.addVars(range(n), vtype=coptpy.COPT.BINARY, nameprefix="gam")
        phi = model.addVars(range(m), vtype=coptpy.COPT.BINARY, nameprefix="phi")

        model.setObjective(
            sum(ome1[i] for i in range(m)) +
            sum(ome2[i] for i in range(m)),
            coptpy.COPT.MINIMIZE
        )

        # Ax == b
        model.addConstrs(
            sum(inst.a[i, j] * x[j] for j in range(m)) == inst.b[i]
            for i in range(n)
        )

        # ATy + psi == c
        model.addConstrs(
            sum(inst.a[i, j] * y[i] for i in range(n)) + psi[j] == c[j]
            for j in range(m)
        )

        # KKT conditions
        model.addConstrs(
            x[j] <= phi[j] * big_m
            for j in range(m)
        )

        # KKT conditions
        model.addConstrs(
            c[j] - sum(inst.a[i, j] * y[i] for i in range(n)) <= (1 - phi[j]) * big_m
            for j in range(m)
        )

        # counting
        model.addConstrs(y[i] >= lam[i + m] * eps - gam[i] * big_m for i in range(n))
        model.addConstrs(y[i] <= -lam[i + m] * eps + (1 - gam[i]) * big_m for i in range(n))
        model.addConstrs(psi[i] >= lam[i] * eps for i in range(m))
        model.addConstr(sum(lam[i] for i in range(n + m)) == m)

        # for objective
        model.addConstrs(-ome1[i] <= x[i] - x0[i] for i in range(m))
        model.addConstrs(x[i] - x0[i] <= ome1[i] for i in range(m))

        # for objective
        model.addConstrs(-ome2[i] <= c[i] - inst.c[i] for i in range(m))
        model.addConstrs(c[i] - inst.c[i] <= ome2[i] for i in range(m))

        return model



