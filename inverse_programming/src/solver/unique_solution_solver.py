import logging

import coptpy

import inverse_programming.src.config.config as config
from inverse_programming.src.structures import inv_instance
from MIBLP.src.tools import model_repr


class UniqueSolutionSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("UniqueSolutionSolver")

    def solve(self, inst: inv_instance.InvLpInstance, x0, big_m=1000000, eps=10e-6):
        if not (inst.lower_bounds == 0.0).all():
            raise ValueError("Lower bounds should be zero.")

        if inst.sign != inv_instance.LpSign.Equal:
            raise ValueError("Sign should be 'equal'.")

        if inst.upper_bounds is not None:
            model = self._create_model_upper_bounds(inst, x0, big_m, eps)
        else:
            model = self._create_model_no_bounds(inst, x0, big_m, eps)

        model.solve()
        if model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Master problem is infeasible.")
            raise ValueError("Master problem should be feasible")

        x = list()
        c = list()
        y = list()
        phi = list()
        lam = list()
        for i in range(inst.a.shape[1]):
            x.append(model.getVarByName(f"x({i})").getInfo("value"))
        for i in range(inst.a.shape[1]):
            c.append(model.getVarByName(f"c({i})").getInfo("value"))
        for i in range(inst.a.shape[0]):
            y.append(model.getVarByName(f"y({i})").getInfo("value"))
        for i in range(inst.a.shape[1]):
            phi.append(model.getVarByName(f"phi({i})").getInfo("value"))
        for i in range(inst.a.shape[0]):
            lam.append(model.getVarByName(f"lam1({i})").getInfo("value"))
        for i in range(inst.a.shape[1]):
            lam.append(model.getVarByName(f"lam2({i})").getInfo("value"))

        return x, c, y, phi, lam

    def _create_model_no_bounds(self, inst: inv_instance.InvLpInstance, x0, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Master")
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x", lb=0.0)
        y = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y")
        c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c")
        ome1 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome1")
        ome2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome2")
        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")
        phi = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="phi", lb=0.0, ub=1.0)

        model.setObjective(ome1.sum() + ome2.sum(), coptpy.COPT.MINIMIZE)

        # Ax == b
        model.addConstrs(inst.a @ x == inst.b)

        # ATy <= c
        model.addConstrs(inst.a.T @ y <= c)

        # KKT conditions
        model.addConstrs(x <= phi * big_m)

        # KKT conditions
        model.addConstrs(c - inst.a.T @ y <= (1 - phi) * big_m)

        # counting
        model.addConstrs(c - inst.a.T @ y >= lam2 * eps)
        model.addConstrs(y >= lam1 * eps - gam * big_m)
        model.addConstrs(y <= -lam1 * eps + (1 - gam) * big_m)
        model.addConstr(lam1.sum() + lam2.sum() == m)

        # for objective
        model.addConstrs(-ome1 <= x - x0)
        model.addConstrs(x - x0 <= ome1)

        # for objective
        model.addConstrs(-ome2 <= c - inst.c)
        model.addConstrs(c - inst.c <= ome2)

        self._logger.info("Model is created.")

        return model

    def _create_model_upper_bounds(self, inst: inv_instance.InvLpInstance, x0, big_m=10e6, eps=10e-6):
        model: coptpy.Model = self._envr.createModel(name="Master")
        model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        n, m = inst.a.shape

        x = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x", lb=0.0)
        y = model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y")
        z = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="z", lb=0.0)
        c = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c")
        ome1 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome1")
        ome2 = model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="ome2")
        lam1 = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="lam1")
        lam2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        lam3 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="lam2")
        gam = model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="gam")
        phi1 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="phi")
        phi2 = model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="phi1")

        model.setObjective(ome1.sum() + ome2.sum(), coptpy.COPT.MINIMIZE)

        if n != 0:
            # Ax == b
            model.addConstrs(inst.a @ x == inst.b)

        # x  <= u
        model.addConstrs(x <= inst.upper_bounds)

        # ATy - z <= c
        model.addConstrs(inst.a.T @ y - z <= c)

        # KKT conditions
        model.addConstrs(x <= phi1 * big_m)
        model.addConstrs(c - inst.a.T @ y - z <= (1 - phi1) * big_m)
        model.addConstrs(z <= phi2 * big_m)
        model.addConstrs(inst.upper_bounds - x <= big_m * (1 - phi2))

        # counting
        model.addConstrs(c - inst.a.T @ y - z >= lam2 * eps)
        model.addConstrs(z >= lam3 * eps)
        if n != 0:
            model.addConstrs(y >= lam1 * eps - gam * big_m)
            model.addConstrs(y <= -lam1 * eps + (1 - gam) * big_m)
        model.addConstr(lam1.sum() + lam2.sum() + lam3.sum() == m)

        # for objective
        model.addConstrs(-ome1 <= x - x0)
        model.addConstrs(x - x0 <= ome1)

        # for objective
        model.addConstrs(-ome2 <= c - inst.c)
        model.addConstrs(c - inst.c <= ome2)

        self._logger.info("Model is created.")
        return model
