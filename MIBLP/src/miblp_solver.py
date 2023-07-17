import logging

import coptpy
import numpy as np

from MIBLP.src.miblp_instance import MIBPLInstance

LOG_LEVEL = 0


class MIBLPSolver:
    """
    Class that solves MIBPL problem using "Decomposition Algorithm" discribed in the paper:

    ```A Projection-Based Reformulation and Decomposition Algorithm for Global
    Optimization of a Class of Mixed Integer Bilevel Linear Programs.

    Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You```

    """

    def __init__(self):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("MIBLPSolver")

    def _master_problem_init(self, inst: MIBPLInstance) -> coptpy.Model:
        model: coptpy.Model = self._envr.createModel(name="Master")
        model.setParam(coptpy.COPT.Param.Logging, LOG_LEVEL)
        # (15)
        x_u = model.addVars(range(inst.m_r), vtype=coptpy.COPT.CONTINUOUS, nameprefix="x_u")
        y_u = model.addVars(range(inst.m_z), vtype=coptpy.COPT.INTEGER, nameprefix="y_u")
        x_l0 = model.addVars(range(inst.n_r), vtype=coptpy.COPT.CONTINUOUS, nameprefix="x_l0")
        y_l0 = model.addVars(range(inst.n_z), vtype=coptpy.COPT.INTEGER, nameprefix="y_l0")

        # min(c_r * x_u + c_z * y_u + d_r * x_l0 + d_z * y_l0, {x_u, y_u, x_l0, y_l0, x_l_j, pi_l_j})
        # (11), (86)
        model.setObjective(
            sum(inst.c_r[j] * x_u[j] for j in range(inst.c_r.shape[0])) +
            sum(inst.c_z[j] * y_u[j] for j in range(inst.c_z.shape[0])) +
            sum(inst.d_r[j] * x_l0[j] for j in range(inst.d_r.shape[0])) +
            sum(inst.d_z[j] * y_l0[j] for j in range(inst.d_z.shape[0])),
            coptpy.COPT.MINIMIZE
        )

        # a_r * x_u + a_z * y_u + b_r * x_l0 + b_z * y_l0 <= r
        # (12)
        model.addConstrs(
            sum(inst.a_r[i, j] * x_u[j] for j in range(inst.a_r.shape[1])) +
            sum(inst.a_z[i, j] * y_u[j] for j in range(inst.a_z.shape[1])) +
            sum(inst.b_r[i, j] * x_l0[j] for j in range(inst.b_r.shape[1])) +
            sum(inst.b_z[i, j] * y_l0[j] for j in range(inst.b_z.shape[1])) <= inst.r[i]
            for i in range(inst.r.shape[0])
        )

        # q_r * x_u + q_z * y_u + p_r * x_l0 + p_z * y_l0 <= s
        # (13)
        model.addConstrs(
            sum(inst.q_r[i, j] * x_u[j] for j in range(inst.q_r.shape[1])) +
            sum(inst.q_z[i, j] * y_u[j] for j in range(inst.q_z.shape[1])) +
            sum(inst.p_r[i, j] * x_l0[j] for j in range(inst.p_r.shape[1])) +
            sum(inst.p_z[i, j] * y_l0[j] for j in range(inst.p_z.shape[1])) <= inst.s[i]
            for i in range(inst.s.shape[0])
        )

        self._x_u = x_u
        self._y_u = y_u
        self._x_l0 = x_l0
        self._y_l0 = y_l0
        return model

    def _subproblem_1_init(
            self,
            inst: MIBPLInstance,
            x_u_k: np.array,
            y_u_k: np.array) -> coptpy.Model:
        model: coptpy.Model = self._envr.createModel(name="Subproblem 1")
        model.setParam(coptpy.COPT.Param.Logging, LOG_LEVEL)
        x_l = model.addVars(range(inst.n_r), vtype=coptpy.COPT.CONTINUOUS, nameprefix="x_l")
        y_l = model.addVars(range(inst.n_z), vtype=coptpy.COPT.INTEGER, nameprefix="y_l")

        # theta_small(x_u_k, y_u_k) = max(w_r * x_l + w_z * y_l, {x_l, y_l})
        model.setObjective(
            sum(inst.w_r[j] * x_l[j] for j in range(inst.w_r.shape[0])) +
            sum(inst.w_z[j] * y_l[j] for j in range(inst.w_z.shape[0])),
            coptpy.COPT.MAXIMIZE
        )

        # p_r * x_l + p_z * y_l <= s - q_r * x_u_k - q_z * y_u_k
        model.addConstrs(
            sum(inst.p_r[i, j] * x_l[j] for j in range(inst.p_r.shape[1])) +
            sum(inst.p_z[i, j] * y_l[j] for j in range(inst.p_z.shape[1])) <=
            inst.s[i] -
            sum(inst.q_r[i, j] * x_u_k[j] for j in range(inst.q_r.shape[1])) -
            sum(inst.q_z[i, j] * y_u_k[j] for j in range(inst.q_z.shape[1]))
            for i in range(inst.s.shape[0])
        )

        return model

    def _subproblem_2_init(
            self,
            inst: MIBPLInstance,
            x_u_k: np.array,
            y_u_k: np.array,
            theta_small_k: float) -> coptpy.Model:
        model: coptpy.Model = self._envr.createModel(name="Subproblem 2")
        model.setParam(coptpy.COPT.Param.Logging, LOG_LEVEL)
        x_l = model.addVars(range(inst.n_r), vtype=coptpy.COPT.CONTINUOUS, nameprefix="x_l")
        y_l = model.addVars(range(inst.n_z), vtype=coptpy.COPT.INTEGER, nameprefix="y_l")

        # tetta_big(x_u_k, y_u_k) = min(d_r * x_l + d_z * y_l, {x_l, y_l})
        model.setObjective(
            sum(inst.d_r[j] * x_l[j] for j in range(inst.d_r.shape[0])) +
            sum(inst.d_z[j] * y_l[j] for j in range(inst.d_z.shape[0])),
            coptpy.COPT.MINIMIZE
        )

        # p_r * x_l + p_z * y_l <= s - q_r * x_u_k - q_z * y_u_k
        model.addConstrs(
            sum(inst.p_r[i, j] * x_l[j] for j in range(inst.p_r.shape[1])) +
            sum(inst.p_z[i, j] * y_l[j] for j in range(inst.p_z.shape[1])) <=
            inst.s[i] -
            sum(inst.q_r[i, j] * x_u_k[j] for j in range(inst.q_r.shape[1])) -
            sum(inst.q_z[i, j] * y_u_k[j] for j in range(inst.q_z.shape[1]))
            for i in range(inst.s.shape[0])
        )

        # b_r * x_l0 + b_z * y_l0 <= r - a_r * x_u_k - a_z * y_u_k
        model.addConstrs(
            sum(inst.b_r[i, j] * x_l[j] for j in range(inst.b_r.shape[1])) +
            sum(inst.b_z[i, j] * y_l[j] for j in range(inst.b_z.shape[1])) <=
            inst.r[i] -
            sum(inst.a_r[i, j] * x_u_k[j] for j in range(inst.a_r.shape[1])) -
            sum(inst.a_z[i, j] * y_u_k[j] for j in range(inst.a_z.shape[1]))
            for i in range(inst.r.shape[0])
        )

        # w_r * x_l + w_z * y_l >= theta_small(x_l, y_l)
        model.addConstr(
            sum(inst.w_r[j] * x_l[j] for j in range(inst.w_r.shape[0])) +
            sum(inst.w_z[j] * y_l[j] for j in range(inst.w_z.shape[0])) >= theta_small_k
        )

        return model

    @staticmethod
    def _get_optimal_solution_from_master(master: coptpy.Model, inst: MIBPLInstance):
        x_u = list()
        y_u = list()
        x_l0 = list()
        y_l0 = list()
        for i in range(inst.m_r):
            x_u.append(master.getVarByName(f"x_u({i})").getInfo("value"))
        for i in range(inst.m_z):
            y_u.append(master.getVarByName(f"y_u({i})").getInfo("value"))
        for i in range(inst.n_r):
            x_l0.append(master.getVarByName(f"x_l0({i})").getInfo("value"))
        for i in range(inst.n_z):
            y_l0.append(master.getVarByName(f"y_l0({i})").getInfo("value"))

        return np.array(x_u), np.array(y_u), np.array(x_l0), np.array(y_l0)

    @staticmethod
    def _get_optimal_solution_from_subproblem(subproblem: coptpy.Model, inst: MIBPLInstance):
        x_l = list()
        y_l = list()
        for i in range(inst.n_r):
            x_l.append(subproblem.getVarByName(f"x_l({i})").getInfo("value"))
        for i in range(inst.n_z):
            y_l.append(subproblem.getVarByName(f"y_l({i})").getInfo("value"))

        return np.array(x_l), np.array(y_l)

    def _update_master_problem(self, master: coptpy.Model, y_l_j: np.array, inst: MIBPLInstance, k):
        big_m = 10e7
        eps = 10e-1
        n_l = inst.s.shape[0]
        # (80)
        x_l_j = master.addVars(range(inst.n_r), vtype=coptpy.COPT.CONTINUOUS, nameprefix=f"x_l_{k}")
        pi_j = master.addVars(range(n_l), vtype=coptpy.COPT.CONTINUOUS, nameprefix=f"pi_l_{k}")
        la = master.addVars(range(n_l), vtype=coptpy.COPT.CONTINUOUS, nameprefix=f"la_{k}")

        kkt_1 = master.addVars(range(inst.n_r), vtype=coptpy.COPT.BINARY, nameprefix=f"kkt_1_{k}")
        kkt_2 = master.addVars(range(n_l), vtype=coptpy.COPT.BINARY, nameprefix=f"kkt_2_{k}")
        kkt_3 = master.addVars(range(inst.n_r), vtype=coptpy.COPT.BINARY, nameprefix=f"kkt_3_{k}")
        kkt_4 = master.addVars(range(n_l), vtype=coptpy.COPT.BINARY, nameprefix=f"kkt_4_{k}")
        kkt_5 = master.addVars(range(n_l), vtype=coptpy.COPT.BINARY, nameprefix=f"kkt_5_{k}")

        t_j = master.addVars(range(n_l), vtype=coptpy.COPT.CONTINUOUS, nameprefix=f"t_{k}")
        psi_j = master.addVar(vtype=coptpy.COPT.BINARY, name=f"psi_{k}")

        # (79)
        master.addConstrs(
            sum(inst.p_r[i, j] * x_l_j[j] for j in range(inst.p_r.shape[1])) -
            t_j[i] <=
            inst.s[i] -
            sum(inst.q_r[i, j] * self._x_u[j] for j in range(inst.q_r.shape[1])) -
            sum(inst.q_z[i, j] * self._y_u[j] for j in range(inst.q_z.shape[1])) -
            sum(inst.p_z[i, j] * y_l_j[j] for j in range(inst.p_z.shape[1]))
            for i in range(n_l)
        )

        # (83)
        master.addConstrs(
            sum(inst.p_r[i, j] * la[i] for i in range(n_l)) >= 0
            for j in range(inst.n_r)
        )
        master.addConstrs(x_l_j[i] <= big_m * kkt_3[i] for i in range(inst.n_r))
        master.addConstrs(
            sum(inst.p_r[i, j] * la[i] for i in range(n_l)) <= big_m - big_m * kkt_3[j]
            for j in range(inst.n_r)
        )

        # (84)
        master.addConstrs(la[i] <= 1.0 for i in range(n_l))
        master.addConstrs(t_j[i] <= big_m * kkt_4[i] for i in range(n_l))
        master.addConstrs(
            1.0 - la[i] <= big_m - big_m * kkt_4[i]
            for i in range(n_l)
        )

        # (85)
        master.addConstrs(la[i] <= big_m * kkt_5[i] for i in range(n_l))
        master.addConstrs(
            inst.s[i] -
            sum(inst.p_r[i, j] * x_l_j[j] for j in range(inst.p_r.shape[1])) -
            sum(inst.q_r[i, j] * self._x_u[j] for j in range(inst.q_r.shape[1])) -
            sum(inst.q_z[i, j] * self._y_u[j] for j in range(inst.q_z.shape[1])) -
            sum(inst.p_z[i, j] * y_l_j[j] for j in range(inst.p_z.shape[1])) +
            t_j[i] <=
            big_m - big_m * kkt_5[i]
            for i in range(n_l)
        )

        bin_val = 1
        # (87)
        master.addConstr(
            (psi_j == bin_val) >>
            (sum(inst.w_r[j] * self._x_l0[j] for j in range(inst.w_r.shape[0])) +
             sum(inst.w_z[j] * self._y_l0[j] for j in range(inst.w_z.shape[0])) >=
             sum(inst.w_r[j] * x_l_j[j] for j in range(inst.w_r.shape[0])) +
             sum(inst.w_z[j] * y_l_j[j] for j in range(inst.w_z.shape[0])))
        )

        master.addConstrs(
            ((psi_j == bin_val) >>
             (sum(inst.p_r[i, j] * x_l_j[j] for j in range(inst.p_r.shape[1])) <=
              inst.s[i] -
              sum(inst.q_r[i, j] * self._x_u[j] for j in range(inst.q_r.shape[1])) -
              sum(inst.q_z[i, j] * self._y_u[j] for j in range(inst.q_z.shape[1])) -
              sum(inst.p_z[i, j] * y_l_j[j] for j in range(inst.p_z.shape[1]))))
            for i in range(n_l)
        )

        master.addConstrs(
            ((psi_j == bin_val) >>
             (sum(inst.p_r[i, j] * pi_j[i] for i in range(n_l)) >= inst.w_r[j]))
            for j in range(inst.w_r.shape[0])
        )
        master.addConstrs(((psi_j == bin_val) >> (x_l_j[j] <= kkt_1[j] * big_m)) for j in range(inst.w_r.shape[0]))
        master.addConstrs(
            ((psi_j == bin_val) >>
             (sum(inst.p_r[i, j] * pi_j[i] for i in range(n_l)) - inst.w_r[j] <= big_m - kkt_1[j] * big_m))
            for j in range(inst.w_r.shape[0])
        )

        master.addConstrs(((psi_j == bin_val) >> (pi_j[i] <= kkt_2[i] * big_m)) for i in range(n_l))
        master.addConstrs(
            ((psi_j == bin_val) >>
             (inst.s[i] -
              sum(inst.p_r[i, j] * x_l_j[j] for j in range(inst.p_r.shape[1])) -
              sum(inst.q_r[i, j] * self._x_u[j] for j in range(inst.q_r.shape[1])) -
              sum(inst.q_z[i, j] * self._y_u[j] for j in range(inst.q_z.shape[1])) -
              sum(inst.p_z[i, j] * y_l_j[j] for j in range(inst.p_z.shape[1])) <=
              big_m - kkt_2[i] * big_m))
            for i in range(n_l)
        )

        # (88)
        master.addConstr(eps - eps * psi_j <= sum(t_j[i] for i in range(n_l)))

    def solve(self, inst: MIBPLInstance, iter_limit=100, eps=10e-7, ):
        upper_bound = coptpy.COPT.INFINITY
        k = 0
        master = self._master_problem_init(inst)

        self._logger.info("Start solving MIBLP problem.")
        while True:
            master.solve()
            if master.status != coptpy.COPT.OPTIMAL:
                self._logger.info("Master problem is infeasible.")
                raise ValueError("Master problem should be feasible")

            self._logger.debug("Master problem is solved.")
            x_u, y_u, x_l0, y_l0 = self._get_optimal_solution_from_master(master, inst)
            self._logger.debug(f"Master answer: \nx_u = {x_u}, \ny_u = {y_u}, \nx_l0 = {x_l0}, \ny_l0 = {y_l0}")

            lower_bound = inst.c_r.dot(x_u) + inst.c_z.dot(y_u) + inst.d_r.dot(x_l0) + inst.d_z.dot(y_l0)
            self._logger.debug(f"Current bounds: ({lower_bound}, {upper_bound})")

            if abs(upper_bound - lower_bound) <= eps:
                self._logger.debug(f"Required precision obtained at {k+1} step.")
                break

            subproblem_1 = self._subproblem_1_init(inst, x_u, y_u)
            subproblem_1.solve()
            if master.status != coptpy.COPT.OPTIMAL:
                self._logger.debug(f"Subproblem 1 is infeasible.")
                raise ValueError("Subproblem 1 should be feasible")

            self._logger.debug("Subproblem 1 problem is solved.")
            x_l_hat, y_l_hat = self._get_optimal_solution_from_subproblem(subproblem_1, inst)
            theta_small_k = inst.w_r.dot(x_l_hat) + inst.w_z.dot(y_l_hat)
            self._logger.debug(f"Subproblem 1 answer: \nx_l = {x_l_hat}, \ny_l = {y_l_hat}, \ntheta = {theta_small_k}.")

            subproblem_2 = self._subproblem_2_init(inst, x_u, y_u, theta_small_k)
            subproblem_2.solve()
            if subproblem_2.status == coptpy.COPT.OPTIMAL:
                self._logger.debug("Subproblem 2 problem is solved.")
                x_l, y_l = self._get_optimal_solution_from_subproblem(subproblem_2, inst)
                theta_big_k = inst.d_r.dot(x_l) + inst.d_z.dot(y_l)
                upper_bound = min(upper_bound, inst.c_r.dot(x_u) + inst.c_z.dot(y_u) + theta_big_k)
                self._logger.debug(f"Subproblem 2 answer: \nx_l = {x_l}, \ny_l = {y_l}, \ntheta = {theta_big_k}.")

                y_l_arc = y_l
            else:
                self._logger.debug("Subproblem 2 problem is infeasible.")
                y_l_arc = y_l_hat

            if abs(upper_bound - lower_bound) < eps:
                self._logger.debug(f"Current bounds: ({lower_bound}, {upper_bound})")
                self._logger.debug(f"Required precision obtained at {k+1} step.")
                break

            if k == iter_limit:
                self._logger.debug("Iteration limit has been reached.")
                raise ValueError("Iteration limit has been reached")

            self._logger.debug(f"Next y_l_j to master problem = {y_l_arc}.")
            self._update_master_problem(master, y_l_arc, inst, k)
            k += 1

        self._logger.info("Finish solving MIBLP problem.")

        x_u, y_u, x_l0, y_l0 = self._get_optimal_solution_from_master(master, inst)
        self._logger.debug(f"Optimal solution problem: \nx_u = {x_u}, \ny_u = {y_u}, \nx_l0 = {x_l0}, \ny_l0 = {y_l0}")
        return x_u, y_u, x_l0, y_l0
