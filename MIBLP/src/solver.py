from MIBLP.src.mibpl_instance import MIBPLInstance
import coptpy
import numpy as np


class MIBLPSolver:
    """
    Class that solves MIBPL problem using "Decomposition Algorithm" discribed in the paper:

    ```A Projection-Based Reformulation and Decomposition Algorithm for Global
    Optimization of a Class of Mixed Integer Bilevel Linear Programs.

    Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You```

    """

    def __init__(self):
        self.envr = coptpy.Envr()

    def _master_problem_init(self, inst: MIBPLInstance) -> coptpy.Model:
        model: coptpy.Model = self.envr.createModel(name="Master")
        x_u = model.addVars(range(inst.m_r), vtype=coptpy.COPT.CONTINUOUS, nameprefix="x_u")
        y_u = model.addVars(range(inst.m_z), vtype=coptpy.COPT.INTEGER, nameprefix="y_u")
        x_l0 = model.addVars(range(inst.n_r), vtype=coptpy.COPT.CONTINUOUS, nameprefix="x_l0")
        y_l0 = model.addVars(range(inst.n_z), vtype=coptpy.COPT.INTEGER, nameprefix="y_l0")

        # min(c_r * x_u + c_z * y_u + d_r * x_l0 + d_z * y_l0, {x_u, y_u, x_l0, y_l0, x_l_j, pi_l_j})
        model.setObjective(
            sum(inst.c_r[j] * x_u[j] for j in range(inst.c_r.shape[0])) +
            sum(inst.c_z[j] * y_u[j] for j in range(inst.c_z.shape[0])) +
            sum(inst.d_r[j] * x_l0[j] for j in range(inst.d_r.shape[0])) +
            sum(inst.d_z[j] * y_l0[j] for j in range(inst.d_z.shape[0])),
            coptpy.COPT.MINIMIZE
        )

        # a_r * x_u + a_z * y_u + b_r * x_l0 + b_z * y_l0 <= r
        model.addConstrs(
            sum(inst.a_r[i, j] * x_u[j] for j in range(inst.a_r.shape[1])) +
            sum(inst.a_z[i, j] * y_u[j] for j in range(inst.a_z.shape[1])) +
            sum(inst.b_r[i, j] * x_l0[j] for j in range(inst.b_r.shape[1])) +
            sum(inst.b_z[i, j] * y_l0[j] for j in range(inst.b_z.shape[1])) <= inst.r[i]
            for i in range(inst.r.shape[0])
        )

        # q_r * x_u + q_z * y_u + p_r * x_l0 + p_z * y_l0 <= s
        model.addConstrs(
            sum(inst.q_r[i, j] * x_u[j] for j in range(inst.q_r.shape[1])) +
            sum(inst.q_z[i, j] * y_u[j] for j in range(inst.q_z.shape[1])) +
            sum(inst.p_r[i, j] * x_l0[j] for j in range(inst.p_r.shape[1])) +
            sum(inst.p_z[i, j] * y_l0[j] for j in range(inst.p_z.shape[1])) <= inst.s[i]
            for i in range(inst.s.shape[0])
        )

        return model

    def _subproblem_1_init(
            self,
            inst: MIBPLInstance,
            x_u_k: np.array,
            y_u_k: np.array) -> coptpy.Model:
        model: coptpy.Model = self.envr.createModel(name="Subproblem 1")
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
            sum(inst.q_r[i, j] * x_u_k[j] for j in range(inst.a_r.shape[1])) -
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
        model: coptpy.Model = self.envr.createModel(name="Subproblem 2")
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
            sum(inst.q_r[i, j] * x_u_k[j] for j in range(inst.a_r.shape[1])) -
            sum(inst.q_z[i, j] * y_u_k[j] for j in range(inst.q_z.shape[1]))
            for i in range(inst.s.shape[0])
        )

        # b_r * x_l0 + b_z * y_l0 <= r - a_r * x_u_k - a_z * y_u_k
        model.addConstrs(
            sum(inst.b_r[i, j] * x_l[j] for j in range(inst.p_r.shape[1])) +
            sum(inst.b_z[i, j] * y_l[j] for j in range(inst.p_z.shape[1])) <=
            inst.r[i] -
            sum(inst.a_r[i, j] * x_u_k[j] for j in range(inst.a_r.shape[1])) -
            sum(inst.a_z[i, j] * y_u_k[j] for j in range(inst.q_z.shape[1]))
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

    def solve(self, inst: MIBPLInstance, eps=0.0):
        lower_bound = -coptpy.COPT.INFINITY
        upper_bound = coptpy.COPT.INFINITY
        k = 0
        y_l_k = list()
        master = self._master_problem_init(inst)
        while True:
            master.solve()
            if master.status != coptpy.COPT.OPTIMAL:
                raise ValueError("Master problem should be feasible")

            x_u, y_u, x_l0, y_l0 = self._get_optimal_solution_from_master(master, inst)

            lower_bound = inst.c_r.dot(x_u) + inst.c_z.dot(y_u) + inst.d_r.dot(x_l0) + inst.d_z.dot(y_l0)

            if upper_bound - lower_bound < eps:
                break

            subproblem_1 = self._subproblem_1_init(inst, x_u, y_u)
            subproblem_1.solve()
            if master.status != coptpy.COPT.OPTIMAL:
                raise ValueError("Subproblem 1 should be feasible")

            x_l_hat, y_l_hat = self._get_optimal_solution_from_subproblem(subproblem_1, inst)

            theta_small_k = inst.w_r.dot(x_l_hat) + inst.w_z.dot(y_l_hat)
            subproblem_2 = self._subproblem_2_init(inst, x_u, y_u, theta_small_k)

            if subproblem_2.status == coptpy.COPT.OPTIMAL:
                x_l, y_l = self._get_optimal_solution_from_subproblem(subproblem_2, inst)
                theta_big_k = inst.d_r.dot(x_l) + inst.d_z.dot(y_l)
                upper_bound = min(upper_bound, inst.c_r.dot(x_l) + inst.c_z.dot(y_l) + theta_big_k)
                y_l_arc = y_l
            else:
                y_l_arc = y_l_hat


            # Step 7...

            if upper_bound - lower_bound < eps:
                break

        return self._get_optimal_solution_from_master(master, inst)
