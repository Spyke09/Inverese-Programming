import abc

import pulp
import typing as tp
import numpy as np
from enum import Enum
from src import simple_instance
from abc import ABC

class NormType(Enum):
    L1 = 0,
    LInfinity = 1


class AbstractInverseLpSolver(ABC):
    @abc.abstractmethod
    def solve(self, instance: simple_instance.LpInstance, x0: np.array, weights: np.array = None):
        raise NotImplementedError

    def _find_binding_constraints(self, instance: simple_instance.LpInstance, answer: np.array):
        diff_for_b = instance.a @ answer - instance.b
        idx_mask_b = [i == .0 for i in diff_for_b]

        t1, t2 = (instance.lower_bounds is not None), (instance.upper_bounds is not None)
        if t1 and not t2:
            diff_for_l = answer - instance.lower_bounds
            idx_mask_l = [i == .0 for i in diff_for_l]
            idx_mask_f = [i != .0 for i in diff_for_l]
            return idx_mask_b, idx_mask_f, idx_mask_l, None
        elif t2 and not t1:
            diff_for_u = answer - instance.upper_bounds
            idx_mask_u = [i == .0 for i in diff_for_u]
            idx_mask_f = [i != .0 for i in diff_for_u]
            return idx_mask_b, idx_mask_f, None, idx_mask_u
        elif t1 and t2:
            diff_for_l = answer - instance.lower_bounds
            diff_for_u = answer - instance.upper_bounds
            idx_mask_l = [i == .0 for i in diff_for_l]
            idx_mask_u = [i == .0 for i in diff_for_u]
            idx_mask_f = [(not idx_mask_l[i]) and (not idx_mask_u[i]) for i in range(len(idx_mask_u))]
            return idx_mask_b, idx_mask_f, idx_mask_l, idx_mask_u
        else:
            return idx_mask_b, [True for _ in idx_mask_b], None, None


class InverseLpSolverL1(AbstractInverseLpSolver):
    @staticmethod
    def __get_d(instance: simple_instance.LpInstance, dual_inv_answer: np.array, x0: np.array):
        c_pi = instance.c - instance.a.transpose().dot(dual_inv_answer)
        d = instance.c.copy()
        for j in range(len(d)):
            if c_pi[j] > 0 and x0[j] > instance.lower_bounds[j]:
                d[j] -= abs(c_pi[j])
            elif c_pi[j] < 0 and x0[j] < instance.upper_bounds[j]:
                d[j] += abs(c_pi[j])
        return d

    @staticmethod
    def __create_inv_lp_instance(instance: simple_instance.LpInstance, masks, x0):
        n, m = instance.a.shape
        b_mask, f_mask, l_mask, u_mask = masks
        a = list()
        b = list()
        c = instance.c
        u = np.array([.0 for _ in range(m)])
        l = np.array([.0 for _ in range(m)])

        for i in range(n):
            if b_mask[i]:
                a.append(instance.a[i])
                b.append(instance.b[i])

        for j in range(m):
            if l_mask and l_mask[j]:
                l[j] = instance.lower_bounds[j]
                u[j] = instance.lower_bounds[j] + 1.
            if u_mask and u_mask[j]:
                l[j] = instance.upper_bounds[j] - 1.
                u[j] = instance.upper_bounds[j]
            if f_mask[j]:
                l[j] = x0[j] - 1.
                u[j] = x0[j] + 1.

        return simple_instance.LpInstance(np.array(a), np.array(b), c, l, u)

    def solve(self, instance: simple_instance.LpInstance, x0: np.array, weights: np.array = None):
        if (instance.a.dot(x0) - instance.b < 0).any():
            raise ValueError("sum(a_ij x0_j) - b_i contains elements < 0")
        model = simple_instance.create_pulp_model(instance)

        status = model.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != 1:
            raise ValueError("Solve status is False")

        answer = np.array([i.varValue for i in model.variables()])

        masks = super()._find_binding_constraints(instance, answer)

        inv_instance = self.__create_inv_lp_instance(instance, masks, x0)
        inv_model = simple_instance.create_pulp_model(inv_instance)

        inv_status = inv_model.solve(pulp.PULP_CBC_CMD(msg=False))
        if inv_status != 1:
            raise ValueError("Solve status is False")

        dual_inv_answer = np.array([i.pi for _, i in inv_model.constraints.items()][:inv_instance.b.shape[0]])
        result_d = self.__get_d(inv_instance, dual_inv_answer, x0)

        if not self.__check_L1(inv_instance, result_d, dual_inv_answer):
            raise ValueError("Solve Error")

        return result_d

    @staticmethod
    def __check_L1(instance: simple_instance.LpInstance, d, pi):
        c1 = (instance.a.transpose().dot(pi) - d == 0).all()
        c2 = (pi >= 0).all()

        return c1 and c2


class InverseLpSolverLInfinity(AbstractInverseLpSolver):
    @staticmethod
    def __get_a_b_mask(a, b, b_mask):
        n, m = a.shape
        a_ = list()
        b_ = list()

        for i in range(n):
            if b_mask[i]:
                a_.append(a[i])
                b_.append(b[i])
        return np.array(a_), np.array(b_)

    def solve(self, instance: simple_instance.LpInstance, x0: np.array, weights: np.array = None):
        if (instance.a.dot(x0) - instance.b < 0).any():
            raise ValueError("sum(a_ij x0_j) - b_i contains elements < 0")
        model = simple_instance.create_pulp_model(instance)

        status = model.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != 1:
            raise ValueError("Solve status is False")

        answer = np.array([i.varValue for i in model.variables()])
        masks = super()._find_binding_constraints(instance, answer)
        a_, b_ = self.__get_a_b_mask(instance.a, instance.b, masks[0])

        inv_model = self.__create_inv_model(a_, b_, instance.c, x0)
        inv_model.solve(pulp.PULP_CBC_CMD(msg=False))
        dual_inv_answer = np.array([i.pi for _, i in inv_model.constraints.items()][:b_.shape[0]])

        return self.__get_d(a_, instance.c, dual_inv_answer, x0)

    @staticmethod
    def __create_inv_model(a, b, c, x0, name: str = "UNNAMED"):
        n, m = a.shape
        model = pulp.LpProblem(name, pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", [i for i in range(m)])
        t = pulp.LpVariable.dicts("t", [i for i in range(m)])

        model += pulp.lpSum([c[i] * x[i] for i in range(m)])

        for i in range(n):
            model += (pulp.lpSum([x[j] * a[i, j] for j in range(m)]) >= b[i])

        for j in range(m):
            model += (t[j] >= 0)
            model += (x[j] - x0[j] <= t[j])
            model += (-t[j] <= x[j] - x0[j])

        model += (pulp.lpSum([t[j] for j in range(m)]) == 1)

        return model

    @staticmethod
    def __get_d(a, c, dual_inv_answer: np.array, x0: np.array):
        c_pi = c - a.transpose().dot(dual_inv_answer)
        d = c.copy()
        for j in range(len(d)):
            if c_pi[j] > 0:
                d[j] -= abs(c_pi[j])
            elif c_pi[j] < 0:
                d[j] += abs(c_pi[j])
        return d

    @staticmethod
    def __check_LInfinity(a, d, pi):
        c1 = (a.transpose().dot(pi) - d == 0).all()
        c2 = (pi >= 0).all()

        return c1 and c2
