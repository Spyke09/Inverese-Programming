import pulp
import typing as tp
import numpy as np
from enum import Enum
from src import simple_instance


class NormType(Enum):
    L1 = 0,
    LInfinity = 1


class InverseLpSolver:
    def __init__(self, p: NormType):
        self.__p = p
        self.__answer: tp.Optional[bool] = None
        self.__solution: tp.Optional[np.array] = None

    @staticmethod
    def __find_binding_constraints(instance: simple_instance.LpInstance, answer: np.array):
        diff_for_b = instance.a @ answer - instance.b
        idx_mask_b = [i == 0 for i in diff_for_b]

        t1, t2 = (instance.lower_bounds is not None), (instance.upper_bounds is not None)
        if t1 and not t2:
            diff_for_l = answer - instance.lower_bounds
            idx_mask_l = [i == 0 for i in diff_for_l]
            idx_mask_f = [i != 0 for i in diff_for_l]
            return idx_mask_b, idx_mask_f, idx_mask_l, None
        elif t2 and not t1:
            diff_for_u = answer - instance.upper_bounds
            idx_mask_u = [i == 0 for i in diff_for_u]
            idx_mask_f = [i != 0 for i in diff_for_u]
            return idx_mask_b, idx_mask_f, None, idx_mask_u
        elif t1 and t2:
            diff_for_l = answer - instance.lower_bounds
            diff_for_u = answer - instance.upper_bounds
            idx_mask_l = [i == 0 for i in diff_for_l]
            idx_mask_u = [i == 0 for i in diff_for_u]
            idx_mask_f = [(not idx_mask_l[i]) and (not idx_mask_u[i]) for i in range(len(idx_mask_u))]
            return idx_mask_b, idx_mask_f, idx_mask_l, idx_mask_u
        else:
            return idx_mask_b, [True for _ in idx_mask_b], None, None

    @staticmethod
    def __create_inv_lp_instance(instance: simple_instance.LpInstance, masks, x0):
        n, m = instance.a.shape
        b_mask, f_mask, l_mask, u_mask = masks
        len_b, len_f, len_l, len_u = map((lambda x: 0 if x is None else len(x)), masks)
        a = list()
        b = list()
        c = instance.c
        u = np.array([None for _ in range(m)])
        l = np.array([None for _ in range(m)])

        for i in range(n):
            if b_mask[i]:
                a.append(instance.a[i])
                b.append(instance.b[i])

        for j in range(m):
            if l_mask and l_mask[j]:
                l[j] = instance.lower_bounds[j]
                u[j] = instance.lower_bounds[j] + 1
            if u_mask and u_mask[j]:
                l[j] = instance.upper_bounds[j] - 1
                u[j] = instance.upper_bounds[j]
            if f_mask[j]:
                l[j] = x0[j] - 1
                u[j] = x0[j] + 1

        return simple_instance.LpInstance(np.array(a), np.array(b), c, l, u)

    @staticmethod
    def __get_d(instance: simple_instance.LpInstance, dual_inv_answer: np.array, x0: np.array):
        c_pi = instance.c - instance.a.transpose().dot(dual_inv_answer)
        d = instance.c.copy()
        for j in range(len(d)):
            if c_pi[j] > 0 and x0[j] > instance.lower_bounds[j]:
                d[j] -= abs(c_pi[j])
            elif c_pi[j] < 0 and x0[j] < instance.upper_bounds[j]:
                d[j] -= abs(c_pi[j])
        return d

    def solve(self, instance: simple_instance.LpInstance, x0: np.array, weights: np.array = None):
        if p == NormType.L1:
            model = simple_instance.create_pulp_model(instance)

            status = model.solve(pulp.PULP_CBC_CMD(msg=False))
            if status != 1:
                raise ValueError("Instance status is False")

            answer = np.array([i.varValue for i in model.variables()])

            masks = self.__find_binding_constraints(instance, answer)

            inv_instance = self.__create_inv_lp_instance(instance, masks, x0)
            inv_model = simple_instance.create_pulp_model(inv_instance)
            inv_status = inv_model.solve(pulp.PULP_CBC_CMD(msg=False))
            if inv_status != 1:
                raise ValueError("Instance status is False")

            dual_inv_answer = np.array([i.pi for _, i in inv_model.constraints.items()][:inv_instance.b.shape[0]])
            result_d = self.__get_d(inv_instance, dual_inv_answer, x0)

            return result_d

