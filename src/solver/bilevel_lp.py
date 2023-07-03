import numpy as np
import pulp

from src.config import config
from src.structures import bilevel_instance, simple_instance


class BilevelLpSolver:
    @staticmethod
    def _create_MPEC_model(inst, x0, name="UNNAMED", big_m=10000000):
        """
        Создание модели MPEC.

        :return: модель pulp.
        """
        n, m = 0, 0
        if len(inst.a) != 0:
            n, m = inst.a.shape

        model = pulp.LpProblem(name, pulp.LpMinimize)
        # переменные
        x = [pulp.LpVariable(f"x_{i}", lowBound=inst.lower_bounds[i]) for i in range(m)]
        b = [pulp.LpVariable(f"b_{i}") for i in range(n)]
        c = [pulp.LpVariable(f"c_{i}") for i in range(m)]
        y = [pulp.LpVariable(f"y_{i}") for i in range(n)]
        kkt = [pulp.LpVariable(f"kkt_{i}", cat='Binary') for i in range(m)]
        alpha = [pulp.LpVariable(f"alpha_{i}", lowBound=0) for i in range(m)]
        beta = [pulp.LpVariable(f"beta_{i}", lowBound=0) for i in range(m)]

        # ограничения Ax == b
        assert inst.sign == simple_instance.LpSign.Equal
        for i in range(n):
            model += (pulp.lpSum([x[j] * inst.a[i, j] for j in range(m)]) == b[i])

        # ограничения transpose(A)y >= c
        for j in range(m):
            model += (pulp.lpSum([y[i] * inst.a[i, j] for i in range(n)]) >= c[j])

        # ограничения Bb = ~b
        assert inst.big_b.shape[0] == inst.b.shape[0]
        for i in range(inst.big_b.shape[0]):
            model += (pulp.lpSum([b[j] * inst.big_b[i, j] for j in range(inst.big_b.shape[1])]) == inst.b[i])

        # ограничения Cc = ~c
        assert inst.big_c.shape[0] == inst.c.shape[0]
        for i in range(inst.big_c.shape[0]):
            model += (pulp.lpSum([c[j] * inst.big_c[i, j] for j in range(inst.big_c.shape[1])]) == inst.c[i])

        # ограничения KKT
        # (x_j * (a_j * y_j - c_j) <=> x_j <= kkt_j * M and (a_j * y_j - c_j) <= (1 - kkt_j) * M
        for j in range(m):
            model += (x[j] <= kkt[j] * big_m)
            model += (pulp.lpSum([y[i] * inst.a[i, j] for i in range(n)]) - c[j] <= big_m * (1 - kkt[j]))

        # это для целевой функции
        for j in range(m):
            model += (x[j] - x0[j] == alpha[j] - beta[j])

        # целевая функция
        model += pulp.lpSum(alpha + beta)

        return model

    def solve(self, inst: bilevel_instance.BilevelInstance, x0):
        model = self._create_MPEC_model(inst, x0)
        model.solve(config.SOLVER)

        if model.status != 1:
            raise ValueError("Status after model solving is False")

        x, b, c = list(), list(), list()
        for v in model.variables():
            if "x_" in v.name:
                x.append(v.varValue)
            if "b_" in v.name:
                b.append(v.varValue)
            if "c_" in v.name:
                c.append(v.varValue)

        return np.array(x), np.array(b), np.array(c)
