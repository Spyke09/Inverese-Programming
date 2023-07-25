import numpy as np
import pulp

import inverse_programming.src.config.config as config
from inverse_programming.src.structures import inv_instance, bilevel_instance


class BilevelLpSolver:
    @staticmethod
    def _create_MPEC_model_with_upper_bounds(inst, x0, name="UNNAMED", big_m=10000000):
        """
        Создание модели MPEC с верхними ограничениями.

        :return: модель pulp.
        """
        n, m = inst.a.shape

        model = pulp.LpProblem(name, pulp.LpMinimize)
        # переменные
        x = [pulp.LpVariable(f"x_{i}", lowBound=0.0) for i in range(m)]
        la = [pulp.LpVariable(f"la_{i}", lowBound=0.0) for i in range(m)]
        big_x = x + la
        b = [pulp.LpVariable(f"b_{i}") for i in range(n)]
        c = [pulp.LpVariable(f"c_{i}") for i in range(m)]
        y = [pulp.LpVariable(f"y_{i}") for i in range(n)]
        nu = [pulp.LpVariable(f"nu_{i}") for i in range(m)]
        big_y = y + nu
        kkt = [pulp.LpVariable(f"kkt_{i}", cat='Binary') for i in range(2 * m)]
        alpha = [pulp.LpVariable(f"alpha_{i}", lowBound=0) for i in range(m)]
        beta = [pulp.LpVariable(f"beta_{i}", lowBound=0) for i in range(m)]

        true_c = c + list(np.full(m, 0))

        # TODO: fix this
        true_b = b + inst.upper_bounds.to_list()
        true_a = inv_instance.LPArray((n + m, 2 * m))

        for i in range(n):
            for j in range(m):
                if inst.a[i, j] != 0:
                    true_a[i, j] = inst.a[i, j]

        for i in range(m):
            true_a[n + i, i] = 1.
            true_a[n + i, m + i] = 1.
        # ограничения Ax == b
        assert inst.sign == inv_instance.LpSign.Equal
        for i in range(n + m):
            model += (pulp.lpSum([big_x[j] * true_a[i, j] for j in range(2 * m)]) == true_b[i])

        # ограничения transpose(A)y >= c
        for j in range(2 * m):
            model += (pulp.lpSum([big_y[i] * true_a[i, j] for i in range(n + m)]) >= true_c[j])

        # ограничения Bb = ~b
        assert inst.big_b.shape[0] == inst.b.shape[1]
        for i in range(inst.big_b.shape[0]):
            model += (pulp.lpSum([b[j] * inst.big_b[i, j] for j in range(inst.big_b.shape[1])]) == inst.b[0, i])

        # ограничения Cc = ~c
        assert inst.big_c.shape[0] == inst.c.shape[1]
        for i in range(inst.big_c.shape[0]):
            model += (pulp.lpSum([c[j] * inst.big_c[i, j] for j in range(inst.big_c.shape[1])]) == inst.c[0, i])

        # ограничения KKT
        # (x_j * (a_j * y_j - c_j) <=> x_j <= kkt_j * M and (a_j * y_j - c_j) <= (1 - kkt_j) * M
        for j in range(2 * m):
            model += (big_x[j] <= kkt[j] * big_m)
            model += (pulp.lpSum([big_y[i] * true_a[i, j] for i in range(m + n)]) - true_c[j] <= big_m * (1 - kkt[j]))

        # это для целевой функции
        for j in range(m):
            model += (x[j] - x0[0, j] == alpha[j] - beta[j])

        # целевая функция
        model += pulp.lpSum(alpha + beta)

        return model

    @staticmethod
    def _create_MPEC_model(inst, x0, name="UNNAMED", big_m=10000000):
        """
        Создание модели MPEC.

        :return: модель pulp.
        """
        n, m = 0, 0
        n, m = inst.a.shape

        model = pulp.LpProblem(name, pulp.LpMinimize)
        # переменные
        x = [pulp.LpVariable(f"x_{i}", lowBound=0.0) for i in range(m)]
        b = [pulp.LpVariable(f"b_{i}") for i in range(n)]
        c = [pulp.LpVariable(f"c_{i}") for i in range(m)]
        y = [pulp.LpVariable(f"y_{i}") for i in range(n)]
        kkt = [pulp.LpVariable(f"kkt_{i}", cat='Binary') for i in range(m)]
        alpha = [pulp.LpVariable(f"alpha_{i}", lowBound=0.0) for i in range(m)]
        beta = [pulp.LpVariable(f"beta_{i}", lowBound=0.0) for i in range(m)]

        # ограничения Ax == b
        assert inst.sign == inv_instance.LpSign.Equal
        for i in range(n):
            model += (pulp.lpSum([x[j] * inst.a[i, j] for j in range(m)]) == b[i])

        # ограничения transpose(A)y >= c
        for j in range(m):
            model += (pulp.lpSum([y[i] * inst.a[i, j] for i in range(n)]) >= c[j])

        # ограничения Bb = ~b
        assert inst.big_b.shape[0] == inst.b.shape[1]
        for i in range(inst.big_b.shape[0]):
            model += (pulp.lpSum([b[j] * inst.big_b[i, j] for j in range(inst.big_b.shape[1])]) == inst.b[0, i])

        # ограничения Cc = ~c
        assert inst.big_c.shape[0] == inst.c.shape[1]
        for i in range(inst.big_c.shape[0]):
            model += (pulp.lpSum([c[j] * inst.big_c[i, j] for j in range(inst.big_c.shape[1])]) == inst.c[0, i])

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
        if inst.lower_bounds.l1_norm() != 0:
            raise ValueError("Lower bounds should be zero.")

        if inst.upper_bounds is None:
            model = self._create_MPEC_model(inst, x0)
        else:
            model = self._create_MPEC_model_with_upper_bounds(inst, x0)
        model.solve(config.PULP_SOLVER)

        if model.status != 1:
            raise ValueError("Status after model solving is False")

        n, m = inst.a.shape
        x, b, c = np.full(m, 0.), np.full(n, 0.), np.full(m, 0.)
        for v in model.variables():
            if "x_" in v.name:
                x[int(v.name[2:])] = v.varValue
            if "c_" in v.name:
                c[int(v.name[2:])] = v.varValue
            if "b_" in v.name:
                b[int(v.name[2:])] = v.varValue

        return inv_instance.LPArray(x), inv_instance.LPArray(b), inv_instance.LPArray(c)
