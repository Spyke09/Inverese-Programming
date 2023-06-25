import typing as tp

import numpy as np
import pulp


class LpInstance:
    """
    Класс для хранения ЗЛП вида.
    c.x -> min
    Ax >= b
    l <= x <= u
    """
    def __init__(self, a, b, c, lower_bounds=None, upper_bounds=None):
        self._c: np.array = np.array(c)
        self._a: np.array = np.array(a)
        self._b: np.array = np.array(b)
        self._upper_bounds: tp.Optional[np.array] = None
        self._lower_bounds: tp.Optional[np.array] = None
        if upper_bounds is not None:
            self._upper_bounds = np.array(upper_bounds)
        if lower_bounds is not None:
            self._lower_bounds = np.array(lower_bounds)

    @property
    def a(self) -> np.array:
        return self._a

    @property
    def b(self) -> np.array:
        return self._b

    @property
    def c(self) -> np.array:
        return self._c

    @property
    def upper_bounds(self) -> np.array:
        return self._upper_bounds

    @property
    def lower_bounds(self) -> np.array:
        return self._lower_bounds


def create_pulp_model(instance: LpInstance, name: str = "UNNAMED"):
    """
    Создание модели pulp из модели LpInstance.

    :param instance: исходный экземпляр ЗЛП.
    :param name: опционально - имя модели pulp
    :return: модель pulp.
    """
    n, m = 0, 0
    if len(instance.a) != 0:
        n, m = instance.a.shape

    model = pulp.LpProblem(name, pulp.LpMinimize)
    x = list()
    t1, t2 = instance.lower_bounds is None, instance.upper_bounds is None
    for i in range(m):
        if t1 and t2:
            x_i = pulp.LpVariable(f"x_{i}")
        elif not t1 and t2:
            x_i = pulp.LpVariable(f"x_{i}", lowBound=instance.lower_bounds[i])
        elif t1 and not t2:
            x_i = pulp.LpVariable(f"x_{i}", upBound=instance.upper_bounds[i])
        else:
            x_i = pulp.LpVariable(f"x_{i}", lowBound=instance.lower_bounds[i], upBound=instance.upper_bounds[i])

        x.append(x_i)
    # целевая функция
    model += pulp.lpSum([instance.c[i] * x[i] for i in range(m)])
    # ограницения из матрицы a
    for i in range(n):
        model += (pulp.lpSum([x[j] * instance.a[i, j] for j in range(m)]) >= instance.b[i])

    return model


def get_x_after_model_solve(model):
    x = np.full(len(model.variables()), 0)
    for v in model.variables():
        x[int(v.name.replace("x_", ''))] = v.varValue
    return x

