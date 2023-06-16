import numpy as np
import pulp
import typing as tp


class LpInstance:
    def __init__(self,
                 a: np.array,
                 b: np.array,
                 c: np.array,
                 lower_bounds: tp.Optional[np.array] = None,
                 upper_bounds: tp.Optional[np.array] = None):
        self._c: np.array = c
        self._a: np.array = a
        self._b: np.array = b
        self._upper_bounds: tp.Optional[np.array] = upper_bounds
        self._lower_bounds: tp.Optional[np.array] = lower_bounds

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
    n, m = instance.a.shape
    model = pulp.LpProblem(name, pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", [i for i in range(m)])
    model += pulp.lpSum([instance.c[i] * x[i] for i in range(m)])
    for i in range(n):
        model += (pulp.lpSum([x[j] * instance.a[i, j] for j in range(m)]) >= instance.b[i])

    if instance.upper_bounds is not None:
        for i in range(m):
            model += (x[i] <= instance.upper_bounds[i])

    if instance.lower_bounds is not None:
        for i in range(m):
            model += (x[i] >= instance.lower_bounds[i])

    return model
