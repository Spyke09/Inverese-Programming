import numpy as np

from inverse_programming.src.structures.simple_instance import LpSign
from inverse_programming.src.structures import simple_instance


class BilevelInstance:
    def __init__(self, a, b, c, big_b, big_c, upper_bounds=None):
        n, m = a.shape
        self._inst = simple_instance.InvLpInstance(a, b, c, LpSign.Equal, np.full(m, 0), upper_bounds)
        self._big_b = big_b
        self._big_c = big_c

    def hide_upper_bounds(self):
        n, m = self.a.shape
        self._inst.hide_upper_bounds()
        big_b = np.full((self._big_b.shape[0], n + m), 0.0)
        for i in range(self._big_b.shape[0]):
            for j in range(n):
                big_b[i, j] = self.big_b[i, j]

        self._big_b = big_b

    @property
    def a(self):
        return self._inst.a

    @property
    def b(self):
        return self._inst.b

    @property
    def c(self):
        return self._inst.c

    @property
    def lower_bounds(self):
        return self._inst.lower_bounds

    @property
    def upper_bounds(self):
        return self._inst.upper_bounds

    @property
    def big_b(self):
        return self._big_b

    @property
    def big_c(self):
        return self._big_c

    @property
    def sign(self):
        return self._inst.sign
