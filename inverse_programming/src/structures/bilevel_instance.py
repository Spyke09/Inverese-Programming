import numpy as np

from inverse_programming.src.structures.inv_instance import LpSign
from inverse_programming.src.structures import inv_instance


class BilevelInstance:
    def __init__(self, a, b, c, big_b, big_c, upper_bounds=None):
        n, m = a.shape
        self._inst = inv_instance.InvLpInstance(a, b, c, LpSign.Equal, np.full(m, 0), upper_bounds)
        self._big_b = big_b
        self._big_c = big_c

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

    def hide_upper_bounds(self):
        n, m = self.a.shape
        a = np.full((n + m, 2 * m), 0.0)
        for i in range(n):
            for j in range(m):
                a[i, j] = self.a[i, j]

        for i in range(m):
            a[i + n, i] = 1.0
            a[i + n, m + i] = 1.0

        big_b = np.full((self._big_b.shape[0] + m, n + m), 0.0)
        for i in range(self._big_b.shape[0]):
            for j in range(self._big_b.shape[1]):
                big_b[i, j] = self._big_b[i, j]
        for i in range(m):
            big_b[self._big_b.shape[0] + i, n + i] = 1.0

        b = np.concatenate([self.b, self.upper_bounds])

        big_c = np.full((self._big_c.shape[0] + m, 2 * m), 0.0)
        for i in range(self._big_c.shape[0]):
            for j in range(self._big_c.shape[1]):
                big_c[i, j] = self._big_c[i, j]
        for i in range(m):
            big_c[self._big_c.shape[0] + i, m + i] = 1.0

        c = np.concatenate([self.c, np.full(m, 0.0)])

        self._inst._a = a
        self._inst._c = c
        self._inst._b = b
        self._big_c = big_c
        self._big_b = big_b
        self._inst._upper_bounds = None
        self._inst._lower_bounds = np.full(self.a.shape[1], 0.0)
