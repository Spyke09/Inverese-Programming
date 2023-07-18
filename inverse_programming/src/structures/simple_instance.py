import typing as tp
from enum import Enum

import numpy as np


class LpSign(Enum):
    LessE = 0
    MoreE = 1
    Equal = 2


class InvLpInstance:
    """
    Класс для хранения ЗЛП вида.
    c.x -> min
    Ax `sign` b
    l <= x <= u
    """

    def __init__(self, a, b, c, sign: LpSign, lower_bounds=None, upper_bounds=None):
        self._c: np.array = np.array(c)
        self._a: np.array = np.array(a)
        self._b: np.array = np.array(b)
        self._upper_bounds: tp.Optional[np.array] = None
        self._lower_bounds: tp.Optional[np.array] = None
        self.sign: LpSign = sign
        if upper_bounds is not None:
            self._upper_bounds = np.array(upper_bounds)
        if lower_bounds is not None:
            self._lower_bounds = np.array(lower_bounds)

    def hide_upper_bounds(self):
        n, m = self.a.shape
        a = np.full((n + m, m), 0.0)
        b = np.full(n + m, 0.0)
        for i in range(n):
            for j in range(m):
                a = self.a[i, j]
            b = self.b[i]
        for i in range(n, n + m):
            a = self.a[i, i - n]
            b = self._upper_bounds[i - n]

        self._a = a
        self._b = b
        self._upper_bounds = None

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

    def check_feasible(self, x):
        l_bounds_q = (self.lower_bounds is None) or (x - self.lower_bounds >= 0).all()
        u_bounds_q = (self.upper_bounds is None) or (x - self.upper_bounds <= 0).all()

        if self.sign == LpSign.Equal:
            q = (self.a.dot(x) - self.b == 0).all()
        elif self.sign == LpSign.LessE:
            q = (self.a.dot(x) - self.b <= 0).all()
        else:
            q = (self.a.dot(x) - self.b >= 0).all()
        return q and l_bounds_q and u_bounds_q

    def convert(self, new_sign):
        if self.sign == LpSign.Equal:
            if new_sign != LpSign.Equal:
                n, m = self.a.shape
                a = np.full((n * 2, m), 0)
                b = np.full(n * 2, 0)
                for i in range(n):
                    b[i] = self.b[i]
                    b[n + i] = -self.b[i]
                    for j in range(m):
                        a[i, j] = self.a[i, j]
                        a[n + i, j] = -self.a[i, j]
                return InvLpInstance(a, b, self.c, new_sign, self.lower_bounds, self.upper_bounds)
            else:
                return InvLpInstance(self.a, self.b, self.c, self.sign, self.lower_bounds, self.upper_bounds)
        else:
            raise NotImplementedError("Not implemented convert.")
