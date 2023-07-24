import typing as tp
from enum import Enum

import numpy as np
import scipy.sparse


class LpSign(Enum):
    LessE = 0
    MoreE = 1
    Equal = 2


LPVector = scipy.sparse.csc_array
LPMatrix = scipy.sparse.csc_array
LPValue = np.float64


def LPVectorAll(vector):
    q = True
    for i in range(vector.shape[0]):
        for j in range(vector.shape[1]):
            q &= vector[i, j]
    return q


def LPVectorAny(vector):
    q = False
    for i in range(vector.shape[0]):
        for j in range(vector.shape[1]):
            q |= vector[i, j]
    return q

class InvLpInstance:
    """
    Класс для хранения ЗЛП вида.
    c.x -> min
    Ax `sign` b
    l <= x <= u
    """

    def __init__(self, a, b, c, sign: LpSign, lower_bounds=None, upper_bounds=None):
        self._c: LPVector = LPVector(c)
        self._a: LPMatrix = LPMatrix(a)
        self._b: LPVector = LPVector(b)
        self._upper_bounds: tp.Optional[np.array] = None
        self._lower_bounds: tp.Optional[np.array] = None
        self.sign: LpSign = sign
        if upper_bounds is not None:
            self._upper_bounds = LPVector(upper_bounds)
        if lower_bounds is not None:
            self._lower_bounds = LPVector(lower_bounds)

    @property
    def a(self) -> LPMatrix:
        return self._a

    @property
    def b(self) -> LPVector:
        return self._b

    @property
    def c(self) -> LPVector:
        return self._c

    @property
    def upper_bounds(self) -> LPVector:
        return self._upper_bounds

    @property
    def lower_bounds(self) -> LPVector:
        return self._lower_bounds

    def check_feasible(self, x):
        a = x > self.lower_bounds
        l_bounds_q = (self.lower_bounds is None) or LPVectorAll(x >= self.lower_bounds)
        u_bounds_q = (self.upper_bounds is None) or LPVectorAll(x <= self.upper_bounds)

        if self.sign == LpSign.Equal:
            q = LPVectorAll(self.a @ x.T == self.b.T)
        elif self.sign == LpSign.LessE:
            q = LPVectorAll(self.a @ x.T <= self.b.T)
        else:
            q = LPVectorAll(self.a @ x.T >= self.b.T)
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
