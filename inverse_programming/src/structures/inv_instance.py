import typing as tp
from enum import Enum

import numpy as np
import scipy.sparse


class LpSign(Enum):
    LessE = 0
    MoreE = 1
    Equal = 2


BaseLPArray = scipy.sparse.csr_array
LPValue = scipy.float64


class LPArray(BaseLPArray):
    def all(self):
        return self.min()

    def any(self):
        return self.max()

    def l1_norm(self):
        return self.abs().sum()

    def l_inf_norm(self):
        return self.abs().max()

    def abs(self):
        return self.sign() * self

    def concatenate(self, other):
        n1, m1 = self.shape
        n2, m2 = other.shape
        idx = (lambda _i: (0, _i)) if n1 == n2 == 1 else (lambda _i: (_i, 0))
        h1, h2 = (m1, m2) if n1 == n2 == 1 else (n1, n2)
        if n1 == n2 == 1 or m1 == m2 == 1:
            temp = LPArray((1, h1 + h2))
            for i in range(h1):
                if self[idx(i)] != 0:
                    temp[idx(i)] = self[idx(i)]

            for i in range(h2):
                if other[idx(i)] != 0:
                    temp[idx(i) + h1] = other[idx(i)]

            return temp
        else:
            raise NotImplementedError("Not implemented for matrix.")

    def to_list(self, reduce_dimension=True):
        if self.shape[0] == 1 and reduce_dimension:
            return [self[0, j] for j in range(self.shape[1])]
        else:
            return [[self[i, j] for j in range(self.shape[1])] for i in range(self.shape[0])]

    def to_np_array(self, reduce_dimension=True):
        if self.shape[0] == 1 and reduce_dimension:
            return self.toarray()[0]
        else:
            return self.toarray()

    def dot(self, other):
        if self.shape[0] == other.shape[1] == 1:
            return super().dot(other)[0, 0]
        else:
            return super().dot(other)


class InvLpInstance:
    """
    Класс для хранения ЗЛП вида.
    c.x -> min
    Ax `sign` b
    l <= x <= u
    """

    def __init__(self, a, b, c, sign: LpSign, lower_bounds=None, upper_bounds=None):
        self._c: LPArray = LPArray(c)
        self._a: LPArray = LPArray(a)
        self._b: LPArray = LPArray(b)
        self._upper_bounds: tp.Optional[np.array] = None
        self._lower_bounds: tp.Optional[np.array] = None

        self.sign: LpSign = sign
        if upper_bounds is not None:
            self._upper_bounds = LPArray(upper_bounds)
        if lower_bounds is not None:
            self._lower_bounds = LPArray(lower_bounds)

        self.eliminate_zeros()

    def eliminate_zeros(self):
        self._c: LPArray.eliminate_zeros()
        self._a: LPArray.eliminate_zeros()
        self._b: LPArray.eliminate_zeros()
        if self.upper_bounds is not None:
            self._upper_bounds.eliminate_zeros()
        if self.lower_bounds is not None:
            self._lower_bounds.eliminate_zeros()

    @property
    def a(self) -> LPArray:
        return self._a

    @property
    def b(self) -> LPArray:
        return self._b

    @property
    def c(self) -> LPArray:
        return self._c

    @property
    def upper_bounds(self) -> LPArray:
        return self._upper_bounds

    @property
    def lower_bounds(self) -> LPArray:
        return self._lower_bounds

    def check_feasible(self, x: LPArray):
        a = x > self.lower_bounds
        l_bounds_q = (self.lower_bounds is None) or not (x < self.lower_bounds).any()
        u_bounds_q = (self.upper_bounds is None) or not (x > self.upper_bounds).any()

        if self.sign == LpSign.Equal:
            q = (self.a @ x.T == self.b.T).all()
        elif self.sign == LpSign.LessE:
            q = (self.a @ x.T <= self.b.T).all()
        else:
            q = (self.a @ x.T >= self.b.T).all()
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
