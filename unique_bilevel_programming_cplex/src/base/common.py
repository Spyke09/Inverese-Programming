import typing as tp
from enum import Enum

import numpy as np

LPFloat = np.float64
LPNan = np.nan
Integral = (int, float, LPFloat)
tpIntegral = tp.Union[int, float, LPFloat]
LPVector = list
LPVectorT = tp.List


def is_lp_nan(num):
    return np.isnan(num)


class VarType(Enum):
    REAL = 0
    INTEGER = 1
    BIN = 2


class Sign(Enum):
    L_EQUAL = 0
    G_EQUAL = 1
    EQUAL = 2

    @property
    def to_str(self) -> str:
        return {0: "<=", 1: ">=", 2: "=="}[self.value]

    def __call__(self, a, b):
        if self == Sign.EQUAL:
            return a == b
        if self == Sign.G_EQUAL:
            return a >= b
        if self == Sign.L_EQUAL:
            return a <= b


class Sense(Enum):
    MAX = 0
    MIN = 1

    @property
    def to_str(self) -> str:
        return {0: "-> max", 1: "-> min"}[self.value]


class Var:
    pass


class LinExpr:
    pass


LPEntity = tp.Union[LPFloat, Var, LinExpr, int, float]
