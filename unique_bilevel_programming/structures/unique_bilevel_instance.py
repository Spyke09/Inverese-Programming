from dataclasses import dataclass, field
import numpy as np
import typing as tp


ArrayType = np.array

@dataclass
class UBInstance:
    """
    Класс для хранения задачи Unique Bilevel Programming:
        Найти коэффициенты c, b, u, l, и x' такие, что задача
            {c.T * x -> min; Ax == b, l <= x <= u}
        имела бы единственный минимум при x',
        и должны быть выполнены равенства
        Bb == b_hat
        Cc == c_hat
        Uu == u_hat
        Ll == l_hat
        и взвешенная сумма отклонений
        {||x' - x0||, ||b - b0||, ||c - c0||, ||u - u0||, ||l - l0||}
        была минимальнымой
    """
    A: ArrayType = None
    x0: ArrayType = None

    c0: ArrayType = None
    b0: np.array = None
    l0: tp.Optional[ArrayType] = None
    u0: tp.Optional[ArrayType] = None

    C: tp.Optional[ArrayType] = None
    c_hat: tp.Optional[ArrayType] = None

    B: tp.Optional[ArrayType] = None
    b_hat: tp.Optional[ArrayType] = None

    L: tp.Optional[ArrayType] = None
    l_hat: tp.Optional[ArrayType] = None

    U: tp.Optional[ArrayType] = None
    u_hat: tp.Optional[ArrayType] = None

    def __post_init__(self):
        self.A = np.array(self.A, dtype=float)

        if self.x0 is not None:
            self.x0 = np.array(self.x0, dtype=float)

        if self.c0 is not None:
            self.c0 = np.array(self.c0, dtype=float)

        if self.b0 is not None:
            self.b0 = np.array(self.b0, dtype=float)

        if self.l0 is not None:
            self.l0 = np.array(self.l0, dtype=float)

        if self.u0 is not None:
            self.u0 = np.array(self.u0, dtype=float)

        if (self.C is None or self.c_hat is None) and self.c0 is not None:
            self.C = np.full((1, self.c0.shape[0]), 0.0)
            self.c_hat = np.full(1, 0.0)
        else:
            self.C = np.array(self.C, dtype=float)
            self.c_hat = np.array(self.c_hat, dtype=float)

        if (self.B is None or self.b_hat is None) and self.b0 is not None:
            self.B = np.full((1, self.b0.shape[0]), 0.0)
            self.b_hat = np.full(1, 0.0)
        else:
            self.B = np.array(self.B, dtype=float)
            self.b_hat = np.array(self.b_hat, dtype=float)

        if (self.L is None or self.l_hat is None) and self.l0 is not None:
            self.L = np.full((1, self.l0.shape[0]), 0.0)
            self.l_hat = np.full(1, 0.0)
        else:
            self.L = np.array(self.L, dtype=float)
            self.L_hat = np.array(self.l_hat, dtype=float)

        if (self.U is None or self.u_hat is None) and self.u0 is not None:
            self.U = np.full((1, self.u0.shape[0]), 0.0)
            self.u_hat = np.full(1, 0.0)
        else:
            self.U = np.array(self.U, dtype=float)
            self.u_hat = np.array(self.u_hat, dtype=float)

    @property
    def shape(self):
        return self.A.shape
