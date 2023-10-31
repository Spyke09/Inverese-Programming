from dataclasses import dataclass
import numpy as np
import typing as tp


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
    x0: np.array

    A: np.array

    c0: np.array
    C: tp.Optional[np.array]
    c_hat: tp.Optional[np.array]

    b0: np.array
    B: tp.Optional[np.array]
    b_hat: tp.Optional[np.array]

    l0: tp.Optional[np.array]
    L: tp.Optional[np.array]
    l_hat: tp.Optional[np.array]

    u0: tp.Optional[np.array]
    U: tp.Optional[np.array]
    u_hat: tp.Optional[np.array]
