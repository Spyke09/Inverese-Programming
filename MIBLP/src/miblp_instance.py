import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass
class MIBPLInstance:
    """
    Class that stores the parameters of the next problem:

    min(c_r * x_u + c_z * y_u + d_r * x_l0 + d_z * y_l0, {x_u, y_u, x_l0, y_l0})
      s.t.  a_r * x_u + a_z * y_u + b_r * x_l0 + b_z * y_l0 <= r
            x_u - continuous positive vector of size m_r
            y_u - integer positive vector of size m_z
            x_l0 - continuous positive vector of size n_r
            y_l0 - integer positive vector of size n_z

            (x_l0, y_l0) ∈ argmax{  w_r * x_l + w_z * y_l :
                                    q_r * x_u + q_z * y_u  + p_r * x_l + p_z * y_l <= r
                                    x_l - continuous positive vector of size n_r
                                    y_l - integer positive vector of size n_z

    """

    c_r: np.array = None
    c_z: np.array = None
    d_r: np.array = None
    d_z: np.array = None

    a_r: np.array = None
    a_z: np.array = None
    b_r: np.array = None
    b_z: np.array = None
    r: np.array = None

    w_r: np.array = None
    w_z: np.array = None

    q_r: np.array = None
    q_z: np.array = None
    p_r: np.array = None
    p_z: np.array = None
    s: np.array = None

    m_r: int = dataclasses.field(init=False)
    m_z: int = dataclasses.field(init=False)
    n_r: int = dataclasses.field(init=False)
    n_z: int = dataclasses.field(init=False)

    @staticmethod
    def __shape(obj, s):
        return obj.shape[s] if obj is not None else 0

    def __post_init__(self):
        self.m_r = max(self.__shape(self.a_r, 1), self.__shape(self.q_r, 1), self.__shape(self.c_r, 0))
        self.m_z = max(self.__shape(self.a_z, 1), self.__shape(self.q_z, 1), self.__shape(self.c_z, 0))
        self.n_r = max(
            self.__shape(self.b_r, 1),
            self.__shape(self.p_r, 1),
            self.__shape(self.d_r, 0),
            self.__shape(self.w_r, 0))

        self.n_z = max(
            self.__shape(self.b_z, 1),
            self.__shape(self.p_z, 1),
            self.__shape(self.d_z, 0),
            self.__shape(self.w_z, 0))

        self.n_u = max(
            self.__shape(self.a_r, 0),
            self.__shape(self.a_z, 0),
            self.__shape(self.b_r, 0),
            self.__shape(self.b_z, 0),
            self.__shape(self.r, 0), 1)

        self.n_l = max(
            self.__shape(self.p_r, 0),
            self.__shape(self.p_z, 0),
            self.__shape(self.q_r, 0),
            self.__shape(self.q_z, 0),
            self.__shape(self.s, 0), 1)

        if self.c_r is None:
            self.c_r = np.full(self.m_r, 0.0)
        if self.c_z is None:
            self.c_z = np.full(self.m_z, 0.0)
        if self.d_r is None:
            self.d_r = np.full(self.n_r, 0.0)
        if self.d_z is None:
            self.d_z = np.full(self.n_z, 0.0)

        if self.a_r is None:
            self.a_r = np.full((self.n_u, self.m_r), 0.0)
        if self.a_z is None:
            self.a_z = np.full((self.n_u, self.m_z), 0.0)
        if self.b_r is None:
            self.b_r = np.full((self.n_u, self.n_r), 0.0)
        if self.b_z is None:
            self.b_z = np.full((self.n_u, self.n_z), 0.0)
        if self.r is None:
            self.r = np.full(self.n_u, 0.0)

        if self.w_r is None:
            self.w_r = np.full(self.n_r, 0.0)
        if self.w_z is None:
            self.w_z = np.full(self.n_z, 0.0)

        if self.q_r is None:
            self.q_r = np.full((self.n_l, self.m_r), 0.0)
        if self.q_z is None:
            self.q_z = np.full((self.n_l, self.m_z), 0.0)
        if self.p_r is None:
            self.p_r = np.full((self.n_l, self.n_r), 0.0)
        if self.p_z is None:
            self.p_z = np.full((self.n_l, self.n_z), 0.0)
        if self.s is None:
            self.s = np.full(self.n_l, 0.0)


if __name__ == "__main__":
    print()
