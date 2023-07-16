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

    c_r: np.array
    c_z: np.array
    d_r: np.array
    d_z: np.array

    a_r: np.array
    a_z: np.array
    b_r: np.array
    b_z: np.array
    r: np.array

    w_r: np.array
    w_z: np.array

    q_r: np.array
    q_z: np.array
    p_r: np.array
    p_z: np.array
    s: np.array

    m_r: int = dataclasses.field(init=False)
    m_z: int = dataclasses.field(init=False)
    n_r: int = dataclasses.field(init=False)
    n_z: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.m_r = self.c_r.shape[0]
        self.m_z = self.c_z.shape[0]
        self.n_r = self.d_r.shape[0]
        self.n_z = self.d_z.shape[0]


if __name__ == "__main__":
    print()
