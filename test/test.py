import numpy as np
import pulp
from src import simple_instance
from src import inverse_lp


def test1():
    c = np.array([-3., -5.])
    a = np.array([[1., 1.],
                  [-3., -1.],
                  [1., -1.]])
    b = np.array([2., 4., -20])
    x0 = np.array([-3.1, 5.3])

    # inst = simple_instance.LpInstance(a, b, c, [-4, 5], [-4, 6.])
    inst = simple_instance.LpInstance(a, b, c)

    solver1 = inverse_lp.InverseLpSolverL1()
    d1 = solver1.solve(inst, x0)
    print("Result", d1, ", norm", np.absolute(d1 - inst.c).sum())

    solver2 = inverse_lp.InverseLpSolverLInfinity()
    d2 = solver2.solve(inst, x0)
    print("Result", d2, ", norm", np.absolute(d2 - inst.c).max())


test1()
