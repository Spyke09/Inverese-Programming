import numpy as np
import pulp
from src import simple_instance
from src import inverse_lp


def test1():
    c = [-3, -5, 3]
    a = np.array([[1, 1, 0],
                  [-3, -1, 0],
                  [1, -1, 1]])
    b = [2, 4, 5]
    inst = simple_instance.LpInstance(a, b, c, [-3, 5, 13], [10, 5, 20])
    # inst = simple_instance.LpInstance(a, b, c)
    solver = inverse_lp.InverseLpSolver(inverse_lp.NormType.L1)
    result = solver.solve(inst, [-3.1,  5.1, 13.1])
    print(result)


test1()
