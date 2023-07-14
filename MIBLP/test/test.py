from MIBLP.src.mibpl_instance import MIBPLInstance
from MIBLP.src.solver import MIBLPSolver
import numpy as np


def toy_example_1_test():
    inst = MIBPLInstance(
        np.full(0, 0),
        np.array([-1]),
        np.full(0, 0),
        np.array([-10]),

        np.full((0, 0), 0),
        np.full((0, 1), 0),
        np.full((0, 0), 0),
        np.full((0, 1), 0),
        np.full(0, 0),

        np.full(0, 0),
        np.array([-1]),

        np.full((4, 0), 0),
        np.array([[20], [2], [-1], [-10]]),
        np.full((4, 0), 0),
        np.array([[-25], [1], [2], [-2]]),
        np.array([30, 10, 15, -15]),
    )

    solver = MIBLPSolver()
    solver.solve(inst)


toy_example_1_test()
