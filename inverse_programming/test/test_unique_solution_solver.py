import logging

import numpy as np

from inverse_programming.src.lpp_generator import min_cost_flow_gen
from inverse_programming.src.solver.unique_solution_solver import UniqueSolutionSolver
from inverse_programming.src.structures.inv_instance import InvLpInstance, LpSign

logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S', level=logging.DEBUG)


def test1():
    inst = InvLpInstance(
        a=[[1, 1]],
        b=[1],
        c=[1, 1],
        sign=LpSign.Equal,
        lower_bounds=[0, 0],
    )
    solver = UniqueSolutionSolver()

    x, c = solver.solve(
        inst, [0, 1], eps=10e-6, big_m=10e20
    )
    print((x, c))

    # x == [0.0, 1.0]
    # c == [0.999999999923855, 0.9999899999238551]


def test2():
    sp = min_cost_flow_gen.LPPMinCostFlow(100, 50)
    inst = sp.lpp
    inst.hide_upper_bounds()

    n, m = inst.a.shape

    solver = UniqueSolutionSolver()

    x, c = solver.solve(
        inst, np.full(m, 0.0), eps=10e-6, big_m=10e20
    )
    print((x, c))


# test1()
test2()
