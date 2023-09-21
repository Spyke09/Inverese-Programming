import logging

import numpy as np

from inverse_programming.src.lpp_generator import min_cost_flow_gen
from inverse_programming.src.solver.unique_solution_solver import UniqueSolutionSolver
from inverse_programming.src.structures.inv_instance import InvLpInstance, LpSign
from inverse_programming.src import tools
logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S', level=logging.DEBUG)


def test1():
    inst = InvLpInstance(
        a=[[1, 1]],
        b=[1],
        c=[1, 1],
        sign=LpSign.Equal,
        lower_bounds=[0, 0],
        upper_bounds=[1, 1],
    )
    solver = UniqueSolutionSolver()

    res = solver.solve(
        inst, [0.0, 1.0], [1, 1, 1], eps=10e-2, big_m=10e2
    )
    print(res)
    new_inst = InvLpInstance(
        inst.a,
        inst.b,
        res["c"] if "c" in res else inst.c,
        LpSign.Equal,
        res["l"] if "l" in res else inst.lower_bounds,
        res["u"] if "u" in res else inst.upper_bounds,
    )
    print("Optimal: ", tools.check_optimal_q(new_inst, res))
    print("Unique optimal: ", tools.check_optimal_unique_q(new_inst, res))


def test2():
    inst = InvLpInstance(
        a=[[1, 1]],
        b=[1.5],
        c=[1, 1],
        sign=LpSign.Equal,
        lower_bounds=[0, 0],
        upper_bounds=[1, 1],
    )
    solver = UniqueSolutionSolver()

    res = solver.solve(
        inst, [0.0, 1.0], [1, 0, 0], eps=10e-5, big_m=10e5
    )
    print(res)


def test3():
    sp = min_cost_flow_gen.LPPMinCostFlow(200, 100)
    inst = sp.lpp

    n, m = inst.a.shape

    solver = UniqueSolutionSolver()

    res = solver.solve(
        inst, [0.0, 1.0], [1, 0, 0], eps=10e-5, big_m=10e5
    )
    print(res)




test1()
# test2()
# test3()
