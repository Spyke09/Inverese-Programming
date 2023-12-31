import logging

import numpy as np

from inverse_programming.src.lpp_generator import min_cost_flow_gen
from inverse_programming.src.solver.unique_solution_solver import UniqueSolutionSolver
from inverse_programming.src.structures.inv_instance import InvLpInstance, LpSign
from inverse_programming.src import tools
logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S', level=logging.DEBUG)


def test1():
    eps = 1e-2
    inst = InvLpInstance(
        a=[[1, 1]],
        b=[1],
        c=[1, 1],
        sign=LpSign.Equal,
        # lower_bounds=[0, 0],
        upper_bounds=[1, 1],
    )
    solver = UniqueSolutionSolver()

    solver.solve(
        inst, [0.0, 1.0], [1, 0, 0], [0], eps=eps, big_m=1e2
    )
    answer = solver.get_values_by_names(["x", "c", "u", 'l'])
    print(answer)


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

    solver.solve(
        inst, [0.0, 1.0], [1, 0, 0], eps=10e-5, big_m=10e5
    )
    print(solver.get_values_by_names(["x", "c", "b", "u", 'l']))


def test3():
    inst = InvLpInstance(
        a=[[1, 1, 0, 0], [0, 0, 1, 1]],
        b=[1, 1],
        c=[1, 1, 1, 1],
        sign=LpSign.Equal,
        lower_bounds=[0, 0, 0, 0],
    )
    solver = UniqueSolutionSolver()

    solver.solve(
        inst, [1, 0, 1, 0], [0, 1, 0], [0, 1, 2, 3], eps=1e-2, big_m=1e2
    )
    print(solver.get_values_by_names(["x", "c", "b", "u", 'l']))


def test4():
    sp = min_cost_flow_gen.LPPMinCostFlow(20, 10)
    inst = sp.lpp

    solver = UniqueSolutionSolver()

    solver.solve(
        inst, np.random.uniform(0, 10 * 20, sp.edges_number()), [1, 1, 1], [10], eps=10e-5, big_m=10e5
    )
    print(solver.get_values_by_names(["x", "c", "b", "u", 'l']))


# test1()
# test2()
test3()
# test4()
