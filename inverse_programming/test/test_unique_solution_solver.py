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

    x, c, y, phi, lam = solver.solve(
        inst, [0.0, 1.0], eps=10e-5, big_m=10e5
    )
    print(f"x = {x}\ny = {y}\nc = {c}\nc-ATy = {inst.c - inst.a.T @ y}\nphi = {phi}\nlam = {lam}")


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

    x, c, y, phi, lam = solver.solve(
        inst, [0.5, 1.1], eps=10e-5, big_m=10e5
    )
    print(f"x = {x}\ny = {y}\nc = {c}\nc-ATy = {inst.c - inst.a.T @ y}\nphi = {phi}\nlam = {lam}")


def test1_prikol():
    inst = InvLpInstance(
        a=[[1, 1]],
        b=[1],
        c=[1, 1],
        sign=LpSign.Equal,
        lower_bounds=[0, 0],
    )
    solver = UniqueSolutionSolver()

    x, c, y, phi, lam = solver.solve(
        inst, [0.5, 0.5], eps=10e-2, big_m=10e3
    )
    print(f"x = {x}\ny = {y}\nc = {c}\nc-ATy = {inst.c - inst.a.T @ y}\nphi = {phi}\nlam = {lam}")


def test3():
    sp = min_cost_flow_gen.LPPMinCostFlow(200, 100)
    inst = sp.lpp

    n, m = inst.a.shape

    solver = UniqueSolutionSolver()

    x, c, y, phi, lam = solver.solve(
        inst, np.full(m, 0.0), eps=10e-6, big_m=10e20
    )

    print(f"x = {x}\ny = {y}\nc = {c}\nc-ATy = {inst.c - inst.a.T @ y}\nphi = {phi}\nlam = {lam}")


# test1()
# test1_prikol()
# test2()
test3()
