import logging

import numpy as np

from inverse_programming.src.solver import bilevel_lp
from inverse_programming.src.structures import bilevel_instance

logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S', level=logging.DEBUG)


def test1():
    """
    Здесь допустимая область - треугольник на плоскости
    """
    print("Test 1")

    c = np.array([-3., -5.])
    a = np.array([[1., 1.]])
    b = np.array([2])
    x0 = np.array([1.21, 0.79])

    big_b = np.random.rand(1, b.shape[0])
    big_c = np.random.rand(1, c.shape[0])

    print(f"b и c - нельзя менять, x0 = {x0}")

    inst = bilevel_instance.BilevelInstance(a, big_b.dot(b), big_c.dot(c), big_b, big_c)

    solver = bilevel_lp.MinMaxDistBilevelLpSolver()
    x, b, c = solver.solve(inst, x0)
    print("x = ", x)
    print("b = ", b)
    print("c = ", c, "\n")


test1()