import logging

import numpy as np

from inverse_programming.src.lpp_generator import min_cost_flow_gen
from inverse_programming.src.solver import tools, min_max_dist_bilevel_lp
from inverse_programming.src.structures import bilevel_instance, inv_instance

logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S', level=logging.DEBUG)


def repeat_if_exception(test):
    def wrapper():
        while True:
            try:
                test()
            except:
                print("\tWrapper: Try againg")
            else:
                break

    return wrapper


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

    solver = min_max_dist_bilevel_lp.MinMaxDistBilevelLpSolver()
    x, b, c = solver.solve(inst, x0)
    print("x = ", x)
    print("b = ", b)
    print("c = ", c, "\n")


def test2():
    print("Test 2")
    sp = min_cost_flow_gen.LPPMinCostFlow(3, 3)
    inst_1 = sp.lpp
    x_1 = tools.get_x_after_model_solve(inst_1)
    res_1 = x_1.dot(inst_1.c)
    print("Минимальное значение функции 1 = ", res_1)

    x0 = x_1 + np.random.random(x_1.shape)

    b_inst = bilevel_instance.BilevelInstance(
        inst_1.a,
        inst_1.b,
        inst_1.c,
        np.eye(inst_1.b.shape[0]),
        np.eye(inst_1.c.shape[0]),
        inst_1.upper_bounds)

    b_inst.hide_upper_bounds()
    x0 = np.concatenate([x0, np.full(x0.shape[0], 0.0)])

    solver = min_max_dist_bilevel_lp.MinMaxDistBilevelLpSolver()
    x, b, c = solver.solve(b_inst, x0)

    print("Значение новой ЗЛП = c * x = ", x0.dot(c))

    if (c == 0).all():
        print("c = 0\n")
        return
    inst_3 = inv_instance.InvLpInstance(inst_1.a, b, c, inst_1.sign, inst_1.lower_bounds, inst_1.upper_bounds)
    x_3 = tools.get_x_after_model_solve(inst_3)
    print("Минмальное значение новой ЗЛП = ", c.dot(x_3))


# test1()
test2()
