import random

import numpy as np
import pulp

from src.solver import inverse_lp
from src.structures import simple_instance
from src.lpp_generator import shortest_path_gen


def test1():
    """
    Здесь допустимая область - треугольник на плоскости, от которого чуть чуть отсечено ограницением 5 <= x2 <= 6
    """
    c = np.array([-3., -5.])
    a = np.array([[1., 1.],
                  [-3., -1.],
                  [1., -1.]])
    b = np.array([2., 4., -20])
    x0 = np.array([-3.1, 5.3])
    # доп. ограничения
    l, u = [-9., 5.], [-3., 14.]
    inst = simple_instance.LpInstance(a, b, c, l, u)

    solver = inverse_lp.InverseLpSolverL1()
    d = solver.solve(inst, x0)
    print("Result", d, ", norm", np.absolute(d - inst.c).sum())


def test2():
    # Здесь допустимая область - треугольник на плоскости
    c = np.array([-3., -5.])
    a = np.array([[1., 1.],
                  [-3., -1.],
                  [1., -1.]])
    b = np.array([2., 4., -20])
    x0 = np.array([-3.1, 5.3])

    inst = simple_instance.LpInstance(a, b, c)

    solver = inverse_lp.InverseLpSolverLInfinity()
    # Весовая функция
    w = np.array([1.0, 1.0])
    d = solver.solve(inst, x0, w)
    print("Result", d, ", norm", (np.absolute(d - inst.c) * w).max())


def test3():
    sp = shortest_path_gen.LPPShortestPath(10, 10)

    model = simple_instance.create_pulp_model(sp.lpp)

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != 1:
        raise ValueError("Status after model solving is False")

    x = simple_instance.get_x_after_model_solve(model)

    ban = np.full((1, x.shape[0]), 0)
    ban[0, list(x).index(0)] = 1

    new_a = np.concatenate((sp.lpp.a, ban), axis=0)
    new_b = np.concatenate((sp.lpp.b, np.array([1])))

    inst_2 = simple_instance.LpInstance(new_a, new_b, sp.lpp.c, sp.lpp.lower_bounds, sp.lpp.upper_bounds)
    model_2 = simple_instance.create_pulp_model(inst_2)

    status_2 = model_2.solve(pulp.PULP_CBC_CMD(msg=False))
    if status_2 != 1:
        raise ValueError("Status after model solving is False")

    x0 = simple_instance.get_x_after_model_solve(model_2)
    print("answ x =  ", x.dot(sp.lpp.c))
    print("answ x0 = ", x0.dot(sp.lpp.c))

    solver = inverse_lp.InverseLpSolverL1()

    d = solver.solve(sp.lpp, x0)
    print("Result", "[", ", ".join([str(i) for i in d]), "], norm", np.absolute(d - sp.lpp.c).sum())

    lpp = simple_instance.LpInstance(sp.lpp.a, sp.lpp.b, d)
    d0 = solver.solve(lpp, x0)
    print("Result", "[", ", ".join([str(i) for i in d]), "], norm", np.absolute(d - lpp.c).sum())


# test1()
# test2()
test3()
