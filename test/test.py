import numpy as np

from src import inverse_lp
from src import simple_instance


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
    l, u = [-9., 5.], [-3., 6.]
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


test1()
test2()
