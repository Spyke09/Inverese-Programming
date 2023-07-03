import numpy as np

from src.lpp_generator import min_cost_flow_gen
from src.lpp_generator import shortest_path_gen
from src.solver import inverse_lp, tools
from src.structures import simple_instance


def test1():
    """
    Здесь допустимая область - треугольник на плоскости, от которого чуть чуть отсечено ограницением 5 <= x2 <= 6
    """
    print("Test 1")
    c = np.array([-3., -5.])
    a = np.array([[1., 1.],
                  [-3., -1.],
                  [1., -1.]])
    b = np.array([2., 4., -20])
    x0 = np.array([-3.1, 5.3])
    # доп. ограничения
    l, u = [-9., 5.], [-3., 14.]
    inst = simple_instance.InvLpInstance(a, b, c, simple_instance.LpSign.MoreE, l, u)

    solver = inverse_lp.InverseLpSolverL1()
    d = solver.solve(inst, x0)
    print("Result", d, ", norm", np.absolute(d - inst.c).sum(), "\n")


def test2():
    # Здесь допустимая область - треугольник на плоскости
    print("Test 2")
    c = np.array([-3., -5.])
    a = np.array([[1., 1.],
                  [-3., -1.],
                  [1., -1.]])
    b = np.array([2., 4., -20])
    x0 = np.array([-3.1, 5.3])

    inst = simple_instance.InvLpInstance(a, b, c, simple_instance.LpSign.MoreE)

    solver = inverse_lp.InverseLpSolverLInfinity()
    # Весовая функция
    w = np.array([1.0, 1.0])
    d = solver.solve(inst, x0, w)
    print("Result", d, ", norm", (np.absolute(d - inst.c) * w).max(), "\n")


def test3():
    print("Test 3")
    sp = shortest_path_gen.LPPShortestPath(100, 10)
    x = tools.get_x_after_model_solve(sp.lpp)

    ban = np.full((1, x.shape[0]), 0)
    ban[0, list(x).index(0)] = 1

    new_a = np.concatenate((sp.lpp.a, ban), axis=0)
    new_b = np.concatenate((sp.lpp.b, np.array([1])))

    inst_2 = simple_instance.InvLpInstance(new_a, new_b, sp.lpp.c, sp.lpp.sign, sp.lpp.lower_bounds, sp.lpp.upper_bounds)
    x0 = tools.get_x_after_model_solve(inst_2)

    print("Минимальное значение функции = ", x.dot(sp.lpp.c))
    print("Значение функции при x0 = ", x0.dot(sp.lpp.c))

    solver = inverse_lp.InverseLpSolverL1()

    d = solver.solve(sp.lpp, x0)
    print("Значение нормы = ", np.absolute(d - sp.lpp.c).sum(), "\n")
    print("Значение новой ЗЛП при новом d = d * x0 = ", d.dot(x0))

    lpp = simple_instance.InvLpInstance(sp.lpp.a, sp.lpp.b, d, sp.lpp.sign, sp.lpp.lower_bounds, sp.lpp.upper_bounds)
    x1 = tools.get_x_after_model_solve(lpp)
    print("Минмальное значение новой ЗЛП = ", d.dot(x1), "\n")



def test4():
    print("Test 4")
    sp = min_cost_flow_gen.LPPMinCostFlow(100, 10)

    x_1 = tools.get_x_after_model_solve(sp.lpp)
    res_1 = x_1.dot(sp.lpp.c)
    print("Минимальное значение функции 1 = ", res_1)

    inst_2 = simple_instance.InvLpInstance(sp.lpp.a, sp.lpp.b, np.random.uniform(-1, 1, sp.lpp.c.shape[0]), sp.lpp.sign, sp.lpp.lower_bounds, sp.lpp.upper_bounds)
    x_2 = tools.get_x_after_model_solve(inst_2)
    res_2 = x_2.dot(sp.lpp.c)

    if res_2 == res_1:
        raise ValueError("res_1 == res_2")

    print("Минимальное значение функции 2 = ", res_2)

    x0, lpp = (x_1, inst_2) if res_1 > res_2 else (x_2, sp.lpp)

    solver = inverse_lp.InverseLpSolverL1()
    d = solver.solve(lpp, x0)

    print("Значение нормы = ", np.absolute(d - lpp.c).sum(), "\n")

    print("Значение новой ЗЛП при новом d = d * x0 = ", x0.dot(d))

    inst_3 = simple_instance.InvLpInstance(sp.lpp.a, sp.lpp.b, d, sp.lpp.sign, sp.lpp.lower_bounds, sp.lpp.upper_bounds)
    x_3 = tools.get_x_after_model_solve(inst_3)
    print("Минмальное значение новой ЗЛП = ", d.dot(x_3))



test1()
test2()
test3()
test4()
