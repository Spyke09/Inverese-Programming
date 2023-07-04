import numpy as np

from src.lpp_generator import min_cost_flow_gen
from src.lpp_generator import shortest_path_gen
from src.solver import bilevel_lp, tools
from src.structures import bilevel_instance
from src.structures import simple_instance


# декоратор для тестов, где могут получиться невыполнимые инстансы
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
    print("b и c нельзя менять")
    c = np.array([-3., -5.])
    a = np.array([[1., 1.]])
    b = np.array([2])
    x0 = np.array([1.21, 0.79])

    big_b = np.eye(b.shape[0])
    big_c = np.eye(c.shape[0])

    inst = bilevel_instance.BilevelInstance(a, big_b.dot(b), big_c.dot(c), big_b, big_c)

    solver = bilevel_lp.BilevelLpSolver()
    x, b, c = solver.solve(inst, x0)
    print("x = ", x)
    print("b = ", b)
    print("c = ", c, "\n")


def test2():
    """
    Здесь допустимая область - треугольник на плоскости
    """
    print("Test 2")
    print("b и c - любые")
    c = np.array([-3., -5.])
    a = np.array([[1., 1.]])
    b = np.array([2])
    x0 = np.array([1.21, 0.79])

    big_b = np.random.rand(0, b.shape[0])
    big_c = np.random.rand(0, c.shape[0])

    inst = bilevel_instance.BilevelInstance(a, big_b.dot(b), big_c.dot(c), big_b, big_c)

    solver = bilevel_lp.BilevelLpSolver()
    x, b, c = solver.solve(inst, x0)
    print("x = ", x)
    print("b = ", b)
    print("c = ", c, "\n")


def test3():
    """
    Здесь допустимая область - треугольник на плоскости
    """
    print("Test 2")
    print("Верхние границы (1.5, 1.5), b и c - нельзя менять")
    c = np.array([-3., -5.])
    a = np.array([[1., 1.]])
    b = np.array([2])
    x0 = np.array([1.21, 0.79])
    upper_bounds = np.array([1.5, 1.5])

    big_b = np.eye(b.shape[0])
    big_c = np.eye(c.shape[0])

    inst = bilevel_instance.BilevelInstance(a, big_b.dot(b), big_c.dot(c), big_b, big_c, upper_bounds)

    solver = bilevel_lp.BilevelLpSolver()
    x, b, c = solver.solve(inst, x0)
    print("x = ", x)
    print("b = ", b)
    print("c = ", c, "\n")


def test4():
    """
    Здесь допустимая область - треугольник на плоскости
    """
    print("Test 2")
    print("Верхние границы (1.5, 1.5), b и c - любые")
    c = np.array([-3., -5.])
    a = np.array([[1., 1.]])
    b = np.array([2])
    x0 = np.array([1.21, 0.79])
    upper_bounds = np.array([1.5, 1.5])

    big_b = np.random.rand(0, b.shape[0])
    big_c = np.random.rand(0, c.shape[0])

    inst = bilevel_instance.BilevelInstance(a, big_b.dot(b), big_c.dot(c), big_b, big_c, upper_bounds)

    solver = bilevel_lp.BilevelLpSolver()
    x, b, c = solver.solve(inst, x0)
    print("x = ", x)
    print("b = ", b)
    print("c = ", c, "\n")


@repeat_if_exception
def test5():
    print("Test 4")
    sp = min_cost_flow_gen.LPPMinCostFlow(10, 10)
    inst_1 = sp.lpp
    x_1 = tools.get_x_after_model_solve(inst_1)
    res_1 = x_1.dot(inst_1.c)
    print("Минимальное значение функции 1 = ", res_1)

    inst_2 = simple_instance.InvLpInstance(inst_1.a, inst_1.b, np.random.uniform(-1, 1, inst_1.c.shape[0]), inst_1.sign, inst_1.lower_bounds, inst_1.upper_bounds)
    x_2 = tools.get_x_after_model_solve(inst_2)
    res_2 = x_2.dot(inst_1.c)

    if res_2 == res_1:
        raise ValueError("res_1 == res_2")

    print("Минимальное значение функции 2 = ", res_2)

    x0, lpp = (x_1, inst_2) if res_1 > res_2 else (x_2, inst_1)
    x0, lpp = (x_1, inst_1) if res_1 > res_2 else (x_2, inst_2)

    b_inst = bilevel_instance.BilevelInstance(lpp.a, lpp.b, lpp.c, np.eye(lpp.b.shape[0]), np.eye(lpp.c.shape[0]), inst_1.upper_bounds)
    solver = bilevel_lp.BilevelLpSolver()
    x, b, c = solver.solve(b_inst, x0)

    print("Значение новой ЗЛП = c * x = ", x0.dot(c))

    if (c == 0).all():
        print("c = 0\n")
        return
    inst_3 = simple_instance.InvLpInstance(inst_1.a, b, c, inst_1.sign, inst_1.lower_bounds, inst_1.upper_bounds)
    x_3 = tools.get_x_after_model_solve(inst_3)
    print("Минмальное значение новой ЗЛП = ", c.dot(x_3))



test1()
test2()
test3()
test4()
test5()


