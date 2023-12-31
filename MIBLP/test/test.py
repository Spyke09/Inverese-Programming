from MIBLP.src.miblp_instance import MIBPLInstance
from MIBLP.src.miblp_solver import MIBLPSolver
import numpy as np
import logging

logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S', level=logging.DEBUG)


logger = logging.getLogger("TestMIBPLSolver")


def toy_example_1_test():
    logger.info("Toy example 1 Test")
    inst = MIBPLInstance(
        None,
        np.array([-1]),
        None,
        np.array([-10]),
        None, None, None, None, None,
        None,
        np.array([-1]),
        None,
        np.array([[-25], [1], [2], [-2]]),
        None,
        np.array([[20], [2], [-1], [-10]]),
        np.array([30, 10, 15, -15]),
    )

    solver = MIBLPSolver()
    x_u, y_u, x_l0, y_l0 = solver.solve(inst)
    assert x_u.shape[0] == 0 and y_u == np.array([2.]) and x_l0.shape[0] == 0, y_l0 == np.array([2.])


def toy_example_2_test():
    logger.info("Toy example 2 Test")
    inst = MIBPLInstance(
        np.full(0, 0),
        np.array([-1]),
        np.full(0, 0),
        np.array([-2]),

        np.full((2, 0), 0),
        np.array([[-2], [1]]),
        np.full((2, 0), 0),
        np.array([[3], [1]]),
        np.array([12, 14]),

        np.full(0, 0),
        np.array([1]),

        np.full((2, 0), 0),
        np.array([[-3], [3]]),
        np.full((2, 0), 0),
        np.array([[1], [1]]),
        np.array([-3, 30]),
    )

    solver = MIBLPSolver()
    x_u, y_u, x_l0, y_l0 = solver.solve(inst)
    assert x_u.shape[0] == 0 and y_u == np.array([8.]) and x_l0.shape[0] == 0, y_l0 == np.array([6.])


def toy_example_3_test():
    logger.info("Toy example 3 Test")
    inst = MIBPLInstance(
        np.array([20.0]),
        np.array([-38.0]),
        np.array([1.0]),
        np.array([42.0]),

        np.array([[0.0], [6.0]]),
        np.array([[7.0], [9.0]]),
        np.array([[5.0], [10.0]]),
        np.array([[7.0], [2.0]]),
        np.array([62.0, 117.0]),

        np.array([39.0]),
        np.array([27.0]),

        np.array([[8.0], [9.0]]),
        np.array([[0.0], [0.0]]),
        np.array([[2.0], [2.0]]),
        np.array([[8.0], [1.0]]),
        np.array([53.0, 28.0]),
    )

    solver = MIBLPSolver()
    x_u, y_u, x_l0, y_l0 = solver.solve(inst)
    assert x_u == np.array([28./9.]) and y_u == np.array([8.]) and x_l0 == np.array([0.]), y_l0 == np.array([0.])


# toy_example_1_test()
# toy_example_2_test()
toy_example_3_test()
