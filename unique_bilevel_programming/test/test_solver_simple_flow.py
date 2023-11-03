import numpy as np
import pytest

import MIBLP.src.tools
from unique_bilevel_programming.solver.unique_bilevel_solver import UBSolver
from unique_bilevel_programming.structures.unique_bilevel_instance import UBInstance

inst_1 = UBInstance(
    x0=[1, 0, 1, 1, 1],

    A=[[1, 1, 1, 0, 0], [1, 0, 0, -1, 0], [0, 0, 1, 0, -1]],
    c0=[1, 1, 1, 1, 1],
    b0=[2, 0, 0],
    l0=[0, 0, 0, 0, 0],
    u0=[1, 1, 1, 1, 1],
)

inst_2 = UBInstance(
    x0=[1, 0, 1, 1, 1],

    A=[[1, 1, 1, 0, 0], [1, 0, 0, -1, 0], [0, 0, 1, 0, -1], [0, 1, 0, 1, 1]],
    c0=[1, 1, 1, 1, 1],
    b0=[2, 0, 0, 2],
    l0=[0, 0, 0, 0, 0],
    u0=[1, 1, 1, 1, 1],
)

inst_3 = UBInstance(
    x0=[1, 0, 1, 1, 1],

    A=[[1, 1, 1, 0, 0], [1, 0, 0, -1, 0], [0, 0, 1, 0, -1], [0, 1, 0, 1, 1], [2, 1, 1, -1, 0]],
    c0=[1, 1, 1, 1, 1],
    b0=[2, 0, 0, 2, 2],
    l0=[0, 0, 0, 0, 0],
    u0=[1, 1, 1, 1, 1],
)


def equal_q_f_p(a, b, eps=10e-7):
    return (np.abs(a - b) < eps).all()


@pytest.mark.parametrize(
    "instance", [inst_1, inst_2, inst_3]
)
def test_simple_instance_c(instance):
    weights_1_0 = {"c": 1}
    solver = UBSolver(eps=1e-2, big_m=1e2)
    solver.solve(instance, weights_1_0, 7)
    answer = solver.get_values_by_names(weights_1_0.keys())
    c = answer["c"]

    assert (-c[3] - c[0] + c[1] > 0) and (-c[4] - c[2] + c[1] > 0)


@pytest.mark.parametrize(
    "instance", [inst_1, inst_2, inst_3]
)
def test_simple_instance_u(instance):
    weights_1_1 = {"u": 1}
    solver = UBSolver(eps=1e-2, big_m=1e2)
    solver.solve(instance, weights_1_1)

    answer = solver.get_values_by_names(weights_1_1.keys())
    u = answer["u"]
    assert u[1] == 0
    assert u[0] == u[2] == u[3] == u[4]


@pytest.mark.parametrize(
    "instance", [inst_1, inst_2, inst_3]
)
def test_simple_instance_l(instance):
    weights_1_1 = {"l": 1}
    solver = UBSolver(eps=1e-2, big_m=1e2)
    solver.solve(instance, weights_1_1)

    answer = solver.get_values_by_names(weights_1_1.keys())
    l = answer["l"]
    assert l[0] == l[2] == 1 or l[3] == l[4] == 1 or l[0] == l[4] == 1 or l[2] == l[3] == 1
