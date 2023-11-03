import numpy as np
import pytest

import MIBLP.src.tools
from unique_bilevel_programming.solver.unique_bilevel_solver import UBSolver
from unique_bilevel_programming.structures.unique_bilevel_instance import UBInstance


inst_1 = UBInstance(
    x0=[0, 1, 0, 1],

    A=[[1, 1, 0, 0], [0, 0, 1, 1]],
    c0=[1, 1, 1, 1],
    b0=[1, 1],
    l0=[0, 0, 0, 0],
    u0=[1, 1, 1, 1],
)

inst_2 = UBInstance(
    x0=[0, 1, 0, 1, 0, 1],

    A=[[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]],
    c0=[1, 1, 1, 1, 1, 1],
    b0=[1, 1, 1],
    l0=[0, 0, 0, 0, 0, 0],
    u0=[1, 1, 1, 1, 1, 1],
)


@pytest.mark.parametrize(
    "instance", [inst_1, inst_2]
)
def test_simple_instance_c(instance):
    weights_1_0 = {"c": 1}
    solver = UBSolver(eps=1e-2, big_m=1e2)
    solver.solve(instance, weights_1_0)
    answer = solver.get_values_by_names(weights_1_0.keys())
    c = answer["c"]
    n = c.shape[0]
    assert all(c[i] != c[i + 1] for i in range(0, n, 2))


@pytest.mark.parametrize(
    "instance", [inst_1, inst_2]
)
def test_simple_instance_u(instance):
    weights_1_0 = {"u": 1}
    solver = UBSolver(eps=1e-2, big_m=1e2)
    solver.solve(instance, weights_1_0)
    answer = solver.get_values_by_names(weights_1_0.keys())
    u = answer["u"]

    assert u[0] == 0 and u[2] == 0


@pytest.mark.parametrize(
    "instance", [inst_1, inst_2]
)
def test_simple_instance_l(instance):
    weights_1_0 = {"l": 1}
    solver = UBSolver(eps=1e-2, big_m=1e2)
    solver.solve(instance, weights_1_0)
    answer = solver.get_values_by_names(weights_1_0.keys())
    l = answer['l']

    assert l[1] == 1 and l[3] == 1


@pytest.mark.parametrize(
    "instance", [inst_1, inst_2]
)
def test_simple_instance_b(instance):
    weights_1_0 = {"x": 1000, "b": 1}
    solver = UBSolver(eps=1e-2, big_m=1e2)
    solver.solve(instance, weights_1_0)
    answer = solver.get_values_by_names(weights_1_0.keys())
    b = answer['b']
    x = answer["x"]
    assert (np.all(x == 0) and b[0] == 0 and b[1] == 0) or (np.all(x == 1) and b[0] == 2 and b[1] == 2)

