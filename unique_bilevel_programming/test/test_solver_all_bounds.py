import pytest
from unique_bilevel_programming.solver.unique_bilevel_solver import UBSolver
from unique_bilevel_programming.structures.unique_bilevel_instance import UBInstance
import numpy as np
from MIBLP.src import tools

inst_1 = UBInstance(
    x0=[0, 1],

    A=[[1, 1]],
    c0=[1, 1],
    b0=[1],
    l0=[0, 0],
    u0=[1, 1],
)
weights_1_0 = {"c": 1, "x": 1}
weights_1_1 = {"u": 1}
weights_1_2 = {"l": 1}
weights_1_3 = {"x": 1, "b": 20}
weights_1_4 = {"x": 20, "b": 1}


def equal_q_f_p(a, b, eps=10e-7):
    return (np.abs(a - b) < eps).all()


def test_simple_instance_c():
    solver = UBSolver(eps=1e-2, big_m=1e2)
    solver.solve(inst_1, weights_1_0)
    answer = solver.get_values_by_names(weights_1_0.keys())

    assert equal_q_f_p(answer["x"], inst_1.x0)
    assert answer["c"][0] != answer["c"][1]
    assert "b" not in answer
    assert "l" not in answer
    assert "u" not in answer


def test_simple_instance_u():
    solver = UBSolver(eps=10e-2, big_m=10e2)
    solver.solve(inst_1, weights_1_1)

    answer = solver.get_values_by_names(weights_1_1.keys())
    assert "x" not in answer
    assert "c" not in answer
    assert equal_q_f_p(answer["u"][0], 0) and equal_q_f_p(answer["u"][1], 1)
    assert "b" not in answer
    assert "l" not in answer


def test_simple_instance_l():
    solver = UBSolver(eps=10e-2, big_m=10e2)
    solver.solve(inst_1, weights_1_2)

    answer = solver.get_values_by_names(weights_1_2.keys())
    assert "x" not in answer
    assert "c" not in answer
    assert "b" not in answer
    assert "u" not in answer
    assert equal_q_f_p(answer["l"][0], 0) and equal_q_f_p(answer["l"][1], 1)


def test_simple_instance_b():
    solver = UBSolver(eps=10e-2, big_m=10e2)
    solver.solve(inst_1, weights_1_3)

    answer = solver.get_values_by_names(weights_1_3.keys())
    assert equal_q_f_p(answer["x"][0], 0) and equal_q_f_p(answer["x"][1], 0)
    assert "c" not in answer
    assert equal_q_f_p(answer["b"][0], 0)
    assert "l" not in answer
    assert "u" not in answer


def test_simple_instance_b_2():
    solver = UBSolver(eps=10e-2, big_m=10e2)
    solver.solve(inst_1, weights_1_4)

    answer = solver.get_values_by_names(weights_1_4.keys())
    assert equal_q_f_p(answer["x"][0], 1) and equal_q_f_p(answer["x"][1], 1)
    assert "c" not in answer
    assert equal_q_f_p(answer["b"][0], 2)
    assert "l" not in answer
    assert "u" not in answer
