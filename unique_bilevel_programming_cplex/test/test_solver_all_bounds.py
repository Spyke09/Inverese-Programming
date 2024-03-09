import numpy as np

from unique_bilevel_programming_cplex.src.model import Model
from unique_bilevel_programming_cplex.src.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.var_expr_con import Var
from unique_bilevel_programming_cplex.src.common import Sense


weights_1_0 = {"c": 1, "x": 1}
weights_1_1 = {"u": 1}
weights_1_2 = {"l": 1}
weights_1_3 = {"x": 1, "b": 20}


inst_ = Model()
x1, x2 = Var("x1"), Var("x2")

inst_.add_obj(x1.e + x2.e, Sense.MIN)

inst_.add_constr(x1 + x2 == 1)
inst_.add_constr(x1.e >= 0)
inst_.add_constr(x2.e >= 0)
# inst_.add_constr(x1.e <= 1)
# inst_.add_constr(x2.e <= 1)


def equal_q_f_p(a, b, eps=10e-7):
    return (np.abs(a - b) < eps).all()


def test_simple_instance_c():
    model = UBModel(inst_, eps=1e-2, big_m=1e2)
    model.init_c_as_var()
    model.set_x0({Var("x1"): 1, Var("x2"): 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 1) and equal_q_f_p(answer[x2], 0)
    assert answer[Var("c_x1")] != answer[Var("c_x2")]


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
    assert \
        equal_q_f_p(answer["x"][0], 1) and equal_q_f_p(answer["x"][1], 1) and equal_q_f_p(answer["b"][0], 2) or \
        equal_q_f_p(answer["x"][0], 0) and equal_q_f_p(answer["x"][1], 0) and equal_q_f_p(answer["b"][0], 0)
    assert "c" not in answer
    assert "l" not in answer
    assert "u" not in answer

