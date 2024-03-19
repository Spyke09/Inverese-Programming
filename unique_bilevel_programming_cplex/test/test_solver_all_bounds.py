import numpy as np

from unique_bilevel_programming_cplex.src.base.model import Model
from unique_bilevel_programming_cplex.src.base.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.base.var_expr_con import Var
from unique_bilevel_programming_cplex.src.base.common import Sense


inst_ = Model()
x1, x2 = Var("x1"), Var("x2")

inst_.add_obj(x1 + x2, Sense.MIN)

b = inst_.add_constr(x1 + x2 == 1)
l1 = inst_.add_constr(x1.e >= 0)
l2 = inst_.add_constr(x2.e >= 0)
u1 = inst_.add_constr(x1.e <= 1)
u2 = inst_.add_constr(x2.e <= 1)


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
    model = UBModel(inst_, eps=1e-2, big_m=1e2)
    model.init_b_as_var([u1, u2])
    model.set_x0({Var("x1"): 1, Var("x2"): 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 1) and equal_q_f_p(answer[x2], 0)
    assert equal_q_f_p(answer[Var(f"b_{u1.name}")], 1) and equal_q_f_p(answer[Var(f"b_{u2.name}")], 0)


def test_simple_instance_l():
    model = UBModel(inst_, eps=1e-2, big_m=1e2)
    model.init_b_as_var([l1, l2])
    model.set_x0({Var("x1"): 1, Var("x2"): 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 1) and equal_q_f_p(answer[x2], 0)
    assert equal_q_f_p(answer[Var(f"b_{l1.name}")], 1) and equal_q_f_p(answer[Var(f"b_{l2.name}")], 0)


def test_simple_instance_b_min():
    model = UBModel(inst_, eps=1e-2, big_m=1e2)
    model.init_b_as_var([b])
    model.set_x0({Var("x1"): 1, Var("x2"): 0})

    model.init()
    answer = model.solve()

    assert answer[Var(f"b_{b.name}")] == 0 or answer[Var(f"b_{b.name}")] == 2


def test_simple_instance_b_max():
    inst_.sense = Sense.MAX
    model = UBModel(inst_, eps=1e-2, big_m=1e2)
    model.init_b_as_var([b])
    model.set_x0({Var("x1"): 1, Var("x2"): 0})

    model.init()
    answer = model.solve()

    assert answer[Var(f"b_{b.name}")] == 0 or answer[Var(f"b_{b.name}")] == 2
    inst_.sense = Sense.MIN
