import numpy as np

from unique_bilevel_programming_cplex.src.base.model import Model
from unique_bilevel_programming_cplex.src.base.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.base.var_expr_con import Var
from unique_bilevel_programming_cplex.src.base.common import Sense


inst_1 = Model()
x1, x2 = Var("x1"), Var("x2")

inst_1.add_obj(x1 + x2, Sense.MIN)

b1 = inst_1.add_constr(x1 + x2 == 1)
l1 = inst_1.add_constr(x1.e >= 0)
l2 = inst_1.add_constr(x2.e >= 0)


inst_2 = Model()

inst_2.add_obj(x1 + x2, Sense.MIN)

b2 = inst_2.add_constr(x1 + x2 == 1)
u1 = inst_2.add_constr(x1.e <= 1)
u2 = inst_2.add_constr(x2.e <= 1)


def equal_q_f_p(a, b, eps=10e-7):
    return (np.abs(a - b) < eps).all()


def test_simple_instance_c_without_u():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    c = model.init_c_as_var()
    model.set_x0({x1: 1, x2: 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 1) and equal_q_f_p(answer[x2], 0)
    assert answer[c[x1]] != answer[c[x2]]


def test_simple_instance_l_without_u():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var([l1, l2])
    model.set_x0({x1: 1, x2: 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 1) and equal_q_f_p(answer[x2], 0)
    assert equal_q_f_p(answer[b[l1]], 1) and equal_q_f_p(answer[b[l2]], 0)


def test_simple_instance_b_without_u():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var([b1])
    model.set_x0({x1: 1, x2: 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 0) and equal_q_f_p(answer[x2], 0)
    assert equal_q_f_p(answer[b[b1]], 0)


def test_simple_instance_c_without_l():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    c = model.init_c_as_var()
    model.set_x0({x1: 1, x2: 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 1) and equal_q_f_p(answer[x2], 0)
    assert answer[c[x1]] != answer[c[x2]]


def test_simple_instance_u_without_l():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var([u1, u2])
    model.set_x0({x1: 1, x2: 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 1) and equal_q_f_p(answer[x2], 0)
    assert equal_q_f_p(answer[b[u1]], 1) and equal_q_f_p(answer[b[u2]], 0)


def test_simple_instance_b_without_l():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var([b1])
    model.set_x0({x1: 1, x2: 0})

    model.init()
    answer = model.solve()

    assert equal_q_f_p(answer[x1], 0) and equal_q_f_p(answer[x2], 0)
    assert equal_q_f_p(answer[b[b1]], 0)
