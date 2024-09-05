from unique_bilevel_programming_cplex.src.base.common import Sense
from unique_bilevel_programming_cplex.src.base.model import Model
from unique_bilevel_programming_cplex.src.base.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.base.var_expr_con import Var

inst_1 = Model()
x = [Var(f"x{i}") for i in range(1, 7)]

inst_1.add_obj(x[0] + x[1] + x[2] + x[3], Sense.MIN)

b1_1 = inst_1.add_constr(x[0] + x[1] == 1)
b2_1 = inst_1.add_constr(x[2] + x[3] == 1)
l_1 = inst_1.add_constrs(x[i].e >= 0 for i in range(4))
u_1 = inst_1.add_constrs(x[i].e <= 1 for i in range(4))

x0_1 = {x[i]: float(i % 2) for i in range(4)}


inst_2 = Model()
inst_2.add_obj(sum(x), Sense.MIN)

b1_2 = inst_2.add_constr(x[0] + x[1] == 1)
b2_2 = inst_2.add_constr(x[2] + x[3] == 1)
b3_2 = inst_2.add_constr(x[4] + x[5] == 1)
l_2 = inst_2.add_constrs(x[i].e >= 0 for i in range(6))
u_2 = inst_2.add_constrs(x[i].e <= 1 for i in range(6))

x0_2 = {x[i]: int(i % 2) for i in range(6)}


def test_simple_instance_c_1():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    c = model.init_c_as_var()
    model.set_x0(x0_1)

    model.init()
    answer = model.solve()
    assert answer[c[x[0]]] != answer[c[x[1]]]
    assert answer[c[x[2]]] != answer[c[x[3]]]


def test_simple_instance_c_2():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    c = model.init_c_as_var()
    model.set_x0(x0_2)

    model.init()
    answer = model.solve()
    assert answer[c[x[0]]] != answer[c[x[1]]]
    assert answer[c[x[2]]] != answer[c[x[3]]]
    assert answer[c[x[4]]] != answer[c[x[5]]]


def test_simple_instance_u_1():
    model = UBModel(inst_1, eps=1e-2, big_m=1e1)
    b = model.init_b_as_var(u_1)
    model.set_x0(x0_1)

    model.init()
    answer = model.solve()
    assert answer[b[u_1[0]]] == 0 and answer[b[u_1[2]]] == 0


def test_simple_instance_u_2():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var(u_2)
    model.set_x0(x0_2)

    model.init()
    answer = model.solve()
    assert answer[b[u_2[0]]] == 0 and answer[b[u_2[2]]] == 0
    assert answer[b[u_2[4]]] == 0


def test_simple_instance_l_1():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var(l_1)
    model.set_x0(x0_1)

    model.init()
    answer = model.solve()
    assert answer[b[l_1[1]]] == 1 and answer[b[l_1[3]]] == 1


def test_simple_instance_l_2():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var(l_2)
    model.set_x0(x0_2)

    model.init()
    answer = model.solve()
    assert answer[b[l_2[1]]] == 1 and answer[b[l_2[3]]] == 1
    assert answer[b[l_2[5]]] == 1


def test_simple_instance_b_1():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var([b1_1, b2_1])
    model.set_x0(x0_1)

    model.init()
    answer = model.solve()
    assert answer[b[b1_1]] == 0 or answer[b[b1_1]] == 2
    assert answer[b[b2_1]] == 0 or answer[b[b2_1]] == 2


def test_simple_instance_b_2():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var([b1_2, b2_2, b3_2])
    model.set_x0(x0_2)

    model.init()
    answer = model.solve()
    assert answer[b[b1_2]] == 0 or answer[b[b1_2]] == 2
    assert answer[b[b2_2]] == 0 or answer[b[b2_2]] == 2
    assert answer[b[b3_2]] == 0 or answer[b[b3_2]] == 2

