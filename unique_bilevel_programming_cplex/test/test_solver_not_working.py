from unique_bilevel_programming_cplex.src.common import Sense
from unique_bilevel_programming_cplex.src.model import Model
from unique_bilevel_programming_cplex.src.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.var_expr_con import Var

inst_1 = Model()
x = [Var(f"x{i}") for i in range(1, 7)]

inst_1.add_obj(x[0] + x[1] + x[2] + x[3], Sense.MIN)

b1_1 = inst_1.add_constr(x[0] + x[1] == 1)
b2_1 = inst_1.add_constr(x[2] + x[3] == 1)
l_1 = inst_1.add_constrs(x[i].e >= 0 for i in range(4))
u_1 = inst_1.add_constrs(x[i].e <= 1 for i in range(4))

x0_1 = {x[i]: int(i % 2) for i in range(4)}


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
    model.init_c_as_var()
    model.set_x0(x0_1)

    model.init()
    answer = model.solve()
    assert answer[Var("c_x1")] != answer[Var("c_x2")]
    assert answer[Var("c_x3")] != answer[Var("c_x4")]


def test_simple_instance_c_2():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    model.init_c_as_var()
    model.set_x0(x0_2)

    model.init()
    answer = model.solve()
    assert answer[Var("c_x1")] != answer[Var("c_x2")]
    assert answer[Var("c_x3")] != answer[Var("c_x4")]
    assert answer[Var("c_x5")] != answer[Var("c_x6")]


def test_simple_instance_u_1():
    model = UBModel(inst_1, eps=1e-2, big_m=1e1)
    model.init_b_as_var(u_1)
    model.set_x0(x0_1)

    model.init()
    answer = model.solve()
    assert answer[Var(f"b_{u_1[0].name}")] == 0 and answer[Var(f"b_{u_1[2].name}")] == 0


def test_simple_instance_u_2():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    model.init_b_as_var(u_2)
    model.set_x0(x0_2)

    model.init()
    answer = model.solve()
    assert answer[Var(f"b_{u_2[0].name}")] == 0 and answer[Var(f"b_{u_2[2].name}")] == 0
    assert answer[Var(f"b_{u_2[4].name}")] == 0


def test_simple_instance_l_1():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    model.init_b_as_var(l_1)
    model.set_x0(x0_1)

    model.init()
    answer = model.solve()
    assert answer[Var(f"b_{l_1[1].name}")] == 1 and answer[Var(f"b_{l_1[3].name}")] == 1


def test_simple_instance_l_2():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    model.init_b_as_var(l_2)
    model.set_x0(x0_2)

    model.init()
    answer = model.solve()
    assert answer[Var(f"b_{l_2[1].name}")] == 1 and answer[Var(f"b_{l_2[3].name}")] == 1
    assert answer[Var(f"b_{l_2[5].name}")] == 1


def test_simple_instance_b_1():
    model = UBModel(inst_1, eps=1e-2, big_m=1e2)
    model.init_b_as_var([b1_1, b2_1])
    model.set_x0(x0_1)

    model.init()
    answer = model.solve()
    assert answer[Var(f"b_{b1_1.name}")] == 0 or answer[Var(f"b_{b1_1.name}")] == 2
    assert answer[Var(f"b_{b2_1.name}")] == 0 or answer[Var(f"b_{b2_1.name}")] == 2


def test_simple_instance_b_2():
    model = UBModel(inst_2, eps=1e-2, big_m=1e2)
    model.init_b_as_var([b1_2, b2_2, b3_2])
    model.set_x0(x0_2)

    model.init()
    answer = model.solve()
    assert answer[Var(f"b_{b1_2.name}")] == 0 or answer[Var(f"b_{b1_2.name}")] == 2
    assert answer[Var(f"b_{b2_2.name}")] == 0 or answer[Var(f"b_{b2_2.name}")] == 2
    assert answer[Var(f"b_{b3_2.name}")] == 0 or answer[Var(f"b_{b3_2.name}")] == 2

