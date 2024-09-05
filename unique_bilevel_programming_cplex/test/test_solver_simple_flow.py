import pytest
import numpy as np

from unique_bilevel_programming_cplex.src.base.model import Model
from unique_bilevel_programming_cplex.src.base.ubmodel import UBModel
from unique_bilevel_programming_cplex.src.base.var_expr_con import Var
from unique_bilevel_programming_cplex.src.base.common import Sense


edge = [Var(f"e{i}") for i in range(1, 6)]

inst_1 = Model()
inst_1.add_obj(sum(edge), Sense.MIN)

b1_1 = inst_1.add_constr(edge[0] + edge[1] + edge[2] == 2)
b2_1 = inst_1.add_constr(edge[0] - edge[3] == 0)
b3_1 = inst_1.add_constr(edge[2] - edge[4] == 0)
l_1 = inst_1.add_constrs(edge[i].e >= 0 for i in range(5))
u_1 = inst_1.add_constrs(edge[i].e <= 1 for i in range(5))


inst_2 = Model()
inst_2.add_obj(sum(edge), Sense.MIN)

b1_2 = inst_2.add_constr(edge[0] + edge[1] + edge[2] == 2)
b2_2 = inst_2.add_constr(edge[0] - edge[3] == 0)
b3_2 = inst_2.add_constr(edge[2] - edge[4] == 0)
b4_2 = inst_2.add_constr(edge[1] + edge[3] + edge[4] == 2)
l_2 = inst_2.add_constrs(edge[i].e >= 0 for i in range(5))
u_2 = inst_2.add_constrs(edge[i].e <= 1 for i in range(5))


inst_3 = Model()
inst_3.add_obj(sum(edge), Sense.MIN)

b1_3 = inst_3.add_constr(edge[0] + edge[1] + edge[2] == 2)
b2_3 = inst_3.add_constr(edge[0] - edge[3] == 0)
b3_3 = inst_3.add_constr(edge[2] - edge[4] == 0)
b4_3 = inst_3.add_constr(edge[1] + edge[3] + edge[4] == 2)
b5_3 = inst_3.add_constr(2 * edge[0] + edge[1] + edge[2] - edge[3] == 2)
l_3 = inst_3.add_constrs(edge[i].e >= 0 for i in range(5))
u_3 = inst_3.add_constrs(edge[i].e <= 1 for i in range(5))

x0 = {i: j for i, j in zip(edge, [1, 0, 1, 1, 1])}


def equal_q_f_p(a, b, eps=10e-7):
    return (np.abs(a - b) < eps).all()


@pytest.mark.parametrize(
    "instance", [inst_1, inst_2, inst_3]
)
def test_simple_instance_c(instance):
    model = UBModel(instance, eps=1e-2, big_m=1e2)
    c = model.init_c_as_var()
    model.set_x0(x0)

    model.init()
    answer = model.solve()
    c = [answer[c[i]] for i in edge]

    assert (-c[3] - c[0] + c[1] > 0) and (-c[4] - c[2] + c[1] > 0)


@pytest.mark.parametrize(
    "instance, u_", [(inst_1, u_1), (inst_2, u_2), (inst_3, u_3)]
)
def test_simple_instance_u(instance, u_):
    model = UBModel(instance, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var(u_)
    # model.set_x0(x0)
    model.add_constrs(i.e >= 0 for i in model.get_b().values() if isinstance(i, Var))

    model.init()
    answer = model.solve()
    u = [answer[b[i]] for i in u_]
    print(u)
    print(answer)
    assert ((u[1] == 0 or u[1] == 2) and (u[0] == u[2] == u[3] == u[4]) or (u[0] == 0 or u[3] == 0 or u[2] == 0 or u[4] == 0))


@pytest.mark.parametrize(
    "instance, l_", [(inst_1, l_1), (inst_2, l_2), (inst_3, l_3)]
)
def test_simple_instance_l(instance, l_):
    model = UBModel(instance, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var(l_)
    model.set_x0(x0)

    model.init()
    answer = model.solve()
    lb = [answer[b[i]] for i in l_]

    print(lb)
    print(answer)
    assert lb[0] == lb[2] == 1 or lb[3] == lb[4] == 1 or lb[0] == lb[4] == 1 or lb[2] == lb[3] == 1


@pytest.mark.parametrize(
    "instance, b_", [(inst_1, [b1_1, b2_1, b3_1]), (inst_2, [b1_2, b2_2, b3_2, b4_2]), (inst_3, [b1_3, b2_3, b3_3, b4_3, b5_3])]
)
def test_simple_instance_b(instance, b_):
    model = UBModel(instance, eps=1e-2, big_m=1e2)
    b = model.init_b_as_var(b_)
    model.set_x0(x0)
    model.add_constrs(i.e >= 0 for i in model.get_b().values() if isinstance(i, Var))

    model.init()
    answer = model.solve()
    b = [answer[b[i]] for i in b_]
    print(b)
    assert b[0] == 3 and b[1] == 0 and b[2] == 0
