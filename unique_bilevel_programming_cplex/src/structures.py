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

inst_.add_constr(x1.e + x2.e == 1)
inst_.add_constr(x1.e >= 0)
inst_.add_constr(x2.e >= 0)
# inst_.add_constr(x1.e <= 1)
# inst_.add_constr(x2.e <= 1)

inst_.add_obj(x1.e + x2.e, Sense.MIN)



def equal_q_f_p(a, b, eps=10e-7):
    return (np.abs(a - b) < eps).all()



model = UBModel(inst_, eps=1e-2, big_m=1e2)
model.init_c_as_var()
model.set_x0({Var("x1"): 1, Var("x2"): 0})

model.init()
print(model)
answer = model.solve()
print(answer)
