import typing as tp

import docplex.mp
import docplex.mp.dvar
import docplex.mp.linear
import docplex.mp.model
import docplex.mp.vartype

from unique_bilevel_programming_cplex.src.common import LPFloat, Sign, Sense
from unique_bilevel_programming_cplex.src.var_expr_con import LinExpr, Constraint, Var, VarType
from unique_bilevel_programming_cplex.src.model import Model


class UBModel:
    def __init__(self, model: Model, eps=1e-2, big_m=1e+2):
        self._big_m = big_m
        self._eps = eps

        self._model: Model = model
        self._constraints: tp.List[Constraint] = list()
        self._vars: tp.Set[Var] = set(model.vars)

        self._x_0: tp.Optional[tp.Dict[Var, LPFloat]] = None
        self._c: tp.Dict[Var, tp.Union[Var, LPFloat]] = model.obj.vars_coef
        self._c_0: tp.Dict[Var, LPFloat] = dict(model.obj.vars_coef)
        self._b: tp.Dict[Constraint, tp.Union[Var, LPFloat]] = {i: i.b_coef for i in model.constraints}
        self._b_0: tp.Dict[Constraint, LPFloat] = dict(self._b)

        self._lam = None

        self._obj_p = {"c": 1, "b": 1, "x": 1}

    def init_c_as_var(self, *args) -> None:
        if len(args) == 0:
            self._c = {i: Var(f"c_{i.name}") for i in self._c}
            self._vars.update(self._c.values())
        elif len(args) == 1 and isinstance(args, (list, set)):
            for i in args[0]:
                self._c[i] = Var(f"c_{i.name}")
                self._vars.add(self._c[i])

    def init_b_as_var(self, *args) -> None:
        if len(args) == 0:
            self._b = {i: Var(f"b_{i.name}") for i in self._b}
            self._vars.update(self._b.values())
        elif len(args) == 1 and isinstance(args[0], (list, set)):
            for i in args[0]:
                self._b[i] = Var(f"b_{i.name}")
                self._vars.add(self._b[i])

    def set_x0(self, x_0: tp.Dict[Var, LPFloat]) -> None:
        self._x_0 = dict()
        for x_i, val_x_i in x_0.items():
            if x_i in self._model.vars:
                self._x_0[x_i] = val_x_i
            else:
                raise ValueError

        if len(self._x_0) != len(self._model.vars):
            raise ValueError

    def get_c(self) -> tp.Dict[Var, tp.Union[Var, LPFloat]]:
        return dict(self._c)

    def get_b(self) -> tp.Dict[Constraint, tp.Union[Var, LPFloat]]:
        return dict(self._b)

    def add_constr(self, constr: Constraint) -> None:
        if all(i in self._vars for i in constr.vars):
            self._constraints.append(constr)
        else:
            raise ValueError

    def add_constrs(self, constrs: tp.Iterable[Constraint]) -> None:
        for i in constrs:
            self.add_constr(i)

    def init(self) -> None:
        eps = self._eps
        big_m = self._big_m

        for i in self._model.constraints:
            con = Constraint(i.expr, i.sign, self._b[i])
            self._constraints.append(con)

        y = {i: Var(f"pi_{i.name}") for i in self._model.constraints}
        self._vars.update(y.values())
        for x_i in self._model.vars:
            expr = LinExpr(0)
            for con in y.keys():
                if x_i in con.expr:
                    expr += con.expr.get(x_i) * y[con]
            c = self._c[x_i] if x_i in self._c else 0
            self.add_constr(expr == c)

        lam = {i: Var(f"lam_{i.name}", VarType.BIN) for i in y.values()}
        gam = {i: Var(f"gam_{i.name}", VarType.BIN) for j, i in y.items() if j.sign == Sign.EQUAL}
        self._vars.update(lam.values())
        self._vars.update(gam.values())
        max_q, min_q = self._model.sense == Sense.MAX, self._model.sense == Sense.MIN
        for con, var in y.items():
            evar = LinExpr(var)
            if con.sign == Sign.G_EQUAL and max_q or con.sign == Sign.L_EQUAL and min_q:
                self.add_constr(evar <= 0)
                self.add_constr(evar <= -eps * lam[var])
                self.add_constr(evar >= -big_m * lam[var])
            elif con.sign == Sign.EQUAL:
                self.add_constr(evar >= eps * lam[var] - big_m * gam[var])
                self.add_constr(evar <= -eps * lam[var] + big_m * (1 - gam[var]))
            elif con.sign == Sign.L_EQUAL and max_q or con.sign == Sign.G_EQUAL and min_q:
                self.add_constr(evar >= 0)
                self.add_constr(evar >= eps * lam[var])
                self.add_constr(evar <= big_m * lam[var])

            if con.sign == Sign.G_EQUAL:
                self.add_constr(con.expr - self._b[con] <= big_m * (1 - lam[var]))
            elif con.sign == Sign.L_EQUAL:
                self.add_constr(self._b[con] - con.expr <= big_m * (1 - lam[var]))
        self._lam = list(lam.values())

    def _init_cplex_model(self) -> tp.Tuple[docplex.mp.model.Model, tp.Dict[Var, docplex.mp.dvar.Var]]:
        m = docplex.mp.model.Model(
            name=f'UniqueBilevelProgram'
        )
        x = dict()
        my_type_tp_cplex_type = {
            VarType.BIN: docplex.mp.vartype.BinaryVarType(),
            VarType.INTEGER: docplex.mp.vartype.IntegerVarType(),
            VarType.REAL: docplex.mp.vartype.ContinuousVarType()
        }
        for i in self._vars:
            x[i] = m.var(vartype=my_type_tp_cplex_type[i.type], name=i.name, lb=-m.infinity)

        for con in self._constraints:
            m.add_constraint(con.sign(m.sum(x[i] * con.expr.vars_coef[i] for i in con.vars), con.b_coef))

        m.minimize_static_lex(
            [
                -m.sum(x[i] for i in self._lam),
                m.sum(m.abs(x[self._c[i]] - self._c_0[i]) for i in self._c if self._c[i] in x) * self._obj_p["c"] +
                m.sum(m.abs(x[self._b[i]] - self._b_0[i]) for i in self._b if self._b[i] in x) * self._obj_p["b"] +
                m.sum(m.abs(x[i] - self._x_0[i]) for i in self._model.vars) * self._obj_p["x"]
            ]
        )

        return m, x

    def set_obj_priority(self, name, p):
        if p >= 0:
            self._obj_p[name] = p
        else:
            raise ValueError

    def make_new_model_from_solution(self, solution: tp.Dict[Var, LPFloat]) -> Model:
        rev_c = {j: i for i, j in self._c.items()}
        self._c.update({rev_c[i]: j for i, j in solution.items() if i in rev_c.keys()})
        rev_b = {j: i for i, j in self._b.items()}
        self._b.update({rev_b[i]: j for i, j in solution.items() if i in rev_b})
        new_model = Model()
        new_model.add_obj(LinExpr(sum(i * j for i, j in self._c.items())), sense=self._model.sense)
        for con in self._model.constraints:
            new_model.add_constr(con.sign(con.expr, self._b[con]))

        return new_model

    def solve(self) -> tp.Optional[tp.Dict[Var, LPFloat]]:
        m, x = self._init_cplex_model()
        m.solve()
        if m.solution is None:
            return None

        print(m.blended_objective_values)
        return {i: round(xi.solution_value, 7) for i, xi in x.items()}

    @property
    def to_str(self) -> str:
        s = ""
        for i in self._constraints:
            s += f"{i.to_str}\n"

        return s.strip("\n")

    def __repr__(self) -> str:
        s = self.to_str.replace('\n', '\n\t')
        return f"Model(\n\t{s}\n)"