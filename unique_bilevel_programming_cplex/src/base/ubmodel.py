import typing as tp

import docplex.mp
import docplex.mp.dvar
import docplex.mp.linear
import docplex.mp.model
import docplex.mp.vartype

from unique_bilevel_programming_cplex.src.base.common import LPFloat, Sign, Sense
from unique_bilevel_programming_cplex.src.base.var_expr_con import LinExpr, Constraint, Var, VarType
from unique_bilevel_programming_cplex.src.base.model import Model


class UBModel:
    def __init__(self, model: Model, eps=1e-2, big_m=1e+2):
        self._big_m = big_m
        self._eps = eps

        self._model: Model = model
        self._constraints: tp.List[Constraint] = list()
        self._vars: tp.Set[Var] = set(model.vars)

        self._x_0: tp.Optional[tp.Dict[Var, LPFloat]] = dict()
        self._c: tp.Dict[Var, tp.Union[Var, LPFloat]] = model.obj.vars_coef
        self._c_0: tp.Dict[Var, LPFloat] = dict()
        self._b: tp.Dict[Constraint, tp.Union[Var, LPFloat]] = {i: i.b_coef for i in model.constraints}
        self._b_0: tp.Dict[Constraint, LPFloat] = dict()

        self._lam = None

        self._obj_p = {"c": 1, "b": 1, "x": 1}

    def init_c_as_var(self, *args) -> tp.Dict[Var, Var]:
        if len(args) == 0:
            self._c = {i: Var(f"c_{i.name}") for i in self._c}
            self._vars.update(self._c.values())
            return dict(self._c)
        elif len(args) == 1 and isinstance(args[0], (list, set)):
            res = dict()
            for i in args[0]:
                self._c[i] = Var(f"c_{i.name}")
                res[i] = self._c[i]
                self._vars.add(self._c[i])
            return res

    def init_b_as_var(self, *args) -> tp.Dict[Constraint, Var]:
        if len(args) == 0:
            self._b = {i: Var(f"b_{i.name}") for i in self._b}
            self._vars.update(self._b.values())
            return self._b
        elif len(args) == 1 and isinstance(args[0], (list, set)):
            res = dict()
            for i in args[0]:
                self._b[i] = Var(f"b_{i.name}")
                res[i] = self._b[i]
                self._vars.add(self._b[i])
            return res

    def set_b0(self, constrs: tp.List[Constraint]) -> None:
        self._b_0 = dict()
        for con in constrs:
            self._b_0[con] = con.b_coef

    def set_c0(self, x_coef: tp.Dict[Var, LPFloat]) -> None:
        self._c_0 = dict()
        for x, coef in x_coef:
            self._c_0[x] = coef

    def set_x0(self, x_0: tp.Dict[Var, LPFloat]) -> None:
        self._x_0 = dict(x_0)

    def get_c(self) -> tp.Dict[Var, tp.Union[Var, LPFloat]]:
        return dict(self._c)

    def get_b(self) -> tp.Dict[Constraint, tp.Union[Var, LPFloat]]:
        return dict(self._b)

    def add_constr(self, constr: Constraint) -> None:
        self._constraints.append(constr)
        return constr

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

        m.minimize(
            m.sum(m.abs(x[self._c[i]] - self._c_0[i]) for i in self._c if self._c[i] in x and i in self._c_0) * self._obj_p["c"] +
            m.sum(m.abs(x[self._b[i]] - self._b_0[i]) for i in self._b if self._b[i] in x and i in self._b_0) * self._obj_p["b"] +
            m.sum(m.abs(x[i] - self._x_0[i]) for i in self._model.vars if i in x and i in self._x_0) * self._obj_p["x"]
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
        lam_l = len(self._model.vars) - 1
        lam_u = len(self._model.constraints) + 1
        eps = (self._eps ** 2) / 10
        final_sol = None
        while lam_l + 1 < lam_u:
            lam_m = (lam_u + lam_l) // 2
            con = m.add_constraint(m.sum(x[i] for i in self._lam) == lam_m)
            m.solve()

            if m.solution is None:
                lam_u = lam_m
            else:
                sol = {i: round(xi.solution_value, 7) for i, xi in x.items()}
                if self._check_unique(sol) <= eps:
                    lam_u = lam_m
                    final_sol = sol
                else:
                    lam_l = lam_m
            m.remove_constraint(con)
        return final_sol

    def _check_unique(self, solution):
        m = docplex.mp.model.Model(
            name=f'CheckUnique'
        )
        x = dict()
        for i in self._model.vars:
            x[i] = m.continuous_var(name=i.name, lb=-m.infinity)

        for con in self._model.constraints:
            b_coef = solution[self._b[con]] if isinstance(self._b[con], Var) else con.b_coef
            m.add_constraint(con.sign(m.sum(x[i] * con.expr.vars_coef[i] for i in con.vars), b_coef))

        old_obj_v = 0
        for i in self._model.vars:
            old_obj_v += solution[i] * (solution[self._c[i]] if isinstance(self._c[i], Var) else self._c[i])

        c = {i: (solution[self._c[i]] if isinstance(self._c[i], Var) else self._model.obj.vars_coef[i]) for i in self._model.vars}
        m.add_constraint(m.sum(x[i] * c[i] for i in self._model.obj.vars) == old_obj_v)

        m.maximize(m.sum(m.abs(x[i] - solution[i]) for i in self._model.vars))
        m.solve()
        return m.blended_objective_values[0]

    @property
    def to_str(self) -> str:
        s = ""
        for i in self._constraints:
            s += f"{i.to_str}\n"

        return s.strip("\n")

    def __repr__(self) -> str:
        s = self.to_str.replace('\n', '\n\t')
        return f"Model(\n\t{s}\n)"
