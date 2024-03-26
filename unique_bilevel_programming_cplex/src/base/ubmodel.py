import logging
import time
import typing as tp

import docplex.mp
import docplex.mp.dvar
import docplex.mp.linear
import docplex.mp.model
import docplex.mp.vartype
from collections.abc import Iterable
from unique_bilevel_programming_cplex.src.base.common import LPFloat, Sign, Sense, LPNan
from unique_bilevel_programming_cplex.src.base.model import Model
from unique_bilevel_programming_cplex.src.base.var_expr_con import LinExpr, Constraint, Var, VarType
import numpy as np


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

        self._logger = logging.getLogger("\tUBModel")

        self._x: tp.Optional[tp.Dict[Var, docplex.mp.dvar.Var]] = None
        self._std = None
        self._cplex_m: tp.Optional[docplex.mp.model.Model] = None

    def x_metric(self, metric):
        x_0 = self._x_0
        x = self._x
        x_true = np.array([x_0[i] for i in self._model.vars if i in x and i in x_0])
        x_pred = np.array([x[i].solution_value for i in self._model.vars if i in x and i in x_0])
        return metric(x_true, x_pred)

    def c_metric(self, metric):
        c, c_0 = self._c, self._c_0
        x = self._x
        c_true = np.array([c_0[i] for i in c if c[i] in x and i in c_0])
        c_pred = np.array([x[c[i]].solution_value for i in c if c[i] in x and i in c_0])
        return metric(c_true, c_pred)

    def b_metric(self, metric):
        b, b_0 = self._b, self._b_0
        x = self._x
        b_true = np.array([b_0[i] for i in b if b[i] in x and i in b_0])
        b_pred = np.array([x[b[i]].solution_value for i in b if b[i] in x and i in b_0])
        return metric(b_true, b_pred)

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
        else:
            raise ValueError

    def init_b_as_var(self, *args) -> tp.Dict[Constraint, Var]:
        if len(args) == 0:
            self._b = {i: Var(f"b_{i.name}") for i in self._b}
            self._vars.update(self._b.values())
            return self._b
        elif len(args) == 1 and isinstance(args[0], Iterable):
            res = dict()
            for i in args[0]:
                self._b[i] = Var(f"b_{i.name}")
                res[i] = self._b[i]
                self._vars.add(self._b[i])
            return res
        else:
            raise ValueError

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

    def add_constr(self, constr: tp.Union[Constraint, bool]) -> None:
        if isinstance(constr, bool) and not constr:
            raise ValueError("Always false constraint")
        elif isinstance(constr, Constraint):
            self._constraints.append(constr)
            self._vars.update(constr.vars)
        else:
            raise TypeError

    def add_constrs(self, constrs: tp.Iterable[tp.Union[Constraint, bool]]) -> None:
        for i in constrs:
            self.add_constr(i)

    def init(self) -> None:
        self._logger.info("Starting UB-Inv model initialization.")
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

        self._lam = lam.values()

        self._init_cplex_model()
        self._logger.info(f"UB-Inv model initialization is finished. "
                          f"Model with {len(self._vars)} vars, {len(self._constraints)} constraints.")

    def _init_cplex_model(self) -> None:
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

        c, c_0, b, b_0, x_0 = self._c, self._c_0, self._b, self._b_0, self._x_0
        x_std = m.sum(m.abs(x[i] - x_0[i]) for i in self._model.vars if i in x and i in x_0) * self._obj_p["x"]
        c_std = m.sum(m.abs(x[c[i]] - c_0[i]) for i in c if c[i] in x and i in c_0) * self._obj_p["c"]
        b_std = m.sum(m.abs(x[b[i]] - b_0[i]) for i in b if b[i] in x and i in b_0) * self._obj_p["b"]
        self._std = x_std + c_std + b_std
        m.minimize(self._std)

        self._lam = m.sum(x[i] for i in self._lam)
        self._x, self._cplex_m = x, m

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

    def __perform_solve(self, time_for_optimum=None, gap=None):
        gap = 1e-3 if gap is None else gap
        if time_for_optimum is None:
            while True:
                if self._cplex_m.solve() is not None:
                    cur_gap = self._cplex_m.solve_details.mip_relative_gap
                    if cur_gap <= gap:
                        self._logger.info(f"Given gap is reached. Gap = {cur_gap}")
                        break
        else:
            solved_q = False
            optim_time = None
            while True:
                if self._cplex_m.solve() is not None:
                    cur_gap = self._cplex_m.solve_details.mip_relative_gap
                    if cur_gap <= gap:
                        self._logger.info(f"Given gap is reached. Gap = {cur_gap}.")
                        break
                    if not solved_q:
                        optim_time = time.time()
                    elif (time.time() - optim_time) > time_for_optimum:
                        self._logger.info(f"The time for optimization has expired. Gap = {cur_gap}.")
                        break
                    solved_q = True

    def solve(self, first_unique=False, gap=None, time_for_optimum=None) -> tp.Optional[tp.Dict[Var, LPFloat]]:
        self._logger.info("Starting to solve UB-Inv model.")
        m, x = self._cplex_m, self._x
        m.parameters.mip.limits.solutions = 1

        lam_l = len(self._model.vars) - 1
        lam_u = len(self._model.constraints) + 1
        # lam_u = lam_l + 2
        final_sol = None
        while lam_l + 1 < lam_u:
            lam_m = (lam_u + lam_l) // 2
            self._logger.info(f"Next lower bound = {lam_l}, upper bound = {lam_u}, mid = {lam_m}.")
            con = m.add_constraint(self._lam >= lam_m)

            self.__perform_solve(time_for_optimum, gap)

            if m.solution is None:
                self._logger.info("Solution is None.")
                lam_u = lam_m
            else:
                sol = {i: round(xi.solution_value, 7) for i, xi in x.items()}
                if self._check_unique(sol):
                    self._logger.info(f"Solution is unique. "
                                      f"Error = {round(self._std.solution_value, 3)}.")
                    lam_u = lam_m
                    final_sol = sol
                    if first_unique:
                        break
                else:
                    self._logger.info(f"Solution is not unique.")
                    lam_l = lam_m
                    if final_sol is None:
                        final_sol = sol
            m.remove_constraint(con)

        self._logger.info("Finished to solve UB-Inv model.")
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
        for i in self._model.obj.vars:
            old_obj_v += solution[i] * (solution[self._c[i]] if isinstance(self._c[i], Var) else self._c[i])

        c = {
            i: (solution[self._c[i]] if isinstance(self._c[i], Var) else self._model.obj.vars_coef[i])
            for i in self._model.obj.vars
        }
        m.add_constraint(m.sum(x[i] * c[i] for i in self._model.obj.vars) == old_obj_v)

        eps = self._eps * 100
        m.add_constraint(m.max(m.abs(x[i] - solution[i]) for i in self._model.vars) >= eps)
        # m.maximize(m.max(m.abs(x[i] - solution[i]) for i in self._model.vars))

        m.solve()
        return m.solution is None

    @property
    def to_str(self) -> str:
        s = ""
        for i in self._constraints:
            s += f"{i.to_str}\n"

        return s.strip("\n")

    def __repr__(self) -> str:
        s = self.to_str.replace('\n', '\n\t')
        return f"Model(\n\t{s}\n)"
