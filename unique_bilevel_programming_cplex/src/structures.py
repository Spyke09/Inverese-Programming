import enum
import typing as tp
from dataclasses import dataclass

import docplex.mp
import docplex.mp.model
import docplex.mp.linear
import docplex.mp.vartype


LPFloat = float
Integral = (int, float, LPFloat)
LPVector = list
LPVectorT = tp.List


class VarType(enum.Enum):
    REAL = 0
    INTEGER = 1
    BIN = 2


@dataclass
class Var:
    _name: str
    _type: VarType = VarType.REAL

    @property
    def name(self):
        return self._name

    @property
    def to_str(self) -> str:
        return self._name

    @property
    def type(self):
        return self._type

    def __hash__(self) -> int:
        return self._name.__hash__()

    def __eq__(self, other):
        return isinstance(other, Var) and self._name == other._name

    def __repr__(self) -> str:
        return f"Var({self._name})"

    def __add__(self, y):
        return Expr.to_expr(self) + y

    def __sub__(self, y):
        return Expr.to_expr(self) - y

    def __mul__(self, y):
        return Expr.to_expr(self) * y

    def __truediv__(self, y):
        return Expr.to_expr(self) * (1 / y)

    def __radd__(self, y):
        return self.__add__(y)

    def __rsub__(self, y):
        return (self * (-1)) + y

    def __rmul__(self, y):
        return self.__mul__(y)


class Expr:
    def __init__(self, *args):
        self._vars_coef: tp.Dict[Var, LPFloat] = dict()
        self.coef_0: LPFloat = LPFloat(0)

        if 2 <= len(args) <= 3:
            if isinstance(args[0], LPVector) and isinstance(args[1], LPVector):
                self.__init_from_lists(*args)
            if isinstance(args[0], dict):
                self.__init_from_dict(*args)
        elif len(args) == 1:
            self.__init_from_other_type(*args)

    def get(self, key):
        return self._vars_coef[key]

    def set(self, key, value):
        if value != 0:
            self._vars_coef[key] = value
        elif key in self._vars_coef:
            self._vars_coef.pop(key)

    def __contains__(self, item):
        return item in self._vars_coef

    def __init_from_lists(self, _vars: LPVector, _coefs: LPVector, _coef_0: LPFloat = LPFloat(0)):
        self._vars_coef = {i: j for i, j in zip(_vars, _coefs) if j != 0}
        self.coef_0 = LPFloat(_coef_0)

    def __init_from_dict(self, _coefs: tp.Dict[Var, LPFloat], _coef_0: LPFloat = LPFloat(0)):
        self._vars_coef = {i: j for i, j in _coefs.items() if j != 0}
        self.coef_0 = LPFloat(_coef_0)

    def __init_from_other_type(self, x):
        if isinstance(x, Integral):
            self.__init_from_dict({}, x)
        elif isinstance(x, Var):
            self.__init_from_dict({x: 1})
        elif isinstance(x, Expr):
            return self.__init_from_dict(x._vars_coef, x.coef_0)
        else:
            raise ValueError(f"Failed cast {x} to Expr")

    @staticmethod
    def to_expr(x):
        """
        Cast method i.e call constructor without creating copy of Expr
        """
        if isinstance(x, Integral) or isinstance(x, Var):
            return Expr(x)
        elif isinstance(x, Expr):
            return x
        else:
            raise ValueError(f"Failed cast {x} to Expr")

    @property
    def vars(self):
        return self._vars_coef.keys()

    @property
    def vars_coef(self):
        return dict(self._vars_coef)

    def __add__(self, y):
        y = Expr.to_expr(y)
        vars_coefs = dict(self._vars_coef)
        for v, coef in y._vars_coef.items():
            if v in self._vars_coef:
                vars_coefs[v] += coef
            else:
                vars_coefs[v] = coef

        return Expr(vars_coefs, self.coef_0 + y.coef_0)

    def __sub__(self, y):
        return self + (y * (-1))

    def __mul__(self, y):
        y = LPFloat(y)
        return Expr({i: j * y for i, j in self._vars_coef.items()}, self.coef_0 * y)

    def __truediv__(self, y):
        return self * (1 / y)

    def __iadd__(self, y):
        y = Expr.to_expr(y)
        for v, coef in y._vars_coef.items():
            if v in self._vars_coef:
                self._vars_coef[v] += coef
            else:
                self._vars_coef[v] = coef
        return self

    def __isub__(self, y):
        return self.__iadd__(y * (-1))

    def __imul__(self, y):
        y = LPFloat(y)
        self._vars_coef = {i: j * y for i, j in self._vars_coef.items()}
        self.coef_0 *= y
        return self

    def __itruediv__(self, y):
        return self.__imul__(1 / y)

    def __radd__(self, y):
        return self.__add__(y)

    def __rsub__(self, y):
        return (self * (-1)) + y

    def __rmul__(self, y):
        return self.__mul__(y)


    @property
    def to_str(self) -> str:
        s = ""
        for v, coef in self._vars_coef.items():
            if coef != 0:
                if len(s) > 0:
                    if coef < 0:
                        s += ' - ' if coef == -1 else f" - {-coef} * "
                    elif coef > 0:
                        s += ' + ' if coef == 1 else f" + {coef} * "
                else:
                    s += "" if coef == 1 else (" - " if coef == -1 else f"{coef} * ")
                s += v.to_str

        if len(s) != 0:
            if self.coef_0 > 0:
                s += f" + {self.coef_0}"
            elif self.coef_0 < 0:
                s += f" - {-self.coef_0}"
        else:
            s += f"{self.coef_0}"
        return s

    def __repr__(self):
        return f"Expr({self.to_str})"

    def __le__(self, other):
        return Constraint(self, Sign.L_EQUAL, other)

    def __ge__(self, other):
        return Constraint(self, Sign.G_EQUAL, other)

    def __eq__(self, other):
        return Constraint(self, Sign.EQUAL, other)


LPEntity = tp.Union[LPFloat, Var, Expr]


class Sign(enum.Enum):
    L_EQUAL = 0
    G_EQUAL = 1
    EQUAL = 2

    @property
    def to_str(self) -> str:
        return {0: "<=", 1: ">=", 2: "=="}[self.value]


class Sense(enum.Enum):
    MAX = 0
    MIN = 1

    @property
    def to_str(self) -> str:
        return {0: "-> max", 1: "-> min"}[self.value]


class Constraint:
    constraints_counter = 0

    def __init__(self, left: LPEntity, sign: Sign, right: LPEntity):
        self._expr: Expr = Expr.to_expr(left) - right
        self._b_coef: LPFloat = -self._expr.coef_0
        self._expr.coef_0 = 0
        self._sign: Sign = sign

        self._name = f"C_{Constraint.constraints_counter}"
        Constraint.constraints_counter += 1

    @property
    def to_str(self):
        return f"{self._expr.to_str} {self._sign.to_str} {self._b_coef}"

    def __repr__(self):
        return f"Constraint_{self.name}({self.to_str})"

    @property
    def sign(self):
        return self._sign

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def b_coef(self):
        return self._b_coef

    @property
    def vars(self):
        return self.expr.vars

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return self._name.__hash__()

    def __eq__(self, other):
        return id(self) == id(other)

    def rotate_sign(self):
        if self.sign != Sign.EQUAL:
            self._expr *= -1
            self._b_coef *= -1
            self._sign = Sign.L_EQUAL if self.sign == Sign.G_EQUAL else Sign.G_EQUAL


class Model:
    def __init__(self):
        self.obj: Expr = Expr(0)
        self.sense: Sense = Sense.MIN
        self.constraints: tp.List[Constraint] = list()
        self.vars: tp.Set[Var] = set()

    def add_constr(self, constr: Constraint) -> Constraint:
        self.constraints.append(constr)
        self.vars.update(constr.vars)
        return constr

    def add_constrs(self, constrs: tp.Iterable[Constraint]) -> tp.Set[Constraint]:
        names = set()
        for i in constrs:
            names.add(self.add_constr(i))
        return names

    def add_obj(self, expr: Expr, sense: Sense) -> None:
        self.obj = expr
        self.sense = sense
        self.vars.update(expr.vars)

    @property
    def to_str(self) -> str:
        s = ""
        d = {0: 'max', 1: 'min'}
        s += f"{self.obj.to_str} -> {d[self.sense.value]}\n"
        for i in self.constraints:
            s += f"{i.to_str}\n"

        return s.strip("\n")

    def __repr__(self):
        s = self.to_str.replace('\n', '\n\t')
        return f"Model(\n\t{s}\n)"


class UBModel:
    def __init__(self, model: Model):
        self._model: Model = model
        self._constraints: tp.List[Constraint] = list()
        self._vars: tp.Set[Var] = set(model.vars)

        self._x_0: tp.Optional[tp.Dict[Var, LPFloat]] = None
        self._c: tp.Dict[Var, tp.Union[Var, LPFloat]] = model.obj.vars_coef
        self._c_0: tp.Dict[Var, LPFloat] = dict(model.obj.vars_coef)
        self._b: tp.Dict[Constraint, tp.Union[Var, LPFloat]] = {i: i.b_coef for i in model.constraints}
        self._b_0: tp.Dict[Constraint, LPFloat] = dict(self._b)

        self._lam = None

    def init_c_as_var(self, *args):
        if len(args) == 0:
            self._c = {i: Var(f"c_{i.name}") for i in self._c}
            self._vars.update(self._c.values())
        elif len(args) == 1 and isinstance(args, (list, set)):
            for i in args[0]:
                self._c[i] = Var(f"c_{i.name}")
                self._vars.add(self._c[i])

    def init_b_as_var(self, *args):
        if len(args) == 0:
            self._b = {i: Var(f"b_{i.name}") for i in self._b}
            self._vars.update(self._b.values())
        elif len(args) == 1 and isinstance(args[0], (list, set)):
            for i in args[0]:
                self._b[i] = Var(f"b_{i.name}")
                self._vars.add(self._b[i])

    def set_x0(self, x_0: tp.Dict[Var, LPFloat]):
        self._x_0 = dict()
        for x_i, val_x_i in x_0:
            if x_i in self._model.vars:
                self._x_0[x_i] = val_x_i
            else:
                raise ValueError

        if len(self._x_0) != len(self._model.vars):
            raise ValueError

    def get_c(self):
        return dict(self._c)

    def get_b(self):
        return dict(self._b)

    def add_constr(self, constr: Constraint):
        if all(i in self._vars for i in constr.vars):
            self._constraints.append(constr)
        else:
            raise ValueError

    def add_constrs(self, constrs: tp.Iterable[Constraint]):
        for i in constrs:
            self.add_constr(i)

    def init(self):
        eps = 1e-2
        big_m = 1e2

        for i in self._model.constraints:
            con = Constraint(i.expr, i.sign, self._b[i])
            self._constraints.append(con)

        y = {i: Var(f"pi_{i.name}") for i in self._model.constraints}
        self._vars.update(y.values())
        for x_i in self._model.vars:
            expr = Expr(0)
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
            evar = Expr(var)
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

    def solve(self):
        m = docplex.mp.model.Model(
            name=f'UniqueBilevelProgramming'
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
            if con.sign == Sign.L_EQUAL:
                m.add_constraint(m.sum(x[i] * con.expr.vars_coef[i] for i in con.vars) <= con.b_coef)
            if con.sign == Sign.G_EQUAL:
                m.add_constraint(m.sum(x[i] * con.expr.vars_coef[i] for i in con.vars) >= con.b_coef)
            if con.sign == Sign.EQUAL:
                m.add_constraint(m.sum(x[i] * con.expr.vars_coef[i] for i in con.vars) == con.b_coef)

        m.minimize_static_lex(
            [
                -m.sum(x[i] for i in self._lam),
                m.sum(m.abs(x[self._c[i]] - self._c_0[i]) for i in self._c if self._c[i] in x) +
                m.sum(m.abs(x[self._b[i]] - self._b_0[i]) for i in self._b if self._b[i] in x),
            ]
        )

        m.solve()
        # print(m.solve_status)
        if m.solution is None:
            return None

        solution = {i: xi.solution_value for i, xi in x.items()}
        # print(solution)
        # print(m.blended_objective_values)
        rev_c = {j: i for i, j in self._c.items()}
        self._c.update({rev_c[i]: j for i, j in solution.items() if i in rev_c.keys()})
        rev_b = {j: i for i, j in self._b.items()}
        self._b.update({rev_b[i]: j for i, j in solution.items() if i in rev_b})
        new_model = Model()
        new_model.add_obj(Expr(sum(i * j for i, j in self._c.items())), sense=self._model.sense)
        for con in self._model.constraints:
            if con.sign == Sign.L_EQUAL:
                new_model.add_constr(con.expr <= self._b[con])
            if con.sign == Sign.G_EQUAL:
                new_model.add_constr(con.expr >= self._b[con])
            if con.sign == Sign.EQUAL:
                new_model.add_constr(con.expr == self._b[con])

        return new_model

    @property
    def to_str(self):
        s = ""
        for i in self._constraints:
            s += f"{i.to_str}\n"

        return s.strip("\n")

    def __repr__(self):
        s = self.to_str.replace('\n', '\n\t')
        return f"Model(\n\t{s}\n)"


if __name__ == "__main__":
    def test_1():
        x, y = Var("x"), Var("y")
        m = Model()
        m.add_obj(x + y, Sense.MIN)
        m.add_constr(x + y == 1)
        a1 = m.add_constr(y + 0 >= 0)
        a2 = m.add_constr(x + 0 >= 0)
        print(m)

        ubm = UBModel(m)
        ubm.init_b_as_var([a1, a2])

        ubm.init()
        print(ubm.solve())

    def test_2():
        x, y = Var("x"), Var("y")
        m = Model()
        m.add_obj(x + y, Sense.MIN)
        m.add_constr(x + y == 1)
        m.add_constr(y + 0 >= 0)
        m.add_constr(x + 0 >= 0)
        print(m)

        ubm = UBModel(m)
        ubm.init_c_as_var()

        ubm.init()
        print(ubm.solve())


    test_1()
    test_2()
