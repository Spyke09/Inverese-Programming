from __future__ import annotations

import typing as tp

from unique_bilevel_programming_cplex.src.base.common import LPEntity, VarType, LPVector, LPFloat, Integral, Sign


class Var:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Var):
            self._name: str = args[0].name
            self._type: VarType = args[0].type
        elif len(args) == 1 and isinstance(args[0], str):
            self._name: str = args[0]
            self._type: VarType = VarType.REAL
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], VarType):
            self._name: str = args[0]
            self._type: VarType = args[1]
        else:
            raise ValueError

    @property
    def name(self) -> str:
        return self._name

    @property
    def to_str(self) -> str:
        return self._name

    @property
    def type(self) -> VarType:
        return self._type

    def __hash__(self) -> int:
        return self._name.__hash__()

    @property
    def e(self) -> LinExpr:
        return LinExpr(self)

    def __eq__(self, other: tp.Any) -> bool:
        return isinstance(other, Var) and self._name == other._name

    def __repr__(self) -> str:
        return f"Var({self._name})"

    def __add__(self, y: LPEntity) -> LinExpr:
        return self.e + y

    def __sub__(self, y: LPEntity) -> LinExpr:
        return LinExpr.to_expr(self) - y

    def __mul__(self, y: LPEntity) -> LinExpr:
        return LinExpr.to_expr(self) * y

    def __truediv__(self, y: LPEntity) -> LinExpr:
        return LinExpr.to_expr(self) * (1 / y)

    def __radd__(self, y: LPEntity) -> LinExpr:
        return self.__add__(y)

    def __rsub__(self, y: LPEntity) -> LinExpr:
        return (self * (-1)) + y

    def __rmul__(self, y: LPEntity) -> LinExpr:
        return self.__mul__(y)


class LinExpr:
    def __init__(self, *args):
        self._vars_coef: tp.Dict[Var, LPFloat] = dict()
        self.coef_0: LPFloat = LPFloat(0)

        if 2 <= len(args) <= 3:
            if isinstance(args[0], LPVector) and isinstance(args[1], LPVector):
                self._init_from_lists(*args)
            if isinstance(args[0], dict):
                self._init_from_dict(*args)
        elif len(args) == 1:
            self._init_from_other_type(*args)

    def get(self, key: Var) -> LPFloat:
        return self._vars_coef[key]

    def set(self, key: Var, value: LPFloat) -> None:
        if value != 0:
            self._vars_coef[key] = value
        elif key in self._vars_coef:
            self._vars_coef.pop(key)

    def __contains__(self, item: tp.Any) -> bool:
        return item in self._vars_coef

    def _init_from_lists(self, _vars: LPVector[Var], _coefs: LPVector[LPFloat], _coef_0: LPFloat = LPFloat(0)) -> None:
        self._vars_coef = {i: j for i, j in zip(_vars, _coefs) if j != 0}
        self.coef_0 = LPFloat(_coef_0)

    def _init_from_dict(self, _coefs: tp.Dict[Var, LPFloat], _coef_0: LPFloat = LPFloat(0)) -> None:
        self._vars_coef = {i: j for i, j in _coefs.items() if j != 0}
        self.coef_0 = LPFloat(_coef_0)

    def _init_from_other_type(self, x: tp.Any) -> None:
        if isinstance(x, Integral):
            self._init_from_dict({}, x)
        elif isinstance(x, Var):
            self._init_from_dict({x: LPFloat(1)})
        elif isinstance(x, LinExpr):
            return self._init_from_dict(x._vars_coef, x.coef_0)
        else:
            raise ValueError(f"Failed cast {x} to LinExpr")

    @staticmethod
    def to_expr(x: tp.Any) -> LinExpr:
        """
        Cast method i.e. call constructor without creating copy of LinExpr
        """
        if isinstance(x, Integral) or isinstance(x, Var):
            return LinExpr(x)
        elif isinstance(x, LinExpr):
            return x
        else:
            raise ValueError(f"Failed cast {x} to LinExpr")

    @property
    def vars(self) -> tp.Set[Var]:
        return set(self._vars_coef.keys())

    @property
    def vars_coef(self) -> tp.Dict[Var, LPFloat]:
        return dict(self._vars_coef)

    def __add__(self, y: LPEntity) -> LinExpr:
        y = LinExpr.to_expr(y)
        vars_coefs = dict(self._vars_coef)
        for v, coef in y._vars_coef.items():
            if v in self._vars_coef:
                vars_coefs[v] += coef
            else:
                vars_coefs[v] = coef

        return LinExpr(vars_coefs, self.coef_0 + y.coef_0)

    def __sub__(self, y: LPEntity) -> LinExpr:
        return self + (y * (-1))

    def __mul__(self, y: LPEntity) -> LinExpr:
        y = LPFloat(y)
        return LinExpr({i: j * y for i, j in self._vars_coef.items()}, self.coef_0 * y)

    def __truediv__(self, y: LPEntity) -> LinExpr:
        return self * (1 / y)

    def __iadd__(self, y: LPEntity) -> LinExpr:
        y = LinExpr.to_expr(y)
        for v, coef in y._vars_coef.items():
            if v in self._vars_coef:
                self._vars_coef[v] += coef
            else:
                self._vars_coef[v] = coef
        return self

    def __isub__(self, y: LPEntity) -> LinExpr:
        return self.__iadd__(y * (-1))

    def __imul__(self, y: LPEntity) -> LinExpr:
        y = LPFloat(y)
        self._vars_coef = {i: j * y for i, j in self._vars_coef.items()}
        self.coef_0 *= y
        return self

    def __itruediv__(self, y: LPEntity) -> LinExpr:
        return self.__imul__(1 / y)

    def __radd__(self, y: LPEntity) -> LinExpr:
        return self.__add__(y)

    def __rsub__(self, y: LPEntity) -> LinExpr:
        return (self * (-1)) + y

    def __rmul__(self, y: LPEntity) -> LinExpr:
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
                    s += "" if coef == 1 else ("-" if coef == -1 else f"{coef} * ")
                s += v.to_str

        if len(s) != 0:
            if self.coef_0 > 0:
                s += f" + {self.coef_0}"
            elif self.coef_0 < 0:
                s += f" - {-self.coef_0}"
        else:
            s += f"{self.coef_0}"
        return s

    def __repr__(self) -> str:
        return f"LinExpr({self.to_str})"

    def __le__(self, other: LPEntity) -> Constraint:
        return Constraint(self, Sign.L_EQUAL, other)

    def __ge__(self, other: LPEntity) -> Constraint:
        return Constraint(self, Sign.G_EQUAL, other)

    def __eq__(self, other: LPEntity) -> Constraint:
        return Constraint(self, Sign.EQUAL, other)


class Constraint:
    constraints_counter = 0

    def __init__(self, left: LPEntity, sign: Sign, right: LPEntity):
        self._expr: LinExpr = LinExpr.to_expr(left) - right
        self._b_coef: LPFloat = -self._expr.coef_0
        self._expr.coef_0 = 0
        self._sign: Sign = sign

        self._name = f"C_{Constraint.constraints_counter}"
        Constraint.constraints_counter += 1

    @property
    def to_str(self) -> str:
        return f"{self._expr.to_str} {self._sign.to_str} {self._b_coef}"

    def __repr__(self):
        return f"Constraint_{self.name}({self.to_str})"

    @property
    def sign(self) -> Sign:
        return self._sign

    @property
    def expr(self) -> LinExpr:
        return self._expr

    @property
    def b_coef(self) -> LPFloat:
        return self._b_coef

    @property
    def vars(self) -> tp.Set[Var]:
        return self.expr.vars

    @property
    def name(self) -> str:
        return self._name

    def __hash__(self) -> int:
        return self._name.__hash__()

    def __eq__(self, other: tp.Any) -> bool:
        return id(self) == id(other)

    def rotate_sign(self) -> None:
        if self.sign != Sign.EQUAL:
            self._expr *= -1
            self._b_coef *= -1
            self._sign = Sign.L_EQUAL if self.sign == Sign.G_EQUAL else Sign.G_EQUAL
