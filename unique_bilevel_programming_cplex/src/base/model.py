import typing as tp

from unique_bilevel_programming_cplex.src.base.common import Sense
from unique_bilevel_programming_cplex.src.base.var_expr_con import Var, LinExpr, Constraint


class Model:
    def __init__(self):
        self._obj: LinExpr = LinExpr(0)
        self._sense: Sense = Sense.MIN
        self._constraints: tp.List[Constraint] = list()
        self._vars: tp.Set[Var] = set()

    @property
    def vars(self) -> tp.Set[Var]:
        return set(self._vars)

    @property
    def constraints(self) -> tp.List[Constraint]:
        return self._constraints

    @property
    def sense(self) -> Sense:
        return self._sense

    @sense.setter
    def sense(self, sense) -> None:
        self._sense = Sense(sense)

    @property
    def obj(self) -> LinExpr:
        return self._obj

    def add_constr(self, constr: tp.Union[Constraint, bool]) -> Constraint:
        if isinstance(constr, bool) and not constr:
            raise ValueError("Always false constraint")
        elif isinstance(constr, Constraint):
            self.constraints.append(constr)
            self._vars.update(constr.vars)
        else:
            raise TypeError
        return constr

    def add_constrs(self, constrs: tp.Iterable[tp.Union[Constraint, bool]]) -> tp.List[Constraint]:
        names = list()
        for i in constrs:
            names.append(self.add_constr(i))
        return names

    def add_obj(self, expr: tp.Union[LinExpr], sense: Sense) -> None:
        old_var = self._obj.vars - expr.vars
        self._vars -= old_var
        self._vars.update(expr.vars)

        self._obj = expr
        self._sense = sense

    @property
    def to_str(self) -> str:
        s = ""
        d = {0: 'max', 1: 'min'}
        s += f"{self._obj.to_str} -> {d[self.sense.value]}\n"
        for i in self.constraints:
            s += f"{i.to_str}\n"

        return s.strip("\n")

    def __repr__(self):
        s = self.to_str.replace('\n', '\n\t')
        return f"Model(\n\t{s}\n)"
