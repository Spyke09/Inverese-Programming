import typing as tp

from unique_bilevel_programming_cplex.src.common import Sense
from unique_bilevel_programming_cplex.src.var_expr_con import Var, LinExpr, Constraint


class Model:
    def __init__(self):
        self.obj: LinExpr = LinExpr(0)
        self.sense: Sense = Sense.MIN
        self.constraints: tp.List[Constraint] = list()
        self._vars: tp.Set[Var] = set()

    @property
    def vars(self) -> tp.Set[Var]:
        return set(self._vars)

    def add_constr(self, constr: Constraint) -> Constraint:
        self.constraints.append(constr)
        self._vars.update(constr.vars)
        return constr

    def add_constrs(self, constrs: tp.Iterable[Constraint]) -> tp.Set[Constraint]:
        names = set()
        for i in constrs:
            names.add(self.add_constr(i))
        return names

    def add_obj(self, expr: LinExpr, sense: Sense) -> None:
        if self.obj is not None:
            self._vars = self.vars.difference(self.obj.vars.difference(expr.vars))
        self.obj = expr
        self.sense = sense
        self._vars.update(expr.vars)

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