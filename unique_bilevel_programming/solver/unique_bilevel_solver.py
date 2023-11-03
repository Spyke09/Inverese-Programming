import logging
from enum import Enum

import coptpy
import numpy as np

from unique_bilevel_programming import config
from unique_bilevel_programming.structures.unique_bilevel_instance import UBInstance


class SpWeights(Enum):
    Infinity = 1000000000
    Blank = -1

    @staticmethod
    def usual_q(c):
        return c != SpWeights.Blank and c != SpWeights.Infinity


class UBSolver:
    def __init__(self, big_m=10e1, eps=10e-2):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("UniqueBilevelSolver")
        self.model: coptpy.Model = self._envr.createModel(name="UniqueBilevelModel")
        self.model.setParam(coptpy.COPT.Param.Logging, config.COPT_LOG_LEVEL)
        self.big_m = big_m
        self.eps = eps
        self._weights = None
        self._inst = None

    @staticmethod
    def _preprocess_weights(inst: UBInstance, weights):
        # Вес None означает бесконечный вес, а именно, то что соответсвующую переменную мы не
        # создаем и говорим что она равна своей "нулевой версии"
        # К примеру, если вес x None, то переменная x не создается и в модели x = x0
        if any(i < 0 for i in weights.values()):
            raise ValueError("Weights should be greater or equal zero.")

        i = SpWeights.Infinity
        true_weights = {"x": i, "c": i, "b": i, "l": i, "u": i}
        true_weights.update(weights)
        if inst.x0 is None:
            true_weights["x"] = 0
        if inst.c0 is None:
            true_weights["c"] = 0
        if inst.l0 is None:
            true_weights["l"] = SpWeights.Blank
        if inst.u0 is None:
            true_weights["u"] = SpWeights.Blank
        if inst.b0 is None:
            true_weights["b"] = SpWeights.Blank

        return true_weights

    def solve(
            self,
            inst: UBInstance,
            weights,
            unique_idx=None
    ):
        """
        Метод решающий данную задачу UBInv с заданными весами минимизации.
        :param inst: экземпляр задачи UBInv
        :param weights: отображение вида `"x" -> w_x`, такое, что w_x будет множителем
                        в целевой функции у отклонения, т.е. w_x * ||x - x0||.
        :param unique_idx: индексы переменных, которые нужно сделать ункальными.
        :return: статус решения из солвера copt.
        """
        self._inst = inst
        self._weights = self._preprocess_weights(inst, weights)

        self._create_model()

        self.model.solve()
        if self.model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Master problem is infeasible.")
        return self.model.status

    def _create_abs_constraint(self, x, name):
        omega = self.model.addMVar(x.shape[0], vtype=coptpy.COPT.CONTINUOUS, nameprefix=name)
        self.model.addConstrs(-omega <= x)
        self.model.addConstrs(x <= omega)
        return omega

    def _vars_number(self, name):
        return sum([i.getName().split("(")[0] == name for i in self.model.getVars().getAll()])

    def get_values_by_names(self, names):
        return {name: self.get_values_by_name(name) for name in names if self._vars_number(name) > 0}

    def get_values_by_name(self, name):
        n = self._vars_number(name)
        return np.array([self.model.getVarByName(f"{name}({i})").getInfo("value") for i in range(n)])

    def _create_vars(self):
        n, m = self._inst.shape

        # Создание всех переменных задачи
        if self._weights["x"] != SpWeights.Infinity:
            x = self.model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="x", lb=-coptpy.COPT.INFINITY)
        else:
            x = self._inst.x0

        y = self.model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y", lb=-coptpy.COPT.INFINITY)
        if self._weights["l"] != SpWeights.Blank:
            y_lb = self.model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="y_lb", lb=0.0)
        else:
            y_lb = np.full(m, 0.0)

        # y_ub не создаем, так как он может быть однозначно выражен
        # y_ub = inst.A.T @ y + y_lb - c >= 0

        l1 = self.model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="l1")
        if self._weights["l"] != SpWeights.Blank:
            l2 = self.model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="l2")
        else:
            l2 = np.full(m, 0.0)
        if self._weights["u"] != SpWeights.Blank:
            l3 = self.model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="l3")
        else:
            l3 = np.full(m, 0.0)

        g = self.model.addMVar(n, vtype=coptpy.COPT.BINARY, nameprefix="g")

        if SpWeights.usual_q(self._weights["c"]):
            c = self.model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="c", lb=-coptpy.COPT.INFINITY)
        else:
            c = self._inst.c0

        if SpWeights.usual_q(self._weights["b"]):
            b = self.model.addMVar(n, vtype=coptpy.COPT.CONTINUOUS, nameprefix="b", lb=-coptpy.COPT.INFINITY)
        else:
            b = self._inst.b0

        if SpWeights.usual_q(self._weights["l"]):
            l = self.model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="l", lb=-coptpy.COPT.INFINITY)
        else:
            l = self._inst.l0

        if SpWeights.usual_q(self._weights["u"]):
            u = self.model.addMVar(m, vtype=coptpy.COPT.CONTINUOUS, nameprefix="u", lb=-coptpy.COPT.INFINITY)
        else:
            u = self._inst.u0

        return x, y, y_lb, l1, l2, l3, g, c, l, u, b

    def _add_primal_and_dual_constraints(self, v):
        x, y, y_lb, l1, l2, l3, g, c, l, u, b = v

        if self._weights["x"] == SpWeights.Infinity and self._weights["b"] == SpWeights.Infinity:
            if not (self._inst.A @ x == b).all():
                raise ValueError("x0 is not feasible solution for Ax0 == b0")
        else:
            self.model.addConstrs(self._inst.A @ x == b)

        if self._weights["u"] != SpWeights.Blank:
            self.model.addConstrs(self._inst.A.T @ y + y_lb - c >= 0)
        else:
            self.model.addConstrs(self._inst.A.T @ y + y_lb - c == 0)

        if self._weights["l"] != SpWeights.Blank:
            if self._weights["x"] == SpWeights.Infinity and self._weights["l"] == SpWeights.Infinity:
                if not (x >= l).all():
                    raise ValueError("x0 is not feasible solution for l <= x")
            else:
                self.model.addConstrs(x >= l)

        if self._weights["u"] != SpWeights.Blank:
            if self._weights["x"] == SpWeights.Infinity and self._weights["u"] == SpWeights.Infinity:
                if not (x <= u).all():
                    raise ValueError("x0 is not feasible solution for x <= u")
            else:
                self.model.addConstrs(x <= u)

    def _add_kkt_constraints(self, v):
        n, m = self._inst.shape
        x, y, y_lb, l1, l2, l3, g, c, l, u, b = v

        if self._weights["l"] != SpWeights.Blank:
            kkt2 = self.model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt2")
            self.model.addConstrs(y_lb <= kkt2 * self.big_m)
            self.model.addConstrs(x - l <= (1 - kkt2) * self.big_m)

        if self._weights["u"] != SpWeights.Blank:
            kkt3 = self.model.addMVar(m, vtype=coptpy.COPT.BINARY, nameprefix="kkt3")
            self.model.addConstrs(self._inst.A.T @ y + y_lb - c <= kkt3 * self.big_m)
            self.model.addConstrs(u - x <= (1 - kkt3) * self.big_m)

    def _add_feasible_region_constraints(self, v):
        x, y, y_lb, l1, l2, l3, g, c, l, u, b = v
        if self._weights["c"] != SpWeights.Blank:
            if self._inst.C is not None and self._inst.c_hat is not None:
                if self._weights["c"] == SpWeights.Infinity:
                    if not (self._inst.C @ c == self._inst.c_hat).all():
                        raise ValueError("Given c is not feasible for Cc = c_hat")
                else:
                    self.model.addConstrs(self._inst.C @ c == self._inst.c_hat)

        if self._weights["b"] != SpWeights.Blank:
            if self._inst.B is not None and self._inst.b_hat is not None:
                if self._weights["b"] == SpWeights.Infinity:
                    if not (self._inst.B @ b == self._inst.b_hat).all():
                        raise ValueError("Given b is not feasible for Bb = b_hat")
                else:
                    self.model.addConstrs(self._inst.B @ b == self._inst.b_hat)

        if self._weights["l"] != SpWeights.Blank:
            if self._inst.L is not None and self._inst.l_hat is not None:
                if self._weights["l"] == SpWeights.Infinity:
                    if not (self._inst.L @ l == self._inst.l_hat).all():
                        raise ValueError("Given c is not feasible for Ll = l_hat")
                else:
                    self.model.addConstrs(self._inst.L @ l == self._inst.l_hat)

        if self._weights["u"] != SpWeights.Blank:
            if self._inst.U is not None and self._inst.u_hat is not None:
                if self._weights["u"] == SpWeights.Infinity:
                    if not (self._inst.U @ u == self._inst.u_hat).all():
                        raise ValueError("Given c is not feasible for Uu = u_hat")
                else:
                    self.model.addConstrs(self._inst.U @ u == self._inst.u_hat)

    def _add_counting_constraints(self, v):
        x, y, y_lb, l1, l2, l3, g, c, l, u, b = v
        n, m = self._inst.shape

        self.model.addConstrs(y >= self.eps * l1 - self.big_m * g)
        self.model.addConstrs(y <= -self.eps * l1 + self.big_m * (1 - g))

        if self._weights["l"] != SpWeights.Blank:
            self.model.addConstrs(y_lb >= self.eps * l2)
            self.model.addConstrs(y_lb <= self.big_m * l2)

        if self._weights["u"] != SpWeights.Blank:
            self.model.addConstrs((self._inst.A.T @ y + y_lb - c) >= self.eps * l3)
            self.model.addConstrs((self._inst.A.T @ y + y_lb - c) <= self.big_m * l3)

        # self.model.addConstrs(l1.sum() + l2.sum() + l3.sum() == m)

    def _set_objective(self, v):
        x, y, y_lb, l1, l2, l3, g, c, l, u, b = v
        weighed_obj_sum_ = 0
        if SpWeights.usual_q(self._weights["x"]) and self._weights["x"] != 0:
            weighed_obj_sum_ += self._create_abs_constraint(x - self._inst.x0, "ome_x").sum() * self._weights["x"]
        if SpWeights.usual_q(self._weights["b"]) and self._weights["b"] != 0:
            weighed_obj_sum_ += self._create_abs_constraint(b - self._inst.b0, "ome_b").sum() * self._weights["b"]
        if SpWeights.usual_q(self._weights["c"]) and self._weights["c"] != 0:
            weighed_obj_sum_ += self._create_abs_constraint(c - self._inst.c0, "ome_c").sum() * self._weights["c"]
        if SpWeights.usual_q(self._weights["l"]) and self._weights["l"] != 0:
            weighed_obj_sum_ += self._create_abs_constraint(l - self._inst.l0, "ome_l").sum() * self._weights["l"]
        if SpWeights.usual_q(self._weights["u"]) and self._weights["u"] != 0:
            weighed_obj_sum_ += self._create_abs_constraint(u - self._inst.u0, "ome_u").sum() * self._weights["u"]

        # мега-костыль
        weighed_obj_sum_ += -1000 * (l1.sum() + l2.sum() + l3.sum())
        self.model.setObjective(weighed_obj_sum_, coptpy.COPT.MINIMIZE)

    def _create_model(self):
        v = self._create_vars()
        self._add_primal_and_dual_constraints(v)
        self._add_kkt_constraints(v)
        self._add_counting_constraints(v)
        self._add_feasible_region_constraints(v)

        self._set_objective(v)
