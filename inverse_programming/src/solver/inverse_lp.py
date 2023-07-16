import abc

import numpy as np
import pulp

from inverse_programming.src.config import config
from inverse_programming.src.solver import tools
from inverse_programming.src.structures import simple_instance


def is_zero(x):
    """
    Проверка числа на ноль.

    Каким-то образом в результате округлений в некоторых случаях получается такая плохая точность.
    Поэтому x == 0 <=> abs(x) < 10e-7.
    """
    return abs(x) < 10e-7


class AbstractInverseLpSolver(abc.ABC):
    """
    Абстрактый класс солвера.
    Все наследники должны реализовывать метод `solve`.
    """
    @abc.abstractmethod
    def solve(self, instance: simple_instance.InvLpInstance, x0: np.array, weights: np.array = None):
        raise NotImplementedError

    @staticmethod
    def _find_binding_constraints(instance: simple_instance.InvLpInstance, x0: np.array):
        """
        Нахождение масок для множеств B, L, U, F (страница 16).

        :param instance: экземляр InvLpInstance
        :param x0: допустимое значение ЗЛП (на минимум) `instance`.
        :return: Маски для элементов из B, L, U, F (страница 16)
        """
        diff_for_b = instance.a @ x0 - instance.b
        idx_mask_b = [is_zero(i) for i in diff_for_b]

        t1, t2 = (instance.lower_bounds is not None), (instance.upper_bounds is not None)
        if t1 and not t2:
            diff_for_l = x0 - instance.lower_bounds
            idx_mask_l = [is_zero(i) for i in diff_for_l]
            idx_mask_f = [not is_zero(i) for i in diff_for_l]
            return idx_mask_b, idx_mask_f, idx_mask_l, None
        elif t2 and not t1:
            diff_for_u = x0 - instance.upper_bounds
            idx_mask_u = [is_zero(i) for i in diff_for_u]
            idx_mask_f = [not is_zero(i) for i in diff_for_u]
            return idx_mask_b, idx_mask_f, None, idx_mask_u
        elif t1 and t2:
            diff_for_l = x0 - instance.lower_bounds
            diff_for_u = x0 - instance.upper_bounds
            idx_mask_l = [is_zero(i) for i in diff_for_l]
            idx_mask_u = [is_zero(i) for i in diff_for_u]
            idx_mask_f = [(not idx_mask_l[i]) and (not idx_mask_u[i]) for i in range(len(idx_mask_u))]
            return idx_mask_b, idx_mask_f, idx_mask_l, idx_mask_u
        else:
            return idx_mask_b, [True for _ in range(instance.c.shape[0])], None, None


class InverseLpSolverL1(AbstractInverseLpSolver):
    """
    Класс, решающий задачу INV и использующий L1 норму.

    В этом классе реализован базовый случай задачи INV
        + дополнительный вариант задачи, когда заданы доп. ограничения
        l[i] <= x[i] <= u[i] в исходной задачи.
    Метод `solve` формирует задачу INV и решает ее, давая на выходе вектор d.
    """

    @staticmethod
    def __get_d(instance: simple_instance.InvLpInstance, dual_inv_answer: np.array, x0: np.array):
        """
        Формирование результирующего вектора `d`.

        :param instance: экземляр InvLpInstance
        :param dual_inv_answer: в статье это pi - оптимальные значения двойственных переменных.
        :param x0: заданный вектор x0 задачи INV.
        :return: вектор 'd'
        """

        # Формула из  страницы 11. Убавки из вектора стоимостей.
        c_pi = instance.c - instance.a.transpose().dot(dual_inv_answer)

        # Формулы со страниц 12, 19 (для задачи с ограничениями l[i] <= x[i] <= u[i]
        d = instance.c.copy()
        for j in range(len(d)):
            if c_pi[j] > 0 and x0[j] > instance.lower_bounds[j]:
                d[j] -= abs(c_pi[j])
            elif c_pi[j] < 0 and x0[j] < instance.upper_bounds[j]:
                d[j] += abs(c_pi[j])

        return d

    @staticmethod
    def __create_inv_lp_instance(instance: simple_instance.InvLpInstance, masks, x0):
        """
        Формирование экземпляра задачи обратного программирования.

        :param instance: экземлпляр исходной задачи.
        :param masks: маски множеств B, L, U, F (страница 16).
        :param x0: заданный вектор x0 задачи INV.
        :return: экземпляр задачи INV.
        """
        n, m = 0, 0
        if len(instance.a) != 0:
            n, m = instance.a.shape

        b_mask, f_mask, l_mask, u_mask = masks
        a = list()
        b = list()
        c = instance.c
        u = np.array([.0 for _ in range(m)])
        l_ = np.array([.0 for _ in range(m)])

        # Убираем "unbinding" ограницения
        for i in range(n):
            if b_mask[i]:
                a.append(instance.a[i])
                b.append(instance.b[i])

        # В зависимости от значения масок формируем дополнительные ограничения задачи INV.
        # Формулы со страницы 20. В целом здесь обобщенный случай, когда l_[j] != 0
        for j in range(m):
            if l_mask and l_mask[j]:
                l_[j] = instance.lower_bounds[j]
                u[j] = instance.lower_bounds[j] + 1.
            if u_mask and u_mask[j]:
                l_[j] = instance.upper_bounds[j] - 1.
                u[j] = instance.upper_bounds[j]
            if f_mask[j]:
                l_[j] = x0[j] - 1.
                u[j] = x0[j] + 1.

        return simple_instance.InvLpInstance(np.array(a), np.array(b), c, instance.sign, l_, u)

    def solve(self, instance: simple_instance.InvLpInstance, x0: np.array, weights: np.array = None):
        """
        Формирования и решение задачи INV.

        :param instance: исходный экземляр ЗЛП.
        :param x0: заданный вектор x0.
        :param weights: Здесь веса для вычисления нормы, но здесь это должно быть None.
        :return: вектор d.
        """

        # Здесь бросается ошибка, если заданы веса.
        # Из-за излищние громоздкости было решено убрать их.
        if weights is not None:
            raise ValueError("Algorithm with `weights` isn't implemented")

        # x0 должен быть лежать в допустимом множестве решений.
        if not instance.check_feasible(x0):
            raise ValueError("x_0 is not feasible vector")

        # формируем маски для формирования экземпляра INV
        masks = super()._find_binding_constraints(instance, x0)

        # формируем экземпляр INV
        inv_instance = self.__create_inv_lp_instance(instance, masks, x0)
        inv_model = tools.create_pulp_model_from_inv_lp_instance(inv_instance)

        # решаем и проверяем что экземпляра INV есть решение
        inv_status = inv_model.solve(config.SOLVER)
        if inv_status != 1:
            raise ValueError("Status after model solving is False")

        # получаем вектор pi и формируем из него решение задачи INV.
        dual_inv_answer = np.array([i.pi for _, i in inv_model.constraints.items()][:inv_instance.b.shape[0]])
        result_d = self.__get_d(inv_instance, dual_inv_answer, x0)

        # проверка, что найденные вектора удовлетворяют условиям оптимальности.
        # if not self.__check_L1(inv_instance, result_d, dual_inv_answer, x):
        #     raise ValueError("Solve Error")

        return result_d

    @staticmethod
    def __check_L1(instance: simple_instance.InvLpInstance, d, pi):
        """
        Проверка решения задачи INV на оптимальность.
        :param instance: исходный экземпляр
        :param d: найденное решение задачи INV
        :param pi: значения переменных двойственной задачи к INV
        :return: True если все порядке, иначе False
        """
        c1 = (instance.a.transpose().dot(pi) - d == 0).all()
        c2 = (pi >= 0).all()
        return c1 and c2


class InverseLpSolverLInfinity(AbstractInverseLpSolver):
    """
    Класс, решающий задачу INV и использующий L-infinity норму.

    В этом классе реализован базовый случай задачи INV
        + дополнительный вариант задачи, когда задана весовая норма,
        т. е. вместо обычной нормы  ||x|| = max(x[i], i из I)
        вычисляется норма ||x|| = max(x[i] * w[i], i из I)
    Метод `solve` формирует задачу INV и решает ее, выдает на выходе вектор d.
    """

    @staticmethod
    def __get_binding_instance(instance: simple_instance.InvLpInstance, b_mask):
        """
        Получение нового экземпляра путем избавления от unbinding ограничений в исходном.
        :param instance: исходный экземпляр ЗЛП.
        :param b_mask: маска binding ограничений.
        :return: новый экземпляр ЗЛП без unbinding ограничений.
        """

        n, m = 0, 0
        if len(instance.a) > 0:
            n, m = instance.a.shape

        # новые матрицы ограничений без unbinding ограничений
        a_ = list()
        b_ = list()

        for i in range(n):
            if b_mask[i]:
                a_.append(instance.a[i])
                b_.append(instance.b[i])
        return simple_instance.InvLpInstance(np.array(a_), np.array(b_), instance.c, instance.sign)

    def solve(self, instance: simple_instance.InvLpInstance, x0: np.array, weights: np.array = None):
        """
        Формирования и решение задачи INV.

        :param instance: исходный экземляр ЗЛП.
        :param x0: заданный вектор x0.
        :param weights: Здесь веса для вычисления нормы.
        :return: вектор d.
        """

        # В этом классе подразумевается, что у `instance` отсутствуют `upper_bounds` и `lower_bounds`
        if (instance.lower_bounds is not None) or (instance.upper_bounds is not None):
            raise ValueError("Algorithm with `bounds` isn't implemented")
        # по умолчанию weights = [1, 1, ..., 1]
        if weights is None:
            weights = np.full(instance.c.shape, 1.)

        # проверка на допустимость x0
        if not instance.check_feasible(x0):
            raise ValueError("x_0 is not feasible vector")

        # получение масок и формирование нового экземпляра без unbinding ограничений
        masks = super()._find_binding_constraints(instance, x0)
        binding_inst = self.__get_binding_instance(instance, masks[0])

        # создание модели pulp INV
        inv_model = self.__create_inv_model(binding_inst, x0, weights)
        # решение модели выше и проверка того, нашлось ли решение
        inv_model.solve(config.SOLVER)
        inv_model_status = inv_model.status
        if inv_model_status != 1:
            raise ValueError("Status after model solving is False")

        # двойственные переменные задачи.
        dual_inv_answer = np.array([i.pi for _, i in inv_model.constraints.items()][:binding_inst.b.shape[0]])
        # решение задачи INV
        d = self.__get_d(binding_inst, dual_inv_answer)

        # проверка, что найденные вектора удовлетворяют условиям оптимальности.
        if not self.__check_LInfinity(binding_inst, dual_inv_answer, d):
            raise ValueError("Solve Error")

        return d

    @staticmethod
    def __create_inv_model(binding_inst: simple_instance.InvLpInstance, x0, weights, name: str = "UNNAMED"):
        """
        Создание модели pulp задачи INV. Все по формулам из стр. 24.

        :param binding_inst: экземпляр исходной ЗЛП без unbinding ограничений.
        :param x0: заданый вектор x0
        :param weights: веса для нормы L-infinity.
        :param name: необязательный параметр - имя модели для pulp.
        :return: модель
        """
        n, m = 0, 0
        if len(binding_inst.a) != 0:
            n, m = binding_inst.a.shape

        model = pulp.LpProblem(name, pulp.LpMinimize)
        # вводим переменные x, а также переменные t[i], которые нужны для линеаризации условия
        # sum(|d_j - c_j| / w[j], j из J) <= 1
        x = pulp.LpVariable.dicts("x", [i for i in range(m)])
        t = pulp.LpVariable.dicts("t", [i for i in range(m)])

        # целевая функция: с * х.
        model += pulp.lpSum([binding_inst.c[i] * x[i] for i in range(m)])

        # добавление binding ограничений исходной задачи.
        for i in range(n):
            model += (pulp.lpSum([x[j] * binding_inst.a[i, j] for j in range(m)]) >= binding_inst.b[i])

        # добавление ограничений, связанных с ограничением INV.
        # -t[j] * w[j] <= x[j] - x0[j] <= t[j] * w[j] для всех j из J
        #  t[j] >= 0 для всех j из J
        for j in range(m):
            model += (t[j] >= 0)
            model += (x[j] - x0[j] <= t[j] * weights[j])
            model += (-t[j] * weights[j] <= x[j] - x0[j])

        # Также понятно, что sum(t[j]) == 1
        model += (pulp.lpSum([t[j] for j in range(m)]) == 1)

        return model

    @staticmethod
    def __get_d(instance: simple_instance.InvLpInstance, dual_inv_answer: np.array):
        """
        Формирование результирующего вектора `d`.

        :param instance: экземляр InvLpInstance
        :param dual_inv_answer: в статье это pi - оптимальные значения двойственных переменных.
        :return: вектор 'd'
        """
        c_pi = instance.c - instance.a.transpose().dot(dual_inv_answer)
        d = instance.c.copy()
        for j in range(len(d)):
            if c_pi[j] > 0:
                d[j] -= abs(c_pi[j])
            elif c_pi[j] < 0:
                d[j] += abs(c_pi[j])
        return d

    @staticmethod
    def __check_LInfinity(instance: simple_instance.InvLpInstance, pi, d):
        """
        Проверка решения задачи INV на оптимальность.

        :param instance: исходный экземпляр
        :param d: найденное решение задачи INV
        :param pi: значения переменных двойственной задачи к INV
        :return: True если все порядке, иначе False
        """
        c1 = (instance.a.transpose().dot(pi) - d == 0).all()
        c2 = (pi >= 0).all()

        return c1 and c2
