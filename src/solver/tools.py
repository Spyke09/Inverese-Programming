import numpy as np
import pulp

from src.config import config
from src.structures import simple_instance


def create_pulp_model_from_inv_lp_instance(instance: simple_instance.InvLpInstance, name: str = "UNNAMED"):
    """
    Создание модели pulp из модели InvLpInstance.

    :param instance: исходный экземпляр ЗЛП.
    :param name: опционально - имя модели pulp
    :return: модель pulp.
    """
    n, m = 0, 0
    if len(instance.a) != 0:
        n, m = instance.a.shape

    model = pulp.LpProblem(name, pulp.LpMinimize)
    x = list()
    t1, t2 = instance.lower_bounds is None, instance.upper_bounds is None
    for i in range(m):
        if t1 and t2:
            x_i = pulp.LpVariable(f"x_{i}")
        elif not t1 and t2:
            x_i = pulp.LpVariable(f"x_{i}", lowBound=instance.lower_bounds[i])
        elif t1 and not t2:
            x_i = pulp.LpVariable(f"x_{i}", upBound=instance.upper_bounds[i])
        else:
            x_i = pulp.LpVariable(f"x_{i}", lowBound=instance.lower_bounds[i], upBound=instance.upper_bounds[i])

        x.append(x_i)
    # целевая функция
    model += pulp.lpSum([instance.c[i] * x[i] for i in range(m)])

    # ограницения из матрицы a
    if instance.sign == simple_instance.LpSign.MoreE:
        for i in range(n):
            model += (pulp.lpSum([x[j] * instance.a[i, j] for j in range(m)]) >= instance.b[i])

    if instance.sign == simple_instance.LpSign.Equal:
        for i in range(n):
            model += (pulp.lpSum([x[j] * instance.a[i, j] for j in range(m)]) == instance.b[i])

    if instance.sign == simple_instance.LpSign.LessE:
        for i in range(n):
            model += (pulp.lpSum([x[j] * instance.a[i, j] for j in range(m)]) <= instance.b[i])

    return model


def get_x_after_model_solve(inst):
    model = create_pulp_model_from_inv_lp_instance(inst)
    status = model.solve(config.SOLVER)
    if status != 1:
        raise ValueError("Status after model solving is False")

    x = np.full(len(model.variables()), 0)
    for v in model.variables():
        x[int(v.name.replace("x_", ''))] = v.varValue
    return x
