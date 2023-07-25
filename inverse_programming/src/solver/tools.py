import numpy as np
import pulp

from inverse_programming.src.config import config
from inverse_programming.src.structures import inv_instance


def create_pulp_model_from_inv_lp_instance(instance: inv_instance.InvLpInstance, name: str = "UNNAMED"):
    """
    Создание модели pulp из модели InvLpInstance.

    :param instance: исходный экземпляр ЗЛП.
    :param name: опционально - имя модели pulp
    :return: модель pulp.
    """
    n, m = instance.a.shape

    model = pulp.LpProblem(name, pulp.LpMinimize)
    t1, t2 = instance.lower_bounds is None, instance.upper_bounds is None
    if t1 and t2:
        x = [pulp.LpVariable(f"x_{i}") for i in range(m)]
    elif not t1 and t2:
        x = [pulp.LpVariable(f"x_{i}", lowBound=instance.lower_bounds[0, i]) for i in range(m)]
    elif t1 and not t2:
        x = [pulp.LpVariable(f"x_{i}", upBound=instance.upper_bounds[0, i]) for i in range(m)]
    else:
        x = [pulp.LpVariable(f"x_{i}", lowBound=instance.lower_bounds[0, i], upBound=instance.upper_bounds[0, i]) for i in range(m)]

    # целевая функция
    model += pulp.lpSum([instance.c[0, i] * x[i] for i in range(m)])

    # ограницения из матрицы a
    if instance.sign == inv_instance.LpSign.MoreE:
        for i in range(n):
            model += (pulp.lpSum([x[j] * instance.a[i, j] for j in range(m)]) >= instance.b[0, i])

    if instance.sign == inv_instance.LpSign.Equal:
        for i in range(n):
            model += (pulp.lpSum([x[j] * instance.a[i, j] for j in range(m)]) == instance.b[0, i])

    if instance.sign == inv_instance.LpSign.LessE:
        for i in range(n):
            model += (pulp.lpSum([x[j] * instance.a[i, j] for j in range(m)]) <= instance.b[0, i])

    return model


def get_x_after_model_solve(inst: inv_instance.InvLpInstance):
    model = create_pulp_model_from_inv_lp_instance(inst)
    status = model.solve(config.PULP_SOLVER)
    if status != 1:
        raise ValueError("Status after model solving is False")

    x = inv_instance.LPArray((1, inst.c.shape[1]))
    for v in model.variables():
        if "x_" in v.name:
            x[0, int(v.name[2:])] = v.varValue
    return x
