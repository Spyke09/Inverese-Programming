from inverse_programming.src.structures import inv_instance


def check_optimal_q(inst: inv_instance, sol, eps):
    eps = 10e-7
    if inst.upper_bounds is None and inst.lower_bounds is None:
        return (abs(sol["x"] * sol["y2"]) < eps).all()
    elif inst.upper_bounds is None and inst.lower_bounds is not None:
        return (abs((sol["x"] - inst.lower_bounds) * (sol["y2"])) < eps).all()
    elif inst.upper_bounds is not None and inst.lower_bounds is None:
        return (((sol["x"] - inst.upper_bounds) * (sol["y2"])) < eps).all()
    else:
        return (abs((sol["x"] - inst.lower_bounds) * (sol["y2"])) < eps).all() and \
                (abs((sol["x"] - inst.upper_bounds) * (sol["y3"])) < eps).all()


def check_optimal_unique_q(inst: inv_instance, sol, eps):
    if not check_optimal_q(inst, sol, eps=eps):
        return False

    m = inst.a.shape[1]
    if inst.upper_bounds is not None and inst.lower_bounds is not None:
        return (abs(sol["y1"]) >= eps).sum() + (abs(sol["y2"]) >= eps).sum() + (abs(sol["y3"]) >= eps).sum() == m

    else:
        return (abs(sol["y1"]) >= eps).sum() + (abs(sol["y2"]) >= eps).sum() == m
