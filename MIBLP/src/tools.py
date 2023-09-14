import typing as tp

import coptpy


def model_repr(model: coptpy.Model):
    obj = model.getObjective()
    s = [(obj.getCoeff(i), obj.getVar(i).getName()) for i in range(obj.size)]
    ss = f"Model <coptpy.Model>\n"
    ss += f"{s[0][0]} * {s[0][1]}"

    for i in range(1, len(s)):
        if s[i][0] > 0:
            ss += f" + {s[i][0]} * {s[i][1]}"
        else:
            ss += f" - {-s[i][0]} * {s[i][1]}"

    ss += f" -> {'max' if model.ObjSense == coptpy.COPT.MAXIMIZE else 'min'}\n\n"
    cons = model.getConstrs()
    # usual constraits
    for i in range(cons.size):
        con = cons.getConstr(i)
        s = ""
        for j in range(model.getVars().size):
            var = model.getVar(j)
            coef = model.getCoeff(con, var)
            if coef == 0:
                continue
            if not s:
                s += f'{coef} * {var.getName()}'
            else:
                if coef >= 0:
                    s += f' + {coef} * {var.getName()}'
                if coef < 0:
                    s += f' - {-coef} * {var.getName()}'

        if con.ub >= 10e29:
            s += f" >= {con.lb}"
        elif con.lb <= -10e29:
            s += f" <= {con.ub}"
        else:
            if con.lb == con.ub:
                s += f" = {con.ub}"
            else:
                s = f"{con.lb} <= {s}"
                s += f" <= {con.ub}"
        ss += f"{s}\n"
    return ss


def vars_value_repr(model: coptpy.Model, st_vars: tp.List[str]):
    vars_ = model.getVars()
    values = model.getValues()
    s = "Vars  values: \n"
    for i, j in zip(vars_, values):
        if any(i.getName()[:len(k)] == k for k in st_vars):
            s += f"{i.getName()} = {j}\n"
    return s
