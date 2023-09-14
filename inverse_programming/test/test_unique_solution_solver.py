from inverse_programming.src.solver.unique_solution_solver import UniqueSoluteionSolver
from inverse_programming.src.structures.inv_instance import InvLpInstance, LpSign


def test1():
    inst = InvLpInstance(
        a=[[1, 1]],
        b=[1],
        c=[1, 2],
        sign=LpSign.Equal,
        lower_bounds=[0, 0]
    )
    solver = UniqueSoluteionSolver()

    res = solver.solve(
        inst, [0, 1]
    )

    print(res)


test1()
