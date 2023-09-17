from inverse_programming.src.solver.unique_solution_solver import UniqueSoluteionSolver
from inverse_programming.src.structures.inv_instance import InvLpInstance, LpSign


def test1():
    inst = InvLpInstance(
        a=[[1, 1]],
        b=[1],
        c=[1, 1],
        sign=LpSign.Equal,
        lower_bounds=[0, 0],
    )
    solver = UniqueSoluteionSolver()

    x, c = solver.solve(
        inst, [0, 1], eps=10e-6, big_m=10e20
    )
    print((x, c))

    # x == [0.0, 1.0]
    # c == [0.999999999923855, 0.9999899999238551]


test1()
