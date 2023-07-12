from MIBLP.src.mibpl_instance import MIBPLInstance
import coptpy


class MIBLPSolver:
    """
    Class that solves MIBPL problem using "Decomposition Algorithm" discribed in the paper:

    ```A Projection-Based Reformulation and Decomposition Algorithm for Global
    Optimization of a Class of Mixed Integer Bilevel Linear Programs.

    Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You```

    """
    def __init__(self, instance: MIBPLInstance):
        self.envr = coptpy.Envr()
        self._master_problem: coptpy.Model = self._master_problem_init(instance)
        self._subproblem_1: coptpy.Model = self._subproblem_1_init(instance)
        self._subproblem_2: coptpy.Model = self._subproblem_2_init(instance)

    def _master_problem_init(self, inst: MIBPLInstance):
        model: coptpy.Model = self.envr.createModel(name="Master")
        x_r = model.addMVar(shape=inst.m_r, lb=0.0, ub=coptpy.COPT.INFINITY, vtype=coptpy.COPT.CONTINUOUS, name="x_r")
        x_z = model.addMVar(shape=inst.m_z, lb=0, ub=coptpy.COPT.INFINITY, vtype=coptpy.COPT.INTEGER, name="x_z")

        return self._master_problem

    def _subproblem_1_init(self, inst: MIBPLInstance):
        return self._subproblem_1

    def _subproblem_2_init(self, inst: MIBPLInstance):
        return self._subproblem_2

    def solve(self, inst: MIBPLInstance):
        pass


def test():
    pass


test()