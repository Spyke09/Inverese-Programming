import typing as tp

import numpy as np

from MIBLP.src import miblp_solver as miblp_solver, miblp_instance
from inverse_programming.src.structures import bilevel_instance


class MinMaxDistBilevelLpSolver:
    @staticmethod
    def _convert_to_miblp(inst: bilevel_instance.BilevelInstance, x0, big_m) -> miblp_instance.MIBPLInstance:
        # x_u
        b_p_idx = tuple(range(inst.big_b.shape[1]))
        b_m_idx = tuple(range(b_p_idx[-1] + 1, b_p_idx[-1] + 1 + inst.big_b.shape[1]))
        c_p_idx = tuple(range(b_m_idx[-1] + 1, b_m_idx[-1] + 1 + inst.big_c.shape[1]))
        c_m_idx = tuple(range(c_p_idx[-1] + 1, c_p_idx[-1] + 1 + inst.big_c.shape[1]))
        ome_idx = tuple(range(c_m_idx[-1] + 1, c_m_idx[-1] + 1 + inst.a.shape[1]))
        # x_l
        x_s_idx = tuple(range(inst.big_c.shape[1]))
        y_p_idx = tuple(range(x_s_idx[-1] + 1, x_s_idx[-1] + 1 + inst.a.shape[0]))
        y_m_idx = tuple(range(y_p_idx[-1] + 1, y_p_idx[-1] + 1 + inst.a.shape[0]))
        o_s_idx = tuple(range(y_m_idx[-1] + 1, y_m_idx[-1] + 1 + inst.big_c.shape[1]))
        # x_l_0
        x_l_idx = tuple(range(inst.big_c.shape[1]))
        # y_l
        phi_idx = tuple(range(inst.big_c.shape[1]))
        psi_idx = tuple(range(phi_idx[-1] + 1, phi_idx[-1] + 1 + inst.big_c.shape[1]))

        len_r = 2 * inst.big_c.shape[1] + 2 * inst.b.shape[1] + 2 * inst.c.shape[1]
        a_r = np.full((len_r, ome_idx[-1] + 1), 0.0)
        b_r = np.full((len_r, o_s_idx[-1] + 1), 0.0)
        r = np.full(len_r, 0.0)

        # -w_j + x_j <= x0_j
        for j in range(inst.big_c.shape[1]):
            a_r[(j, ome_idx[j])] = -1
            b_r[(j, x_l_idx[j])] = 1
            r[j] = x0[j]
        col_idx = inst.big_c.shape[1]

        # -w_j - x_j <= x0_j
        for j in range(inst.big_c.shape[1]):
            a_r[(col_idx + j, ome_idx[j])] = -1
            b_r[(col_idx + j, x_l_idx[j])] = -1
            r[col_idx + j] = -x0[j]
        col_idx += inst.big_c.shape[1]

        # B * b_p - B * b_m <= b~
        for i in range(inst.b.shape[1]):
            for j in range(inst.big_b.shape[1]):
                a_r[(col_idx + i, b_p_idx[j])] = inst.big_b[i, j]
                a_r[(col_idx + i, b_m_idx[j])] = -inst.big_b[i, j]
            r[col_idx + i] = inst.b[0, i]
        col_idx += inst.b.shape[1]

        # -B * b_p + B * b_m <= -b~
        for i in range(inst.b.shape[1]):
            for j in range(inst.big_b.shape[1]):
                a_r[(col_idx + i, b_p_idx[j])] = -inst.big_b[i, j]
                a_r[(col_idx + i, b_m_idx[j])] = inst.big_b[i, j]
            r[col_idx + i] = -inst.b[0, i]
        col_idx += inst.b.shape[1]

        # C * c_p - C * c_m <= c~
        for i in range(inst.c.shape[1]):
            for j in range(inst.big_c.shape[1]):
                a_r[(col_idx + i, c_p_idx[j])] = inst.big_c[i, j]
                a_r[(col_idx + i, c_m_idx[j])] = -inst.big_c[i, j]
            r[col_idx + i] = inst.c[0, i]
        col_idx += inst.c.shape[1]

        # -C * c_p + C * c_m <= -c~
        for i in range(inst.c.shape[1]):
            for j in range(inst.big_c.shape[1]):
                a_r[(col_idx + i, c_p_idx[j])] = -inst.big_c[i, j]
                a_r[(col_idx + i, c_m_idx[j])] = inst.big_c[i, j]
            r[col_idx + i] = -inst.c[0, i]
        col_idx += inst.c.shape[1]

        len_s = 2 * inst.a.shape[0] + 7 * inst.a.shape[1] + 2 * inst.big_c.shape[1]
        p_r = np.full((len_s, o_s_idx[-1] + 1), 0.0)
        p_z = np.full((len_s, psi_idx[-1] + 1), 0.0)
        q_r = np.full((len_s, ome_idx[-1] + 1), 0.0)
        s = np.full(len_s, 0.0)

        col_idx = 0
        # A * x_s - b_p + b_m <= 0
        # -A * x_s + b_p - b_m <= 0
        for i in range(inst.a.shape[0]):
            for j in range(inst.a.shape[1]):
                p_r[i, x_s_idx[j]] = inst.a[i, j]
                p_r[inst.a.shape[0] + i, x_s_idx[j]] = -inst.a[i, j]
            q_r[i, b_p_idx[i]] = -1
            q_r[i, b_m_idx[i]] = 1
            q_r[inst.a.shape[0] + i, b_p_idx[i]] = 1
            q_r[inst.a.shape[0] + i, b_m_idx[i]] = -1
        col_idx += 2 * inst.a.shape[0]

        # -A_T * y_p + A_T * y_m + c_p - c_m <= 0
        for i in range(inst.a.shape[1]):
            for j in range(inst.a.shape[0]):
                p_r[col_idx + i, y_p_idx[j]] = -inst.a[j, i]
                p_r[col_idx + i, y_m_idx[j]] = inst.a[j, i]
            q_r[col_idx + i, c_p_idx[i]] = 1
            q_r[col_idx + i, c_m_idx[i]] = -1
        col_idx += inst.a.shape[1]

        # -x_j - M * phi_j + o_s <= -x0_j
        for j in range(inst.a.shape[1]):
            p_r[col_idx + j, x_s_idx[j]] = -1
            p_r[col_idx + j, o_s_idx[j]] = 1
            s[col_idx + j] = -x0[j]
            p_z[col_idx + j, phi_idx[j]] = -big_m
        col_idx += inst.a.shape[1]

        # x_j - M * phi_j + o_s <= x0_j + M
        for j in range(inst.a.shape[1]):
            p_r[col_idx + j, x_s_idx[j]] = 1
            p_r[col_idx + j, o_s_idx[j]] = 1
            s[col_idx + j] = x0[j] + big_m
            p_z[col_idx + j, phi_idx[j]] = -big_m
        col_idx += inst.a.shape[1]

        # A_T * y_p - A_T * y_m - c_p + c_m - psi * M <= 0
        for i in range(inst.a.shape[1]):
            for j in range(inst.a.shape[0]):
                p_r[col_idx + i, y_p_idx[j]] = inst.a[j, i]
                p_r[col_idx + i, y_m_idx[j]] = -inst.a[j, i]
            q_r[col_idx + i, c_p_idx[i]] = -1
            q_r[col_idx + i, c_m_idx[i]] = 1
            p_z[col_idx + i, psi_idx[i]] = -big_m
        col_idx += inst.a.shape[1]

        # x_s + M * psi <= M
        for i in range(inst.a.shape[1]):
            p_r[col_idx + i, x_s_idx[i]] = 1
            p_z[col_idx + i, psi_idx[i]] = big_m
            s[col_idx + i] = big_m
        col_idx += inst.a.shape[1]

        # psi <= 1
        for i in range(inst.a.shape[1]):
            p_z[col_idx + i, psi_idx[i]] = 1
            s[col_idx + i] = 1.0
        col_idx += inst.a.shape[1]

        # phi <= 1
        for i in range(inst.a.shape[1]):
            p_z[col_idx + i, phi_idx[i]] = 1
            s[col_idx + i] = 1.0

        c_r = np.full(ome_idx[-1] + 1, 0.0)
        for i in range(inst.a.shape[1]):
            c_r[ome_idx[i]] = 1

        w_r = np.full(o_s_idx[-1] + 1, 0.0)
        for i in range(inst.a.shape[1]):
            w_r[o_s_idx[i]] = 1

        return miblp_instance.MIBPLInstance(
            c_r, None, None, None,
            a_r, None, b_r, None, r,
            w_r, None,
            q_r, None, p_r, p_z, s
        )

    @staticmethod
    def _reconvert_answer(
            miblp_answer: tp.Tuple[np.array, np.array, np.array, np.array],
            inst: bilevel_instance):
        b_p_idx = tuple(range(inst.big_b.shape[1]))
        b_m_idx = tuple(range(b_p_idx[-1] + 1, b_p_idx[-1] + 1 + inst.big_b.shape[1]))
        c_p_idx = tuple(range(b_m_idx[-1] + 1, b_m_idx[-1] + 1 + inst.big_c.shape[1]))
        c_m_idx = tuple(range(c_p_idx[-1] + 1, c_p_idx[-1] + 1 + inst.big_c.shape[1]))
        x_l_idx = tuple(range(inst.big_c.shape[1]))

        x_u, y_u, x_l0, y_l0 = miblp_answer
        b = x_u[b_p_idx[0]:b_p_idx[-1] + 1] - x_u[b_m_idx[0]:b_m_idx[-1] + 1]
        c = x_u[c_p_idx[0]:c_p_idx[-1] + 1] - x_u[c_m_idx[0]:c_m_idx[-1] + 1]
        x = x_l0[x_l_idx[0]:x_l_idx[-1] + 1]

        return x, b, c

    def solve(self, inst: bilevel_instance.BilevelInstance, x0, big_m=1000000):
        if inst.upper_bounds is not None:
            raise NotImplementedError("Algorithm with upper bounds not implemented.")

        if (inst.lower_bounds != 0.0).any():
            raise ValueError("Lower bounds should be zero.")

        miblp_inst = self._convert_to_miblp(inst, x0, big_m)
        solver = miblp_solver.MIBLPSolver()
        answer = solver.solve(miblp_inst)
        return self._reconvert_answer(answer, inst)
