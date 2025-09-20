import numpy as np
from .direction import Direction


class RegularizedNewton(Direction):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def compute_direction(self, zk, zbar, wk, Fzk, Fzbar, iteration):
        norm_F = np.linalg.norm(Fzk)
        rk = max(norm_F, np.sqrt(norm_F))
        jacobian = self.problem.Jacobian_F(zk)
        system_matrix = rk * np.eye(self.problem.dim_z) + jacobian

        try:
            dk = np.linalg.solve(system_matrix, -Fzk)
            return dk, True
        except np.linalg.LinAlgError:
            return None, False

    def post_step_update(
        self,
        zk, Fzk, zbar, Fzbar, wk,
        zk_plus_1, Fzk_plus_1, zbar_plus_1, Fzbar_plus_1, wk_plus_1,
        iteration,
    ):
        return
