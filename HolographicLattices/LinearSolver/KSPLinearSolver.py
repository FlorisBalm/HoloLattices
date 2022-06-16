from dataclasses import dataclass

from HolographicLattices.LinearSolver.LinearSolverBase import LinearSolverBase


@dataclass
class KSPLinearSolver(LinearSolverBase):

    def solve(self, operator, rhs):
        raise NotImplementedError("This is reserved for more intricate implementations. Might be needed,"
                                  " maybe not.")