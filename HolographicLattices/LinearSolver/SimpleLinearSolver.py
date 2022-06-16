from dataclasses import dataclass
from typing import Callable

import scipy.sparse.linalg

from HolographicLattices.LinearSolver.LinearSolverBase import LinearSolverBase


@dataclass
class SimpleLinearSolver(LinearSolverBase):

    solver_method: Callable = scipy.sparse.linalg.spsolve

    def solve(self, operator, rhs, **options):
        orig_shape = rhs.shape
        return self.solver_method(operator, rhs.ravel(), **options).reshape(orig_shape)