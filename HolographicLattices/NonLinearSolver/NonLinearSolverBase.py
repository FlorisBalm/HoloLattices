from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

import HolographicLattices.DifferentialOperators.DifferentialOperator
import HolographicLattices.Equations.EquationsBase
import HolographicLattices.LinearSolver.LinearSolverBase
import HolographicLattices.Options.Options


@dataclass
class NonLinearSolverBase(metaclass=ABCMeta):

    solver_options: HolographicLattices.Options.Options.NonLinearSolverOptions

    differential_operator: HolographicLattices.DifferentialOperators.DifferentialOperator.BaseDifferentialOperator
    equations_of_motion : HolographicLattices.Equations.EquationsBase.EquationsBase
    linear_solver: HolographicLattices.LinearSolver.LinearSolverBase.LinearSolverBase


    @abstractmethod
    def solve(self, initial_guess: np.ndarray):
        pass