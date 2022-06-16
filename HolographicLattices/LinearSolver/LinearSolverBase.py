from abc import ABCMeta, abstractmethod

# Base classes for Linear Solver methods. For example, will be able to include hooks,
# or other parameters etc.
class LinearSolverBase(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, operator, rhs):
        pass
