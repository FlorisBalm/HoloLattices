import logging
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

import HolographicLattices.DifferentialOperators.DifferentialOperator
import HolographicLattices.Equations.EquationsBase
import HolographicLattices.Equations.Jacobian
import HolographicLattices.Utilities.GridUtilities
from HolographicLattices.IO import OutputResults
from HolographicLattices.NonLinearSolver.NonLinearSolverBase import NonLinearSolverBase
from HolographicLattices.Physics.Observables import Observables

@dataclass
class NewtonRaphsonFiniteDifferenceBVPSolver(NonLinearSolverBase):
    """
    For the NR FDD BVP solver, we require that the operator that is used to evaluate the derivatives is
    the same operator that is used to calculate inverses in the linear solver.
    """
    differential_operator: HolographicLattices.DifferentialOperators. \
        DifferentialOperator.FiniteDifferenceDifferentialOperator

    jacobian: HolographicLattices.Equations.Jacobian.FiniteDifferenceJacobian

    grid_information: HolographicLattices.Utilities.GridUtilities.GridInformation

    observables : HolographicLattices.Physics.Observables.Observables
    pre_solve_hooks : List[Callable] = field(default_factory = lambda: [])
    during_solve_hooks : List[Callable] = field(default_factory = lambda: [])
    post_solve_hooks : List[Callable] = field(default_factory = lambda: [])
    last_result: OutputResults.Result = field(init=False)
    # 
    convergence_criterion: Callable = field(init=False)


    def __post_init__(self):
        def evaluate_function_update_squared_default(result: OutputResults.Result):
            return np.average(result.last_update ** 2)
        self.convergence_criterion = evaluate_function_update_squared_default
        return


    def solve(self, current_guess: np.ndarray):

        logger = logging.getLogger(__name__)

        function_update = None
        fields_derivatives = self.differential_operator(current_guess)
        eom_residual = self.equations_of_motion.evaluate(fields_derivatives, grid_information=self.grid_information)
        for hook in self.pre_solve_hooks:
            hook(**locals())

        if (resid := np.sqrt(np.sum(eom_residual**2))) < self.solver_options.tolerance:
            logger.info(f"Residual equal to {resid} < {self.solver_options.tolerance}, no need to make a new update")

            if self.observables is not None:
                observables = self.observables.evaluate(fields_derivatives, grid_information=self.grid_information)
            else:
                observables = None
            function_update=np.zeros_like(current_guess)

            self.last_result = OutputResults.Result(observables=observables, fields=fields_derivatives,
                                                    last_update=function_update, eom_residual=eom_residual,
                                                    equation_options=self.grid_information.options,
                                                    constant_options=self.equations_of_motion.constant_options)
            return current_guess, function_update, True

        for num_nonlinear_steps in range(self.solver_options.max_nonlinear_steps):
            logger.info(f"Starting step {num_nonlinear_steps + 1}")
            fields_derivatives = self.differential_operator(current_guess)

            eom_residual = self.equations_of_motion.evaluate(fields_derivatives, grid_information=self.grid_information)

            jacobian = self.jacobian.evaluate(fields_derivatives, grid_information=self.grid_information)

            function_update = self.linear_solver.solve(jacobian, eom_residual)

            current_guess -= self.solver_options.nonlinear_update_step*(function_update.reshape(current_guess.shape))

            # This can be used to evaluate the control observables, too
            if self.observables is not None:
                observables = self.observables.evaluate(fields_derivatives, grid_information=self.grid_information)
            else:
                observables = None

            fields_derivatives = self.differential_operator(current_guess)

            self.last_result = OutputResults.Result(observables=observables, fields=fields_derivatives,
                                                    last_update=function_update, eom_residual=eom_residual,
                                                    equation_options=self.grid_information.options,
                                                    constant_options=self.equations_of_motion.constant_options)
            for hook in self.during_solve_hooks:
                hook(**locals())


            convergence_result = self.convergence_criterion(self.last_result)

            if (convergence_result < self.solver_options.tolerance):

                logger.info(f"Tolerance of {self.solver_options.tolerance} achieved in {num_nonlinear_steps+1} steps. Current convergence: {convergence_result}")
                for hook in self.post_solve_hooks:
                    hook(**locals())
                return current_guess, function_update, True
            elif (np.isnan(convergence_result) or np.isinf(convergence_result)): 

                logger.error(f"Tolerance failed, convergence is {convergence_result}. Returning.")
                return current_guess, function_update, False
            else:

                logger.info(
                f"Tolerance criterion of {convergence_result} does not reach specified level of {self.solver_options.tolerance}")

        else:
            logger.warning(f"Not able to achieve convergence of {self.solver_options.tolerance} in "
                           f"{self.solver_options.max_nonlinear_steps + 1} steps")
