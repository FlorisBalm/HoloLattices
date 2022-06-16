import argparse
import h5py
import logging
from pathlib import Path

import numpy as np
import scipy.sparse.linalg

import HolographicLattices
from HolographicLattices.DifferentialOperators.DifferentialOperator import *
from HolographicLattices.Equations.EquationsOfMotion import EquationsOfMotion
from HolographicLattices.Equations.Jacobian import FiniteDifferenceJacobian
from HolographicLattices.IO.SeedLoader import load_seed_algebraic
from HolographicLattices.Physics.Observables import Observables
from HolographicLattices.LinearSolver.SimpleLinearSolver import SimpleLinearSolver
from HolographicLattices.NonLinearSolver.NRNonLinearSolver import NewtonRaphsonFiniteDifferenceBVPSolver as NRSolver
from HolographicLattices.Options import Options
from HolographicLattices.Utilities import GridUtilities


def SolveWithFiniteDifferences():

    # Basic argparse for some settings. Can be extended, but not required.
    parser = argparse.ArgumentParser()

    parser.add_argument("--setupfile", type=str, help="Setup .yaml file to read from")
    parser.add_argument("--output", type=str, nargs="?", help="Where to output")
    parser.add_argument("--log", type=str, default="INFO",
                        help="Logging level to print, choice of [NONE|DEBUG|INFO|WARN|ERROR|CRITICAL]")
    parser.add_argument("--logfile", type=str, default="SolveWithFD.log", help="Location of log file")
    parser.add_argument("--seedfile", type=str, nargs="?")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dense", action="store_true", help="Use fully dense spectral matrices in jacobian")
    grp.add_argument("--sparse", action="store_true", help="Use sparse apprximation to spectral matrices in jacobian")

    parsed, rest = parser.parse_known_args()

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", filename=parsed.logfile,
                        level=parsed.log)
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)

    # Load all the options
    setupfile = parsed.setupfile

    io_options       = Options.IOOptions.load_from_file(setupfile)
    equation_options = Options.EquationOptions.load_from_file(setupfile)
    solver_options   = Options.NonLinearSolverOptions.load_from_file(setupfile)
    constant_options = Options.ConstantOptions.load_from_file_safe(setupfile)

    grid_dimensions = equation_options.grid_sizes

    # Override the output if it is specified
    if parsed.output:
        io_options.output_file = Path(parsed.output)

    equations_of_motion = EquationsOfMotion(constant_options=constant_options, equation_options=equation_options)

    equations_of_motion.load_from_folder(io_options.coefficient_folder, equation_options.grid_periodic)

    grid_information = GridUtilities.GridInformation(equation_options)
    if parsed.dense:
        logger.info("Using fully dense spectral matrices to construct jacobian. This may be slow!")
        differential_operator = MixedFDDSpectralDifferentialOperator.construct_differentiation_matrices(
            equation_options=equation_options, max_deriv=2
        )
        # Here we want to use a dense algorithm
        linear_solver = SimpleLinearSolver(np.linalg.solve)
    else:
        logger.info("Using finite differences everywhere, even for "
                    "spectral dimensions in the construction of the jacobian.")
        differential_operator = MixedFDDSparseSpectralDifferentialOperator.construct_differentiation_matrices(
            equation_options=equation_options, max_deriv=2
        )
        linear_solver = SimpleLinearSolver()

    jacobian = FiniteDifferenceJacobian(
                constant_options=constant_options,
                equation_options=equation_options,
                finite_difference_matrices=differential_operator.get_matrix_representation()
            )

    jacobian.load_from_folder(folder=io_options.coefficient_folder, periodicities=equation_options.grid_periodic)




    observables = Observables.parse_observables(io_options.observables_file, constant_options, equation_options)



    nonlinear_solver = NRSolver(solver_options=solver_options, differential_operator=differential_operator,
                                equations_of_motion=equations_of_motion, linear_solver=linear_solver, jacobian=jacobian,
                                grid_information=grid_information, observables=observables)

    # We can use this but currently the equations I have ready are not quite correct
    # convergence_observables = Observables.parse_observables("Controls.yaml", constant_options, equation_options)
    # def evaluate_convergence_on_TrG(result: HolographicLattices.IO.OutputResults):
        #return np.average(convergence_observables.evaluate(result.fields, grid_information)["TrG"]**2)

    if parsed.seedfile:
        io_options.seed_file = parsed.seedfile

    if io_options.seed_algebraic:
        seed = load_seed_algebraic(io_options.seed_file, constant_options, grid_information)
    else:
        seed = h5py.File(io_options.seed_file,"r")["field"][:,0,:].reshape((equation_options.num_fields,np.prod(grid_dimensions)))[:,:]

    _, _, success = nonlinear_solver.solve(seed)

    if success:
        result = nonlinear_solver.last_result
        result.write_to_file(io_options.output_file)

if __name__ == '__main__':
    SolveWithFiniteDifferences()
