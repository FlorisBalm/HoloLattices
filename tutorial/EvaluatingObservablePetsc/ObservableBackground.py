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


def EvaluateObservable(**kwargs):

    # Basic argparse for some settings. Can be extended, but not required.
    parser = argparse.ArgumentParser()

    parser.add_argument("--setupfile", type=str, help="Setup .yaml file to read from")
    parser.add_argument("--output", type=str, nargs="?", help="Where to output")
    parser.add_argument("--log", type=str, default="INFO",
                        help="Logging level to print, choice of [NONE|DEBUG|INFO|WARN|ERROR|CRITICAL]")
    parser.add_argument("--logfile", type=str, default="SolveWithFD.log", help="Location of log file")
    parser.add_argument("--files", nargs="+", help = "Files to evaluate")

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
    constant_options = Options.ConstantOptions.load_from_file_safe(setupfile)

    grid_information = GridUtilities.GridInformation(equation_options)


    logger.info("Using finite differences everywhere, even for "
                "spectral dimensions in the construction of the jacobian.")
    differential_operator = MixedFDDSparseSpectralDifferentialOperator.construct_differentiation_matrices(
        equation_options=equation_options
    )


    observed_evaluated = []
    print(parsed.files)
    for file in parsed.files:
        field, constants = load_seed_and_constants_from_h5(file)
        constant_options.constants = constants
        grid_sizes = grid_information.options.grid_sizes
        if not np.allclose(field.shape[1:],grid_sizes):
            logger.error(f"Shape does not match with field. Expected: {grid_information.options.grid_sizes} ,"
                         f" got: {field.shape[1:]}")
            exit(-1)
        fields_and_derivatives = differential_operator(field)
        observables = Observables.parse_observables(io_options.observables_file, constant_options, equation_options)
        observables_evaluated = {k:np.array(np.sum(v)) for k,v in observables.evaluate(fields_and_derivatives, grid_information).items()}
        observables_evaluated["CSG"] /= (grid_sizes[0]*grid_sizes[1]*(grid_sizes[2]-1))
        observables_evaluated["parameters"] = np.array(list(constants.values()))
        observed_evaluated.append(observables_evaluated)

    output = parsed.output

    with h5py.File(output, "w") as output_file:
        for i, dat in enumerate(observed_evaluated):
            iform = f"{i:04d}"
            grp = output_file.create_group(iform)
            for dsetkey,dsetval in dat.items():
                grp.create_dataset(dsetkey, data=dsetval)

    return


def load_seed_and_constants_from_h5(filename : Path):
    with h5py.File(filename, "r") as data_file:
        parameters = data_file["parameters"][:]
        field = data_file["result"][:].transpose((3,2,1,0))
        petsc_constant_ordering = ["mu","mu1", "ax", "ay", "lx", "ly", "nperiodsx", "nperiodsy", "phasex", "phasey", "B", "c1"]
        ret_dict = {a:b for a,b in  zip(petsc_constant_ordering, parameters)}
    return field, ret_dict


if __name__ == '__main__':
    EvaluateObservable()
