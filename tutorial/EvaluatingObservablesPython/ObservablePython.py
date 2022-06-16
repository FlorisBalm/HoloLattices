import argparse
import yaml
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

    parser.add_argument("--output", type=str, nargs="?", help="Where to output")
    parser.add_argument("--observables_file", type=str, help= "Which observables to evaluate")
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



    observables_file = parsed.observables_file
    print(parsed.files)
    output = parsed.output

    with h5py.File(output, "w") as output_file:
        for i,file_input in enumerate(parsed.files):

            field, constant_options, equation_options = load_python_style_file(file_input)
            grid_information = GridUtilities.GridInformation(equation_options)
            logger.info("Using finite differences everywhere, even for "
                        "spectral dimensions in the construction of the jacobian.")
            differential_operator = MixedFDDSparseSpectralDifferentialOperator.construct_differentiation_matrices(
                equation_options=equation_options, max_deriv = equation_options.max_deriv
            )

            fields_and_derivatives = differential_operator(field)
            observables = Observables.parse_observables(observables_file, constant_options, equation_options)
            observables_evaluated = {k:np.array(v) for k,v in observables.evaluate(fields_and_derivatives, grid_information).items()}



            iform = f"{i:04d}"
            grp = output_file.create_group(iform)

            grp.create_dataset("filename", data=file_input)
            grp.create_dataset("constants", data= yaml.dump(constant_options.constants))
            for dsetkey,dsetval in observables_evaluated.items():
                grp.create_dataset(dsetkey, data=dsetval)


    return

def load_python_style_file(filename: Path):
    with h5py.File(filename, "r") as data_file:
        field = data_file["field"][:,0,:]
        constants =  yaml.load(data_file["options/constant_options"][()], Loader=yaml.Loader)
        constant_options = Options.ConstantOptions()
        constant_options.constants = {**constant_options.constants, **constants}

        equation_options =yaml.load(data_file["options/equation_options"][()], Loader=yaml.Loader) 

        equation_options.max_deriv = 4 # account for all possible derivatives
        return field, constant_options, equation_options

if __name__ == '__main__':
    #load_python_style_file("data/OutputSpectralDense.h5")
    EvaluateObservable()
