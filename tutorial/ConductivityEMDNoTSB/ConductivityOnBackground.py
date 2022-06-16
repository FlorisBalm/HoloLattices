import argparse
import logging

# This makes warnings be sent to the log file instead of to the console output.
logging.captureWarnings(True)

from pathlib import Path

import h5py
import numpy as np
import scipy
import scipy.sparse

import pathos.multiprocessing as multiprocess

from HolographicLattices.DifferentialOperators.DifferentialOperator import *

from HolographicLattices.LinearSolver.SimpleLinearSolver import SimpleLinearSolver

from HolographicLattices.Options import Options
from HolographicLattices.SystemOnBackground import JacobianOnBackground
from HolographicLattices.SystemOnBackground.BackgroundOptions import BackgroundOptions
from HolographicLattices.SystemOnBackground.PerturbationObservables import PerturbationObservables
from HolographicLattices.Utilities import GridUtilities

def SolveLinearBackground(**kwargs):

    # Basic argparse for some settings. Can be extended, but not required.
    parser = argparse.ArgumentParser()

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sparse", action="store_true", help="approximate any spectral matrix with its finite difference counterpart in the jacobian")
    grp.add_argument("--dense", action="store_true", help="Use fully spectral matrices for the jacobian. This may increase runtimes a lot!")

    parsed_options, _ = parser.parse_known_args()

    # Load all the options
    setupfile = kwargs["setupfile"]


    io_options = Options.IOOptions.load_from_file(setupfile)
    equation_options = Options.EquationOptions.load_from_file(setupfile)
    background_options = BackgroundOptions.load_from_file(setupfile)
    constant_options = Options.ConstantOptions.load_from_file(setupfile)

    for k,v in kwargs.items():
        if k in constant_options.constants:
            constant_options.constants[k] = v

    grid_information = GridUtilities.GridInformation(equation_options)

    if parsed_options.dense:
        bg_differential_operator = MixedFDDSpectralDifferentialOperator.construct_differentiation_matrices(
            equation_options=equation_options, max_deriv=background_options.max_bg_deriv
        )
        differential_operator = MixedFDDSpectralDifferentialOperator.construct_differentiation_matrices(
             equation_options=equation_options, max_deriv=2
        )
        linear_solver = SimpleLinearSolver()
    else:
        assert parsed_options.sparse, "Somehow the arguments ot mixed up. This is bad, check this."
        differential_operator = MixedFDDSparseSpectralDifferentialOperator.construct_differentiation_matrices(
            equation_options=equation_options, max_deriv=2
        )
        bg_differential_operator = MixedFDDSparseSpectralDifferentialOperator.construct_differentiation_matrices(
            equation_options=equation_options, max_deriv=background_options.max_bg_deriv
        )
        import scipy
        import scipy.sparse.linalg
        linear_solver = SimpleLinearSolver(scipy.sparse.linalg.spsolve)

    jacobian = JacobianOnBackground.JacobianOnBackground(constant_options=constant_options,
                                                         equation_options=equation_options,
                                                         background_options=background_options,
                                                         finite_difference_matrices=
                                                                differential_operator.get_matrix_representation())

    jacobian.load_from_folder(folder=io_options.coefficient_folder, periodicities=equation_options.grid_periodic)

    grid_volume = np.prod(equation_options.grid_sizes)

    # Get horizon values of field

    background_options.background_field = background_options.background_field.reshape(
        (background_options.background_field.shape[0],
         np.prod(background_options.background_field.shape[1:])))

    bg_fields_derivatives = differential_operator(background_options.background_field)


    fields_zero = np.zeros((equation_options.num_eqs_of_motion, grid_volume),dtype=equation_options.field_dtype)
    fields_derivatives_zero = differential_operator(fields_zero)

    PDE_Linear = jacobian.evaluate(fields_derivatives_zero, bg_fields_derivatives, grid_information)

    rhs = jacobian.evaluate_rhs(fields_derivatives_zero, bg_fields_derivatives, grid_information)

    solution = linear_solver.solve(PDE_Linear, rhs)

    fields_and_derivatives = differential_operator(solution)

    observables = PerturbationObservables.parse_observables(io_options.observables_file,
                                                            constant_options=constant_options,
                                                            background_options=background_options,
                                                            equation_options=equation_options)

    observables_eval = observables.evaluate(fields_and_derivatives, bg_fields_derivatives, grid_information)

    shape = grid_information.options.grid_sizes

    return {**{i:np.average(j) for i,j in observables_eval.items()},**constant_options.constants,"filename": background_options.background_filename}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite_file", action="store_true", default=False)
    parser.add_argument("--setup_replace", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--no_use_pathos", action="store_true", default=False)
    parser.add_argument("--backgrounds", metavar='N', type=str, nargs='+',
                     help='all backgrounds to run for')
    parser.add_argument("--log", type=str, default="INFO", help="Logging level, choose one of"
                                                                  " DEBUG|INFO|WARN|ERROR|CRITICAL.")
    parser.add_argument("--logfile", type=str, default="out.log", help="Default output log file")

    parsed, _ = parser.parse_known_args()

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", filename=parsed.logfile, level=parsed.log)

    import pandas
    dataframe = pandas.DataFrame()
    data = None


    with open(parsed.setup_replace, "r") as sources:
        lines = sources.readlines()
    import tempfile, re
    for idx,background in enumerate(parsed.backgrounds):
        print(f"Solving {idx+1} out of {len(parsed.backgrounds)}\nBackground= {background}")

        from numpy import sqrt as Sqrt
        from numpy import pi as Pi

        # with to have file deleted when the run is done
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            # fix the background
            for line in lines:
                temp_file.write(re.sub("REPLACE_BG", background, line))

            temp_file.flush()

            constant_options = HolographicLattices.Options.Options.ConstantOptions.load_from_file(temp_file.name)

            # I do not normalise to temperature currently, but that doesn't really matter at this point. It's easy enough to figure out.
            # You do something like above, where constant_options are loaded from now temp_file, so that you can access
            # constant_options.constants["mu"], or you can try directly accessing the background to get to mu

            nsteps = constant_options.constants.get("nsteps", 100)
            omegamin = constant_options.constants.get("omegamin", 0.001)
            omegamax = constant_options.constants.get("omegamax", 20)

            # Pathos allows for a multiprocessing speedup: quite a large factor in this case
            # if it doesn't work well, add flag --no_use_pathos (see --help) to False
            no_use_pathos = parsed.no_use_pathos

            if not no_use_pathos:
                # Multiprocess from Pathos does allow for easy pickling using dill instead of the builtin pickle
                with multiprocess.Pool(None) as pool:
                    class OmegaSolver:
                        def __init__(self, temp_name, output, bg):
                            self.temp_name = temp_name
                            self.output = output
                            self.bg = bg 

                        def __call__(self,omega):
                            data = SolveLinearBackground(setupfile=self.temp_name, output=self.output, background=self.bg,omega=omega)
                            return data
                    solver_obj = OmegaSolver(temp_name = temp_file.name, output = None, bg = background)
                    result = pool.map(solver_obj, np.linspace(omegamin, omegamax, nsteps))

                dataframe = pandas.DataFrame(result)

            else:
                result  = [SolveLinearBackground( setupfile=temp_file.name,output=None, background=background,omega=o) for o in np.linspace(omegamin, omegamax, nsteps)]
            
                dataframe = pandas.DataFrame(result)


    import os
    if os.path.isfile(parsed.output_file) and not parsed.overwrite_file:
        import uuid
        output_csv_filename = str(uuid.uuid4())
        print(f"Warning! File already exists. Saving to {output_csv_filename}")
    else:
        output_csv_filename = parsed.output_file

    columns_not_to_save = ["pi", "omegamin", "omegamax", "nomega"]
    # remove superfluous columns
    df_to_save = dataframe.drop(columns = [i for i in columns_not_to_save if i in dataframe.columns])
    print("Saving to file: ", output_csv_filename)
    df_to_save.to_csv(output_csv_filename, sep="\t", index=False)

if __name__ == '__main__':
    main()
