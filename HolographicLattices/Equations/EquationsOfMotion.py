import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

import HolographicLattices.Utilities.GridUtilities
import HolographicLattices.Equations.EquationsBase

@dataclass
class EquationsOfMotion(HolographicLattices.Equations.EquationsBase.EquationsBase):

    def evaluate(self, fields_and_derivatives: np.ndarray,
                 grid_information: HolographicLattices.Utilities.GridUtilities.GridInformation,
                 output_prealloc: np.ndarray = None):
        """
        Evaluate the equations of motion for a given set of fields. This does not include the
        right-hand side
        :param fields_and_derivatives: Fields to evaluate
        :param grid_information: Options of the grid to evaluate on. Required for efficient obtaining of indices etc.
        :param output_prealloc: Optional preallocated output. Can save time.
        :return: equations of motion evaluated at each point for each field.
        """
        # Get these variables so they can be passed as "locals" to eval
        if output_prealloc is None:
            output_prealloc = np.empty((fields_and_derivatives.shape[0], fields_and_derivatives.shape[-1]),
                                       dtype=fields_and_derivatives.dtype)

        if len(fields_and_derivatives.shape) != 3:
            raise ValueError(f"Wrong shape of fields and derivatives. Expected length 3, got: "
                             f"{fields_and_derivatives.shape}")
        coordinates = self.equation_options.coordinates
        field = self.equation_options.field

        # These are the indices of the boundaries for 1-d flattened coordinates
        # These are used for indexing into the arrays of points
        # The boundary variable is a list of pairs that for the "0" and "1" end of the boundary,
        # you can interpret that for "left" and "right" or whatever seems appropriate. For periodic
        # boundaries, the indices are None
        boundary_indices, bulk_indices = HolographicLattices.Utilities.GridUtilities.get_boundaries_and_bulk_indices(
            grid_information.options.grid_sizes,
            grid_information.options.grid_periodic)

        # Create a local dict of variables to only evaluate in the bulk points
        full_grids = [grid.ravel() for grid in np.meshgrid(*grid_information.get_1d_grids(), indexing="ij")]

        input_data = {self.equation_options.field: fields_and_derivatives,
                      **{name: grid for name, grid in zip(coordinates,
                                                          full_grids)}}
        bulk_variables = {}

        for ax in coordinates:
            bulk_variables[ax] = input_data[ax][bulk_indices]

        bulk_variables[field] = input_data[field][:, :, bulk_indices]

        dtype_equations = fields_and_derivatives.dtype
        # Evaluate and assign the equations of motion to the internal points

        for i, eom in enumerate(self.internal_equations):
            term_eval = eom(**bulk_variables)
            if not hasattr(term_eval, "shape"):
                term_eval = np.zeros(len(bulk_indices), dtype=dtype_equations) + term_eval

            output_prealloc[i, bulk_indices] = term_eval

        # Evaluate and assign the equations of motion to the non-periodic boundary points
        for axis, periodic in enumerate(grid_information.options.grid_periodic):
            if not periodic:
                # Each non-periodic direction has two boundaries
                for end in range(2):

                    idx = boundary_indices[axis][end]
                    eoms = self.boundary_equations[axis][end]

                    # Evaluate these equations only where they apply. This should reduce
                    # computation time somewhat
                    boundary_variables = {}

                    for ax in coordinates:
                        boundary_variables[ax] = input_data[ax][idx]

                    boundary_variables[field] = input_data[field][:, :, idx]

                    for k, eom in enumerate(eoms):

                        term_eval = eom(**boundary_variables)

                        if not hasattr(term_eval, "shape"):
                            term_eval = np.zeros(len(idx), dtype=dtype_equations) + term_eval
                        output_prealloc[k][idx] = term_eval
            else:
                pass

        return output_prealloc

    def load_from_folder(self, folder: Path, periodicities: List[bool],
                         file_base: str = "EOMs"):

        return self._load_from_folder_impl(folder, periodicities, file_base)

    @classmethod
    def read_file(cls, file: Path, equation_options, constant_options):
        """
        Read an equations-of-motion file.
        :param Path: path to read  to read
        :return:
        """
        logger = logging.getLogger(__name__)
        try:
            with file.open("r") as infile:

                # Ignore empty lines
                lines = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]
                return_function = []
                for line in lines:
                    # Parameters are the input

                    params = ", ".join((*equation_options.field, *equation_options.coordinates))
                    line_as_lambda = " ".join(("lambda", params, ":", line))

                    # Evaluate with constants and functions given
                    fun = eval(line_as_lambda, {**constant_options.constants, **constant_options.functions})
                    return_function.append(fun)
                return return_function

        except IOError as e:
            logger.error(f"Error reading file: {file}. Error:{e}")
            raise e