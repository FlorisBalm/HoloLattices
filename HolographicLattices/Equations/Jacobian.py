import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import itertools
import numpy as np
import functools

import scipy.sparse
import HolographicLattices.Utilities.GridUtilities
import HolographicLattices.Options.Options

import HolographicLattices.Equations.EquationsBase




@dataclass
class FiniteDifferenceJacobian(HolographicLattices.Equations.EquationsBase.EquationsBase):
    finite_difference_matrices: List[scipy.sparse.csr_matrix]


    """
    The jacobian is evaluated more efficiently when it is already applied to the
    """

    def evaluate(self, fields_and_derivatives: np.ndarray,
                 grid_information: HolographicLattices.Utilities.GridUtilities.GridInformation,
                 output_prealloc= None):

        """
        Evaluate the equations of motion for a given set of fields. This does not include the
        right-hand side
        :param fields_and_derivatives: Fields to evaluate
        :param grid_information: Options of the grid to evaluate on. Required for efficient obtaining of indices etc.
        :param output_prealloc: Optional preallocated output. Can save time.
        :return: equations of motion evaluated at each point for each field.
        """

        logger = logging.getLogger(__name__)

        if len(fields_and_derivatives.shape) != 3:
            raise ValueError(f"Wrong shape of fields and derivatives. Expected length 3, got: "
                             f"{fields_and_derivatives.shape}")



        coordinates = grid_information.options.coordinates
        field_name = grid_information.options.field
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

        input_data = {field_name: fields_and_derivatives,
                      **{name: grid for name, grid in zip(coordinates,
                                                          full_grids)}}
        bulk_variables = {}

        for ax in coordinates:
            bulk_variables[ax] = input_data[ax][bulk_indices]

        bulk_variables[field_name] = input_data[field_name][:, :, bulk_indices]

        grid_volume = self.finite_difference_matrices[0].shape[0]

        bulk_indices_coo_format = np.empty((bulk_indices.shape[0], 2), dtype=bulk_indices.dtype)
        bulk_indices_coo_format[:, 0] = bulk_indices % grid_volume
        bulk_indices_coo_format[:, 1] = np.rint(bulk_indices - bulk_indices_coo_format[:, 0]) / grid_volume

        dtype_equations = fields_and_derivatives.dtype
        finite_diff_mats = self.finite_difference_matrices

        use_dense_array = isinstance(finite_diff_mats[0], (np.ndarray, np.generic))

        # Empty matrices to construct the differentiation matrices in
        # We can see the diff matrix as a huge num_eqs_of_motion x nField block of gridvolume x gridvolume blocks, which
        # can be inverted to find the derivative.

        # Go from Field - Deriv - EOM to field - EOM - Deriv, this makes more sense here.
        internal_coefficients = np.array(self.internal_equations)#.transpose((2,1,0))

        if use_dense_array:
            preallocated_matrices = np.empty((self.equation_options.num_fields, self.equation_options.num_eqs_of_motion,grid_volume,grid_volume))
        else:
            preallocated_matrices = [
                [scipy.sparse.csr_matrix((grid_volume,grid_volume), dtype=dtype_equations)
                 for _ in range(self.equation_options.num_fields)] for _ in range(self.equation_options.num_eqs_of_motion)]


        for i, field in enumerate(internal_coefficients):
            for j, deriv in enumerate(field):
                for k, eom in enumerate(deriv):
                    term_eval = eom(**bulk_variables)

                    if np.isscalar(term_eval):
                        term_eval = np.zeros(len(bulk_indices), dtype=dtype_equations) + term_eval

                    if use_dense_array:
                        mat = np.zeros((grid_volume))
                        mat[bulk_indices] = term_eval
                        preallocated_matrices[i,k] += np.diag(mat).dot(finite_diff_mats[j])

                    else:
                        mat = scipy.sparse.coo_matrix((term_eval, (bulk_indices, bulk_indices)),
                                                      shape=(grid_volume, grid_volume))
                        preallocated_matrices[i][k] += (mat.tocsr().dot(finite_diff_mats[j]))

        for axis, periodic in enumerate(grid_information.options.grid_periodic):
            if not periodic:
                for end in range(2):
                    idx = boundary_indices[axis][end]

                    # Evaluate these equations only where they apply. This should reduce
                    # computation time somewhat
                    boundary_variables = {}

                    for ax in coordinates:
                        boundary_variables[ax] = input_data[ax][idx]

                    boundary_variables[field_name] = input_data[field_name][:, :, idx]

                    coeffs = np.array(self.boundary_equations[axis][end])#.transpose((2,1,0))

                    # The coefficients are stored as field : deriv : eom for each.
                    for i, field in enumerate(coeffs):
                        for j, deriv in enumerate(field):
                            for k, eom in enumerate(deriv):
                                term_eval = eom(**boundary_variables)

                                if np.isscalar(term_eval):
                                    term_eval = np.zeros(len(idx), dtype=dtype_equations) + term_eval

                                if use_dense_array:
                                    mat = np.zeros((grid_volume))
                                    mat[idx] = term_eval
                                    preallocated_matrices[i, k] += np.diag(mat).dot(finite_diff_mats[j])
                                else:
                                    mat = scipy.sparse.coo_matrix((term_eval, (idx, idx)),
                                                                  shape=(grid_volume, grid_volume))
                                    preallocated_matrices[i][k] += (mat.tocsr().dot(finite_diff_mats[j])).tocoo()

        if not use_dense_array:
            for ROW in range(len(preallocated_matrices)):
                for COL in range(len(preallocated_matrices[ROW])):
                    preallocated_matrices[ROW][COL].eliminate_zeros()


            h_stacked = [scipy.sparse.hstack(row, format="csr") for row in preallocated_matrices]
            full_matrix = scipy.sparse.vstack(h_stacked, format="csr")
        else:
            full_matrix = np.hstack(np.hstack(preallocated_matrices))

        return full_matrix

    def load_from_folder(self, folder: Path, periodicities, file_base="Coefs"):
        return self._load_from_folder_impl(folder, periodicities, file_base)

    @classmethod
    def read_file(cls, file: Path, equation_options, constant_options):
        """
        This defines how to read a set of equations from a file. It reads only one file
        :param file:
        :return:
        """
        logger = logging.getLogger(__name__)

        try:
            with file.open("r") as infile:

                # Strip any whitespace for parsing
                lines = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

                number_eqs_of_motion = equation_options.num_eqs_of_motion
                num_fields = number_eqs_of_motion
                max_derivs = equation_options.max_deriv
                dims = equation_options.dims

                # All possible different derivatives in a problem with this dimensionality and highest
                # derivative.

                number_of_derivatives = len(
                    HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dims, max_derivs))

                terms_in_eom = [[['0' for _ in range(num_fields)] for _ in range(number_of_derivatives)] for _ in
                                range(number_eqs_of_motion)]

                # this is for iterating over all eoms, fields and derivatives
                indices = itertools.product(range(number_eqs_of_motion),
                                            range(number_of_derivatives), range(num_fields))
                eval_dict = {**constant_options.constants, **constant_options.functions}

                for (eom, deriv, field_i), line in zip(indices, lines):
                    params = ", ".join((equation_options.field, *equation_options.coordinates))
                    lambda_string = " ".join(("lambda", params, ":", line))
                    terms_in_eom[eom][deriv][field_i] = eval(lambda_string, eval_dict)

                return terms_in_eom

        except IOError as e:
            logger.error(f"Error reading file in read_file_coeffs: {file}. Error:{e}")
            raise e


# @dataclass
# class MixedIVPBVPJacobian(HolographicLattices.Equations.EquationsBase.EquationsBase):
#
#     def evaluate(self, fields_and_derivatives, output_prealloc):
#         pass
#
#
#     def load_from_folder(self, folder: Path, periodicities, file_base):
#         return self._load_from_folder_impl(folder, periodicities, file_base)
#
#     def _load_from_folder_impl(self, folder: Path, periodicities, file_base: str = None):
#         dimensions = len(periodicities)
#         file_base_full = str((Path(folder) / file_base).resolve())
#         internal_file_name = Path("_".join((file_base_full,"I.txt")))
#         self.internal_equations = self.read_file(internal_file_name, self.equation_options,self.constant_options)
#         import yaml
#         boundary_file_name =  Path(f"{file_base_full}/BoundaryEquations.yaml")
#         self.boundary_equations = [[[None],[None]] for dim in range(dimensions)]
#         parsed_boundary_equations = yaml.load(boundary_file_name)
#         for dim in range(dimensions):
#             for end in (0, 1):
#                 for field
#                 if boundary_file_names[dim][end] is not None:
#                     self.boundary_equations[dim][end] = self.read_file(boundary_file_names[dim][end],
#                                                                        self.equation_options, self.constant_options)
#
#     @classmethod
#     @abstractmethod
#     def read_file(cls, path, equation_options, constant_options):
#         pass
