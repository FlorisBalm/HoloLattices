import itertools
import logging
import dataclasses
import numpy as np
import scipy.sparse
import typing

# My own
import HolographicLattices.Equations.EquationsBase
import HolographicLattices.SystemOnBackground.BackgroundOptions
import HolographicLattices.Utilities.GridUtilities
import HolographicLattices.Options.Options


from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

PathLike = typing.Union[str, Path]


@dataclass
class JacobianOnBackground(HolographicLattices.Equations.EquationsBase.EquationsBase):
    equation_options: HolographicLattices.Options.Options.EquationOptions
    constant_options: HolographicLattices.Options.Options.ConstantOptions
    background_options: HolographicLattices.SystemOnBackground.BackgroundOptions.BackgroundOptions

    finite_difference_matrices : List[scipy.sparse.csr_matrix]

    rhs_internal_equations: Any = dataclasses.field(init=False)
    rhs_boundary_equations: List = dataclasses.field(init=False)

    def evaluate(self, fields_and_derivatives: np.ndarray,
                 bg_fields_derivs : np.ndarray,
                 grid_information: HolographicLattices.Utilities.GridUtilities.GridInformation):

        logger = logging.getLogger(__name__)

        if len(fields_and_derivatives.shape) != 3:
            raise ValueError(f"Wrong shape of fields and derivatives. Expected length 3, got: "
                             f"{fields_and_derivatives.shape}")

        coordinates = self.equation_options.coordinates
        field_name = self.equation_options.field
        dtype_equations = self.equation_options.field_dtype
        bg_field_name = self.background_options.background_field_name
        # These are the indices of the boundaries for 1-d flattened coordinates
        # These are used for indexing into the arrays of points
        # The boundary variable is a list of pairs that for the "0" and "1" end of the boundary,
        # you can interpret that for "left" and "right" or whatever seems appropriate. For periodic
        # boundaries, the indices are None

        boundary_indices, bulk_indices = HolographicLattices.Utilities.GridUtilities.get_boundaries_and_bulk_indices(
            grid_information.options.grid_sizes,
            grid_information.options.grid_periodic)

        # Create a local dict of variables to only evaluate in the bulk points
        full_grids = [g.ravel() for g in np.meshgrid(*grid_information.get_1d_grids(), indexing="ij")]

        input_data = {field_name: np.array(fields_and_derivatives), bg_field_name: np.array(bg_fields_derivs),
                      **{name: grid for name, grid in zip(coordinates,
                                                          full_grids)}}

        bulk_variables = {}

        for ax in coordinates:
            bulk_variables[ax] = input_data[ax][bulk_indices]

        bulk_variables[bg_field_name] = input_data[bg_field_name][:, :, bulk_indices]
        bulk_variables[field_name] = input_data[field_name][:, :, bulk_indices]

        grid_volume = self.finite_difference_matrices[0].shape[0]

        # bulk_indices_coo_format = np.empty((bulk_indices.shape[0], 2), dtype=bulk_indices.dtype)
        # bulk_indices_coo_format[:, 0] = bulk_indices % grid_volume
        # bulk_indices_coo_format[:, 1] = np.rint(bulk_indices - bulk_indices_coo_format[:, 0]) / grid_volume
        finite_diff_mats = self.finite_difference_matrices

        # Empty matrices to construct the differentiation matrices in
        # We can see the diff matrix as a huge num_eqs_of_motion x nField block of gridvolume x gridvolume blocks, which
        # can be inverted to find the derivative.

        # Go from Field - Deriv - EOM to field - EOM - Deriv, this makes more sense here.

        internal_coefficients = np.array(self.internal_equations)

        use_dense_array = isinstance(finite_diff_mats[0], (np.ndarray, np.generic))
        if use_dense_array:
            preallocated_matrices = np.zeros((self.equation_options.num_fields, self.equation_options.num_eqs_of_motion, grid_volume,grid_volume),
                                             dtype = self.equation_options.field_dtype)
        else:
            preallocated_matrices = [
                [scipy.sparse.csr_matrix(
                    (grid_volume, grid_volume)
                    , dtype=dtype_equations)
                    for _ in range(self.equation_options.num_fields)] for _ in
                range(self.equation_options.num_eqs_of_motion)]


        for i, field in enumerate(internal_coefficients):
            for j, deriv in enumerate(field):
                for k, eom in enumerate(deriv):
                    term_eval = eom(**bulk_variables)

                    if np.isscalar(term_eval):
                        term_eval = np.zeros(len(bulk_indices), dtype=dtype_equations) + term_eval

                    if use_dense_array:
                        mat = np.zeros(grid_volume, dtype = self.equation_options.field_dtype)
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
                    boundary_variables[bg_field_name] = input_data[bg_field_name][:, :, idx]

                    coeffs = np.array(self.boundary_equations[axis][end])

                    # The coefficients are stored as field : deriv : eom for each.
                    for i, field in enumerate(coeffs):
                        for j, deriv in enumerate(field):
                            for k, eom in enumerate(deriv):
                                term_eval = eom(**boundary_variables)


                                if np.isscalar(term_eval):
                                    term_eval = np.zeros(len(idx), dtype=dtype_equations) + term_eval

                                if use_dense_array:
                                    mat = np.zeros((grid_volume), dtype = self.equation_options.field_dtype)
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

    def load_from_folder(self, folder: Path, periodicities, file_base=""):
        return self._load_from_folder_impl(folder, periodicities, file_base)

    def evaluate_rhs(self, fields_and_derivatives, bg_fields_derivs, grid_information, output_prealloc = None):

        if output_prealloc is None:
            output_prealloc = np.empty((self.equation_options.num_fields, fields_and_derivatives.shape[-1]),
                                       dtype=self.equation_options.field_dtype)

        if len(fields_and_derivatives.shape) != 3:
            raise ValueError(f"Wrong shape of fields and derivatives. Expected length 3, got: "
                             f"{fields_and_derivatives.shape}")

        coordinates = self.equation_options.coordinates

        bg_field_name = self.background_options.background_field_name
        field_name = self.equation_options.field
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

        input_data = {field_name: np.array(fields_and_derivatives), bg_field_name: np.array(
            bg_fields_derivs, dtype=self.equation_options.field_dtype),
                      **{name: grid for name, grid in zip(coordinates,
                                                          full_grids)}}
        bulk_variables = {}
        for ax in coordinates:
            bulk_variables[ax] = input_data[ax][bulk_indices]

        bulk_variables[bg_field_name] = input_data[bg_field_name][:,:,bulk_indices]
        bulk_variables[field_name] = input_data[field_name][:,:,bulk_indices]

        dtype_equations = self.equation_options.field_dtype

        for i, eom in enumerate(self.rhs_internal_equations):
            term_eval = eom(**bulk_variables)
            if np.isscalar(term_eval):
                term_eval = np.zeros(len(bulk_indices), dtype=dtype_equations) + term_eval

            output_prealloc[i, bulk_indices] = term_eval

        # Evaluate and assign the equations of motion to the non-periodic boundary points
        for axis, periodic in enumerate(grid_information.options.grid_periodic):
            if not periodic:
                # Each non-periodic direction has two boundaries
                for end in range(2):

                    idx = boundary_indices[axis][end]
                    eoms = self.rhs_boundary_equations[axis][end]

                    # Evaluate these equations only where they apply. This should reduce
                    # computation time somewhat
                    boundary_variables = {}

                    for ax in coordinates:
                        boundary_variables[ax] = input_data[ax][idx]

                    boundary_variables[field_name] = input_data[field_name][:, :, idx]
                    boundary_variables[bg_field_name] = input_data[bg_field_name][:, :, idx]

                    for k, eom in enumerate(eoms):

                        term_eval = eom(**boundary_variables)
                        if np.isscalar(term_eval):
                            term_eval = np.zeros(len(idx), dtype=dtype_equations) + term_eval
                        output_prealloc[k][idx] = term_eval
            else:
                pass

        return output_prealloc

    def _load_from_folder_impl(self, folder: PathLike, periodicities, file_base: str = "Fermion"):
        folder_path = Path(folder)
        file_base_full = str((folder_path / file_base).resolve())
        if file_base == "":
            file_base_full="".join((file_base_full, "/"))

        internal_file_name = Path(f"{file_base_full}Coefs_I.txt")
        internal_rhs_file_name = Path(f"{file_base_full}RHS_I.txt")

        self.internal_equations = self.read_file(
            internal_file_name, self.equation_options, self.background_options, self.constant_options
        )

        self.rhs_internal_equations = self.read_file_rhs(
            internal_rhs_file_name, self.equation_options, self.background_options, self.constant_options
        )
        dimensions = len(periodicities)

        boundary_file_names = [
            [Path(f"{file_base_full}Coefs_{dim}_{end}.txt") for end in [0, 1]] if not periodicities[dim]
            else [None, None] for dim in range(dimensions)]

        boundary_rhs_file_names = [
            [Path(f"{file_base_full}RHS_{dim}_{end}.txt") for end in [0, 1]] if not periodicities[dim]
            else [None, None] for dim in range(dimensions)]

        self.boundary_equations = [[None, None] for _ in range(dimensions)]

        for dim in range(dimensions):
            for end in (0, 1):
                if boundary_file_names[dim][end] is not None:
                    self.boundary_equations[dim][end] = \
                        self.read_file(boundary_file_names[dim][end],
                                                    self.equation_options,
                                                    self.background_options,
                                                    self.constant_options)

        self.rhs_boundary_equations = [[None, None] for _ in range(dimensions)]
        for dim in range(dimensions):
            for end in (0, 1):
                if boundary_rhs_file_names[dim][end] is not None:
                    self.rhs_boundary_equations[dim][end] = \
                        self.read_file_rhs(boundary_rhs_file_names[dim][end],
                                                        self.equation_options,
                                                        self.background_options,
                                                        self.constant_options)

    @classmethod
    def read_file(cls, file_name : PathLike,
                  equation_options : HolographicLattices.Options.Options.EquationOptions,
                  background_options : HolographicLattices.SystemOnBackground.BackgroundOptions.BackgroundOptions,
                  constant_options : HolographicLattices.Options.Options.ConstantOptions):
        logger = logging.getLogger(__name__)
        logger.debug(f"Trying to open {file_name}")

        file_path = Path(file_name)
        try:
            with file_path.open("r") as infile:
                lines = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

            num_eoms = equation_options.num_eqs_of_motion
            num_fields = num_eoms
            max_derivs = equation_options.max_deriv

            max_bg_deriv = background_options.max_background_deriv

            dims = equation_options.dims

            # All possible different derivatives in a problem with this dimensionality and highest
            # derivative.

            number_of_derivatives = len(
                HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dims, max_derivs))

            terms_in_eom = [[[None for _ in range(num_fields)] for _ in range(number_of_derivatives)] for _ in
                            range(num_eoms)]

            # this is for iterating over all eoms, fields and derivatives
            indices = itertools.product(range(num_eoms),
                                        range(number_of_derivatives), range(num_fields))
            # BG first, then constant options, because the second overrides the first.
            for key,value in constant_options.constants.items():
                if key in background_options.constants:
                    logger.debug(f"Option {key} was read in both background and settings. Using settings file value {value}")

            eval_dict = {
                         **background_options.constants,
                         **background_options.functions,
                         **constant_options.constants,
                         **constant_options.functions}

            for (eom, deriv, field_i), line in zip(indices, lines):
                params = ", ".join((background_options.background_field_name,
                                    equation_options.field,
                                    *equation_options.coordinates))
                lambda_string = " ".join(("lambda", params, ":", line))
                terms_in_eom[eom][deriv][field_i] = eval(lambda_string, eval_dict)

            return terms_in_eom

        except IOError as e:
            logger.error(f"Error reading file for fermion equations: {file_name}. Error:{e}")

    @classmethod
    def read_file_rhs(
            cls,
            file_name: PathLike,
            equation_options: HolographicLattices.Options.Options.EquationOptions,
            background_options: HolographicLattices.SystemOnBackground.BackgroundOptions.BackgroundOptions,
            constant_options : HolographicLattices.Options.Options.ConstantOptions):
        file = Path(file_name)
        logger = logging.getLogger(__name__)
        try:
            with file.open("r") as infile:
                # Ignore empty lines
                lines = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

                return_function = []
                for line in lines:
                    params = ", ".join((*equation_options.coordinates, background_options.background_field_name,
                                        equation_options.field))
                    line_as_lambda = " ".join(("lambda", params, ":", line))

                    eval_dict = {**background_options.constants,
                                 **background_options.functions,
                                 **constant_options.constants,
                                 **constant_options.functions}
                    # Evaluate with constants and functions given
                    #print(line_as_lambda)
                    fun = eval(line_as_lambda, eval_dict)
                    return_function.append(fun)
                return return_function

        except IOError as e:
            logger.error(f"Error reading Fermion RHS file: {file}. Error:{e}")
            raise e

