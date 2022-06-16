import itertools
import logging
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

import numpy as np
import scipy.sparse

import HolographicLattices.Equations.EquationsBase
import HolographicLattices.Fermions.FermionOptions
import HolographicLattices.Utilities.GridUtilities


@dataclass
class FermionDifferentialOperator(HolographicLattices.Equations.EquationsBase.EquationsBase):
    equation_options: HolographicLattices.Fermions.FermionOptions.FermionOptions
    constant_options: HolographicLattices.Fermions.FermionOptions.FermionConstantOptions
    finite_difference_matrices : List[scipy.sparse.csr_matrix]

    rhs_internal_equations: Any = dataclasses.field(init=False)
    rhs_boundary_equations: List = dataclasses.field(init=False)

    def evaluate(self, fields_and_derivatives, output_prealloc):
        raise NotImplementedError("Not implemented. Use evaluate_fermion_op")

    def evaluate_fermion_op(self, fields_and_derivatives: np.ndarray,
                 grid_information: HolographicLattices.Utilities.GridUtilities.GridInformation,
                 momentum_point):

        mom_point_dict = {k:v for k,v in zip(self.equation_options.fermion_params, momentum_point)}
        logger = logging.getLogger(__name__)

        if len(fields_and_derivatives.shape) != 3:
            raise ValueError(f"Wrong shape of fields and derivatives. Expected length 3, got: "
                             f"{fields_and_derivatives.shape}")

        coordinates = self.equation_options.coordinates
        field_name = self.equation_options.background_field
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

        input_data = {field_name: np.array(fields_and_derivatives, dtype=np.complex128),
                      **{name: grid for name, grid in zip(coordinates,
                                                          full_grids)}}

        bulk_variables = {**mom_point_dict}

        for ax in coordinates:
            bulk_variables[ax] = input_data[ax][bulk_indices]

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

        matrices = [
            [scipy.sparse.csr_matrix(
                (grid_volume, grid_volume)
                , dtype = np.complex128)
             for _ in range(self.equation_options.num_fields)] for _ in range(self.equation_options.num_eqs_of_motion)]

        for i, field in enumerate(internal_coefficients):
            for j, deriv in enumerate(field):
                for k, eom in enumerate(deriv):
                    term_eval = eom(**bulk_variables)
                    if not hasattr(term_eval, "shape"):
                        term_eval = np.zeros(len(bulk_indices), dtype=self.equation_options.field_dtype) + term_eval

                    mat = scipy.sparse.coo_matrix((term_eval, (bulk_indices, bulk_indices)),
                                                  shape=(grid_volume, grid_volume),dtype=np.complex128)
                    matrices[i][k] += (mat.tocsr().dot(finite_diff_mats[j]))

        for axis, periodic in enumerate(grid_information.options.grid_periodic):
            if not periodic:
                for end in range(2):
                    idx = boundary_indices[axis][end]

                    # Evaluate these equations only where they apply. This should reduce
                    # computation time somewhat
                    boundary_variables = {**mom_point_dict}

                    for ax in coordinates:
                        boundary_variables[ax] = input_data[ax][idx]

                    boundary_variables[field_name] = input_data[field_name][:, :, idx]

                    coeffs = np.array(self.boundary_equations[axis][end])

                    # The coefficients are stored as field : deriv : eom for each.
                    for i, field in enumerate(coeffs):
                        for j, deriv in enumerate(field):
                            for k, eom in enumerate(deriv):
                                term_eval = eom(**boundary_variables)
                                # if np.sum(np.abs(term_eval ))> 10**6:
                                    # print(boundary_variables)
                                    # print(coeffs[i,j,k])
                                    # # print(boundary_variables)
                                    # print(f"{i},{j},{k},{term_eval}")
                                if not hasattr(term_eval, "shape"):
                                    # if np.abs(term_eval) < 1e-9:
                                    #     continue
                                    term_eval = np.zeros(len(idx), dtype=self.equation_options.field_dtype) + term_eval
                                else:
                                    mat = scipy.sparse.coo_matrix((term_eval, (idx, idx)),
                                                                  shape=(grid_volume, grid_volume),dtype=np.complex128)
                                    matrices[i][k] += (mat.dot(finite_diff_mats[j]))

        # for ROW in range(len(matrices)):
        #     for COL in range(len(matrices[ROW])):
        #         matrices[ROW][COL].eliminate_zeros()

        h_stacked = [scipy.sparse.hstack(row, format="csr") for row in matrices]
        full_matrix = scipy.sparse.vstack(h_stacked, format="csr")
        return full_matrix

    def load_from_folder(self, folder: Path, periodicities, file_base="Fermion"):
        return self._load_from_folder_impl(folder, periodicities, file_base)


    def evaluate_rhs(self, fields_and_derivatives, grid_information, output_prealloc, momentum_point, source):

        mom_point_dict = {k: v for k, v in zip(self.equation_options.fermion_params, momentum_point)}
        print(mom_point_dict)
        if output_prealloc is None:
            output_prealloc = np.empty((self.equation_options.num_fields, fields_and_derivatives.shape[-1]),
                                       dtype=self.equation_options.field_dtype)

        if len(fields_and_derivatives.shape) != 3:
            raise ValueError(f"Wrong shape of fields and derivatives. Expected length 3, got: "
                             f"{fields_and_derivatives.shape}")

        coordinates = self.equation_options.coordinates
        # field = self.equation_options.background_field

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

        input_data = {#self.equation_options.background_field: fields_and_derivatives,
                      **{name: grid for name, grid in zip(coordinates,
                                                          full_grids)}}
        bulk_variables = {**mom_point_dict, "source": source}

        for ax in coordinates:
            bulk_variables[ax] = input_data[ax][bulk_indices]

        #bulk_variables[field] = input_data[field][:, :, bulk_indices]

        dtype_equations = self.equation_options.field_dtype
        # Evaluate and assign the equations of motion to the internal points

        for i, eom in enumerate(self.rhs_internal_equations):

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
                    eoms = self.rhs_boundary_equations[axis][end]

                    # Evaluate these equations only where they apply. This should reduce
                    # computation time somewhat
                    boundary_variables = {**mom_point_dict, "source": source}

                    for ax in coordinates:
                        boundary_variables[ax] = input_data[ax][idx]

                    #boundary_variables[field] = input_data[field][:, :, idx]

                    for k, eom in enumerate(eoms):

                        term_eval = eom(**boundary_variables)
                        if not hasattr(term_eval, "shape"):
                            term_eval = np.zeros(len(idx), dtype=dtype_equations) + term_eval
                        output_prealloc[k][idx] = term_eval
            else:
                pass

        return output_prealloc

    def _load_from_folder_impl(self, folder: Path, periodicities, file_base: str = "Fermion"):
        folder_path = Path(folder)
        file_base_full = str((folder_path / file_base).resolve())

        internal_file_name = Path(f"{file_base_full}Coefs_I.txt")
        internal_rhs_file_name = Path(f"{file_base_full}RHS_I.txt")

        self.internal_equations = self.read_file(
            internal_file_name, self.equation_options, self.constant_options
        )

        self.rhs_internal_equations = self.read_file_rhs(
            internal_rhs_file_name, self.equation_options, self.constant_options
        )
        dimensions = len(periodicities)

        boundary_file_names = [
            [Path(f"{file_base_full}Coefs_{dim}_{end}.txt") for end in [0, 1]] if not periodicities[dim]
            else [None, None] for dim in range(dimensions)]

        boundary_rhs_file_names = [
            [Path(f"{file_base_full}RHS_{dim}_{end}.txt") for end in [0, 1]] if not periodicities[dim]
            else [None, None] for dim in range(dimensions)]

        self.boundary_equations = [[None, None] for dim in range(dimensions)]

        for dim in range(dimensions):
            for end in (0, 1):
                if boundary_file_names[dim][end] is not None:
                    self.boundary_equations[dim][end] = self.read_file(boundary_file_names[dim][end],
                                                                       self.equation_options, self.constant_options)

        self.rhs_boundary_equations = [[None, None] for dim in range(dimensions)]
        for dim in range(dimensions):
            for end in (0, 1):
                if boundary_rhs_file_names[dim][end] is not None:
                    self.rhs_boundary_equations[dim][end] = self.read_file_rhs(boundary_rhs_file_names[dim][end],
                                                                               self.equation_options,
                                                                               self.constant_options)

    @classmethod
    def read_file(cls, file_name, equation_options, constant_options):
        logger = logging.getLogger(__name__)
        logger.debug(f"Trying to open {file_name}")

        file_path = Path(file_name)
        try:
            with file_path.open("r") as infile:
                lines = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

            num_fermion_eqs = equation_options.num_eqs_of_motion
            num_fields = num_fermion_eqs
            max_derivs = equation_options.max_deriv
            max_bg_deriv = equation_options.max_bg_deriv

            dims = equation_options.dims

            # All possible different derivatives in a problem with this dimensionality and highest
            # derivative.

            number_of_derivatives = len(
                HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dims, max_derivs))

            terms_in_eom = [[['0' for _ in range(num_fields)] for _ in range(number_of_derivatives)] for _ in
                            range(num_fermion_eqs)]

            # this is for iterating over all eoms, fields and derivatives
            indices = itertools.product(range(num_fermion_eqs),
                                        range(number_of_derivatives), range(num_fields))
            eval_dict = {**constant_options.constants, **constant_options.functions}

            for (eom, deriv, field_i), line in zip(indices, lines):
                params = ", ".join((equation_options.background_field, *equation_options.coordinates,
                                    *equation_options.fermion_params))
                lambda_string = " ".join(("lambda", params, ":", line))
                # print(lambda_string)
                # print(eval_dict)
                terms_in_eom[eom][deriv][field_i] = eval(lambda_string, eval_dict)

            return terms_in_eom

        except IOError as e:
            logger.error(f"Error reading file for fermion equations: {file_name}. Error:{e}")

    @classmethod
    def read_file_rhs(cls, file_name, equation_options, constant_options):
        file = Path(file_name)
        logger = logging.getLogger(__name__)
        try:
            with file.open("r") as infile:

                # Ignore empty lines
                lines = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

                return_function = []
                for line in lines:
                    params = ", ".join(
                        (*equation_options.coordinates, *equation_options.fermion_params,
                         "source"))
                    line_as_lambda = " ".join(("lambda", params, ":", line))

                    # Evaluate with constants and functions given
                    fun = eval(line_as_lambda, {**constant_options.constants, **constant_options.functions})
                    return_function.append(fun)
                return return_function

        except IOError as e:
            logger.error(f"Error reading Fermion RHS file: {file}. Error:{e}")
            raise e


@dataclass
class FermionDifferentialOperatorOldStyle(HolographicLattices.Equations.EquationsBase.EquationsBase):
    fermion_options: HolographicLattices.Fermions.FermionOptions.FermionOptions

    rhs_matrix_parts: List[scipy.sparse.csr_matrix]

    def __init__(self):
        raise NotImplementedError("Not implemented yet")

    def get_rhs(self, omega, kx, ky):
        pass

    def _load_from_folder_impl(self):
        pass
