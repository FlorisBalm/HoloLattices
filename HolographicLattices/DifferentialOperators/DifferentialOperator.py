from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Union
import scipy.sparse
import logging
import numpy as np
from HolographicLattices.Utilities.GridUtilities import get_all_derivatives
import scipy.fftpack
import HolographicLattices
import HolographicLattices.Options.Options
import HolographicLattices.FiniteDifferences.FiniteDifferenceMatrices
import scipy.fftpack as fftpack

"""
The DifferentialOperator class represents the action of a differential operator on the equations. 
While it can have a representation as a differentiation matrices, it  
"""


class BaseDifferentialOperator(metaclass=ABCMeta):
    """
    Abstract DifferentialOperator class. At least call and matrix rep need to be implemented.
    It can be that the matrix rep does not make sense, but that should then explicitly be implemented
    to raise a NotImplementedError (or something that conveys a similar meaning)
    """

    @abstractmethod
    def __call__(self, fields: np.ndarray):
        pass

    @abstractmethod
    def get_matrix_representation(self):
        pass


@dataclass
class FiniteDifferenceDifferentialOperator(BaseDifferentialOperator):
    """
    Generic class that implements a finite difference differential operator
    """
    _diff_matrices: List[Union[np.ndarray, scipy.sparse.csr_matrix]]

    def __call__(self, fields: np.ndarray, output_preallocated: np.ndarray = None):
        """
        Apply the finite difference operator to a set of fields. The output can be preallocated
        if desired.
        :param fields: Partially flattened array of the number of fields in the
        :param output_preallocated:
        :return:
        """
        assert len(fields.shape) == 2, "Fields should be in nFields * volume shape before calling this"

        output_dimensions = (len(fields), len(self._diff_matrices), len(fields[0]))

        # This allows for doing things in-place, which saves a lot of time
        if output_preallocated is not None:
            assert output_preallocated.shape == output_dimensions, \
                f"Preallocated array not of correct shape. Expected: " \
                f" {output_dimensions}, got: {output_preallocated.shape}"
        else:
            output_preallocated = np.empty(output_dimensions, dtype=fields.dtype)

        for field in range(len(fields)):
            # no need to act with unity
            output_preallocated[field, 0] = fields[field]

            for deriv in range(1, len(self._diff_matrices)):
                output_preallocated[field, deriv, :] = self._diff_matrices[deriv].dot(fields[field])

        return output_preallocated

    def get_matrix_representation(self):
        return self._diff_matrices


@dataclass
class StructuredFiniteDifferenceDifferentialOperator(FiniteDifferenceDifferentialOperator):
    # Note! differentiation matrices can be shared, as arrays are just pointers in python in the internal
    # C appliication. This is a good thing as it saves space.
    _diff_matrices: List[Union[np.ndarray, scipy.sparse.csr_matrix]]

    @classmethod
    def construct_differentiation_matrices(cls, equation_options: HolographicLattices.Options.Options.EquationOptions,
                                           max_deriv: int = 2):
        grid_sizes = equation_options.grid_sizes
        grid_domains = equation_options.grid_domains

        diff_order = equation_options.diff_order
        periodicities = equation_options.grid_periodic

        spacings = equation_options.grid_spacings

        diff_matrices = HolographicLattices.FiniteDifferences.FiniteDifferenceMatrices. \
            differentiation_matrices_structured(grid_sizes, difference_orders=diff_order,
                                                periodic_boundaries=periodicities, spacings=spacings,
                                                domains=grid_domains, max_derivative=max_deriv)

        return cls(_diff_matrices=diff_matrices)


@dataclass
class UnStructuredFiniteDifferenceDifferentialOperator(FiniteDifferenceDifferentialOperator):

    def __init__(self):
        raise NotImplementedError("This is not finished. no use seen for it in the near future, either."
                                  " This is here for completeness' sake to show a possible implementation.")


@dataclass
class ChebyshevSpectralDifferentialOperator(BaseDifferentialOperator):
    _diff_matrices: List[Union[np.ndarray, scipy.sparse.csr_matrix]]
    equation_options: HolographicLattices.Options.Options.EquationOptions

    @staticmethod
    def take_fft_deriv(data : np.ndarray,deriv_order,domain=1.0):
        """

        @param data: 1-D array to take the derivative of
        @param deriv_order: Order of the derivative to take, either 0,1,2
        @param dtype: Datatype (default: np.complex, but can be other)
        @param domain:
        @return:
        """
        logger = logging.getLogger(__name__)
        if deriv_order == 0:
            return data
        elif deriv_order==1:
            N = len(data) - 1

            x_values = np.cos(np.pi * np.linspace(0, 1, N + 1))

            # Reverse to match the algorithm
            # which requires points to be on  [1,..., -1]
            vector_v = np.hstack([data, data[-2:0:-1]])

            vector_u = fftpack.fft(vector_v)

            fft_freq = fftpack.fftfreq(len(vector_v), d=1 / (2 * N))

            w_hat = ((1.j * fft_freq)) * vector_u
            w_hat[N] = 0
            W = fftpack.ifft(w_hat)

            deriv_result = np.zeros(data.shape, dtype=np.complex128)
            deriv_result[1:N] = - W[1:N] / (np.sqrt(1 - x_values[1:N] ** 2))
            idx = np.arange(0, N, 1)
            deriv_result[0] = np.sum(idx ** 2 * vector_u[idx]) / N + 0.5 * N * vector_u[N]
            deriv_result[N] = np.sum((-1.0) ** (idx + 1) * idx ** 2 * vector_u[idx]) / N + 0.5 * (-1.0) ** (N + 1) * N * \
                              vector_u[N]
            # derivW[0] =-2*((1/(len(vector_v)))* np.sum((range_n**2)*vector_u[:N])+0.5*((N-1)**2)*vector_u[N-1])
            # derivW[N-1] = -2*((1/(len(vector_v)))*(np.sum((-1)**(range_n)*range_n**2*vector_u[:N])-0.5*((N-1)**2)*vector_u[N-1]))
            if np.isrealobj(data):
                return np.real((-(2.0/domain) * deriv_result))
            else:
                return (-(2.0/domain) * deriv_result)
        elif deriv_order == 2:
            N = len(data) - 1
            x_values = np.cos(np.pi * np.linspace(0, 1, N + 1))
            vector_v = np.hstack([data, data[-2:0:-1]])
            vector_u = fftpack.fft(vector_v)
            fft_freq = fftpack.fftfreq(len(vector_v), d=1 / (2 * N))

            w_hat_1 = ((1.j * fft_freq)) * vector_u
            w_hat_2 = (-1.0) * (fft_freq ** 2) * vector_u

            W1 = fftpack.ifft(w_hat_1)
            W2 = fftpack.ifft(w_hat_2)

            theta = np.arange(1, N, 1) * np.pi / N

            ii2 = np.arange(2, N, 1) ** 2.0

            n2b = ii2 * (ii2 - 1) * vector_u[2:N] / (N * 3)
            bM = N * (N ** 2 - 1) * vector_u[N] / 6.0

            w = np.zeros_like(data, dtype=np.complex128)
            w[0] = np.sum(n2b) + bM
            w[1:N] = 1.0 / (np.sin(theta) ** 2) * (W2[1:N] - W1[1:N] / np.tan(theta))
            w[N] = np.sum((-1.0) ** (np.arange(2, N, 1)) * n2b) + (-1.0) ** (N) * bM
            if np.isrealobj(data):
                return np.real((w * (2.0 / domain) ** 2))
            else:
                return (w * (2.0 / domain) ** 2)
        else:
            logger.debug(f"Deriv order too high: {deriv_order}, taking repeated derivatives")
            data_temp = np.copy(data)
            order_temp = deriv_order
            while order_temp > 0:
                if order_temp%2 == 0:
                    data_temp = ChebyshevSpectralDifferentialOperator.take_fft_deriv(data_temp, domain=domain, deriv_order=2)
                    order_temp = order_temp - 2
                else:
                    assert order_temp%2 == 1
                    data_temp = ChebyshevSpectralDifferentialOperator.take_fft_deriv(data_temp, domain=domain, deriv_order=1)
                    order_temp = order_temp -1
            return data_temp


    @staticmethod
    def take_dct_deriv_1d(data : np.ndarray, derivative : int, domain=1.0):
        """
        This is relasted to an article, D. Dunn, Comp Phys Comm., 96 (1996) 10-16
        This is the best method for small grids, where
        :param points:
        :param x0:
        :param x1:
        :return:
        """
        N = data.size - 1
        if derivative == 0:
            return data

        # Reverse to match the algorithm
        # which requires points to be on  [1,..., -1]
        points = data[::-1]

        transformed_coeffs = scipy.fftpack.dct(points, type=1) / N
        recursion_mat = np.zeros((derivative + 1, N + 1), dtype=data.dtype)
        recursion_mat[0, :] = transformed_coeffs
        for M in range(1, derivative + 1):
            for n in np.arange(N - M, -1, -1):
                if n == N - 1:
                    continue

                recursion_mat[M, n] = recursion_mat[M, n + 2] + 2 * (n + 1) * recursion_mat[M - 1, n + 1]

        derivatives = np.zeros(recursion_mat.shape)

        for M in range(0, derivative + 1):
            derivatives[M, :] = 0.5 * np.power(2 / (domain), M) * scipy.fftpack.dct(recursion_mat[M], type=1)
        return derivatives[derivative, ::-1]

    def __call__(self, fields: np.ndarray, output_preallocated: np.ndarray = None):
        """
        Differentation using FFT methods
        :param data: Data in 1d format
        :param sizes: Lenght of each interval, in usual definitions 2pi but can differ
        :param dimensions: number of gridpoints in each dimension
        :param periodic: Whether a dimension is periodic
        :param derivatives: Derivative to take. E.g. [0,1,2] corresponds to dy d2z
        :return: Derivative taken using pyfftw
        """
        logger= logging.getLogger(__name__)
        equation_options = self.equation_options
        num_fields = fields.shape[0]

        grid_sizes = equation_options.grid_sizes
        real_sizes = equation_options.grid_domains
        periodicity = equation_options.grid_periodic

        if any(periodicity):
            raise NotImplementedError("Chebyshev spectral derivative cannot deal with periodic domains")
            # Todo: Unmix notation of "Spectral" "Nonperiodic"  etc.
            # The real order should be: make th

        fields = fields.reshape((num_fields, *grid_sizes))
        all_derivatives = get_all_derivatives(equation_options.dims)

        if output_preallocated is None:
            output_preallocated = np.zeros((num_fields, len(all_derivatives), np.prod(grid_sizes)),
                                           dtype=equation_options.field_dtype)

        if np.iscomplexobj(equation_options.field_dtype(0)):
            logger.debug("Taking derivatives using complex numbers")
            for field in range(len(fields)):
                # no need to act with unity
                output_preallocated[field, 0,:] = fields[field]

                for deriv in range(1, len(self._diff_matrices)):
                    output_preallocated[field, deriv, :] = self._diff_matrices[deriv].dot(fields[field])
        else:
            logger.debug("Real number derivative, using cosine transforms")
            for f in range(num_fields):
                for i, deriv in enumerate(all_derivatives):
                    field_initial = fields[f]
                    for dim, (current_deriv, real_size) in enumerate(zip(deriv, real_sizes)):

                            field_initial = np.apply_along_axis(self.take_dct_deriv_1d, dim, field_initial,
                                                                current_deriv,
                                                                real_size)

                    output_preallocated[f, i, :] = field_initial.flatten()


        return output_preallocated

    @classmethod
    def construct_differentiation_matrices(cls, equation_options: HolographicLattices.Options.Options.EquationOptions,
                                           max_deriv: int = 2):

        # TODO: PseudoSpectral Diff Mat Testing
        grid_sizes = equation_options.grid_sizes
        spacings = equation_options.grid_spacings

        diff_matrices = HolographicLattices.FiniteDifferences.FiniteDifferenceMatrices. \
            diff_matrices_chebyshev_spectral(grid_sizes, methods=["chebspectral" for _ in spacings],
                                             max_deriv=max_deriv)

        return cls(_diff_matrices=diff_matrices, equation_options=equation_options)

    def get_matrix_representation(self):
        return self._diff_matrices


@dataclass
class FourierSpectralDifferentialOperator(BaseDifferentialOperator):
    # Note! differentiation matrices can be shared, as arrays are just pointers in python in the internal
    # C appliication. This is a good thing as it saves space.
    _diff_matrices: List[Union[np.ndarray, scipy.sparse.csr_matrix]]
    equation_options: HolographicLattices.Options.Options.EquationOptions

    @staticmethod
    def take_fft_deriv( field, derivatives, grid_sizes, real_sizes=None):

        import pyfftw
        import numpy as np

        assert len(derivatives) == len(grid_sizes), "Grid sizes not same length as derivatives"
        assert field.size == np.prod(grid_sizes), "Grid size does not match field dimensions"

        # prepare for a (maybe complex)-to-complex transform
        field_shaped = field.reshape(grid_sizes)
        input_fft = pyfftw.empty_aligned(grid_sizes, dtype='complex128')

        if real_sizes is None:
            real_sizes = [1.0 for _ in range(len(grid_sizes))]

        # If the data is not periodic, the real " assumed size " of the domain is one step __further__ in that direction
        # as now we assume for x0 .. x(N-1), but x(N-1) == 1, so that that f(xN) = f(x0)

        # load data into the transform
        input_fft[:] = field_shaped

        # multitthreading is way faster here if you have multiple cores at your disposal
        import multiprocessing
        cpus = multiprocessing.cpu_count()
        transformed_input = pyfftw.interfaces.numpy_fft.fftn(input_fft, threads=cpus)

        k_grids = np.meshgrid(
            *[np.fft.fftfreq(dim, d=size / (2 * np.pi * dim)) for size, dim in zip(real_sizes, grid_sizes)],
            indexing="ij")

        for grid, derivative, dimension in zip(k_grids, derivatives, grid_sizes):
            if derivative == 0:  # multiplication would have no effect
                continue
            else:
                transformed_input = np.multiply(transformed_input, np.power(1.j * grid, derivative))

        result_deriv = np.real(pyfftw.interfaces.numpy_fft.ifftn(transformed_input, threads=cpus))

        return result_deriv.reshape(field.shape)

    def __call__(self, fields: np.ndarray, output_preallocated: np.ndarray = None):
        """
        Differentation using FFT methods
        :param data: Data in 1d format
        :param sizes: Lenght of each interval, in usual definitions 2pi but can differ
        :param dimensions: number of gridpoints in each dimension
        :param periodic: Whether a dimension is periodic
        :param derivatives: Derivative to take. E.g. [0,1,2] corresponds to dy d2z
        :return: Derivative taken using pyfftw
        """

        equation_options = self.equation_options
        num_fields = fields.shape[0]

        grid_sizes = equation_options.grid_sizes
        real_sizes = equation_options.grid_domains
        dimensions = len(grid_sizes)

        all_derivatives = get_all_derivatives(equation_options.dims)

        result = np.array(
            [[self.take_fft_deriv(field, derivative, grid_sizes, real_sizes) for derivative in all_derivatives] for
             field in fields]
        )
        return result.reshape((num_fields, len(all_derivatives), np.prod(grid_sizes)))

    @classmethod
    def construct_differentiation_matrices(cls, equation_options: HolographicLattices.Options.Options.EquationOptions,
                                           max_deriv: int = 2):
        # TODO: Spectral Diff Mat Testing
        grid_sizes = equation_options.grid_sizes
        grid_domains = equation_options.grid_domains

        diff_order = equation_options.diff_order
        periodicities = equation_options.grid_periodic

        spacings = equation_options.grid_spacings

        diff_matrices = HolographicLattices.FiniteDifferences.FiniteDifferenceMatrices. \
            differentiation_matrices_spectral(grid_sizes,periodic_boundaries=periodicities,
                                                grid_domains=grid_domains, max_derivative=max_deriv)

        return cls(_diff_matrices=diff_matrices, equation_options=equation_options)

    def get_matrix_representation(self):
        return self._diff_matrices


@dataclass
class MixedFDDSpectralDifferentialOperator(BaseDifferentialOperator):
    # THis is it. The big one. This is the one that makes it all come together (or not, it might be the one
    # that is not the final version, and that the finite difference one mixed with this is going to be
    # the true final version of this but we'll see.

    _diff_matrices: List[Union[np.ndarray, scipy.sparse.csr_matrix]]
    equation_options: HolographicLattices.Options.Options.EquationOptions
    _finite_diff_matrices: List[Union[np.ndarray, scipy.sparse.csr_matrix]]
    _max_deriv: int

    def __call__(self, fields: np.ndarray, output_preallocated: np.ndarray = None):
        """
        Differentation using FFT methods
        :param data: Data in 1d format
        :param sizes: Lenght of each interval, in usual definitions 2pi but can differ
        :param dimensions: number of gridpoints in each dimension
        :param periodic: Whether a dimension is periodic
        :param derivatives: Derivative to take. E.g. [0,1,2] corresponds to dy d2z
        :return: Derivative taken using pyfftw
        """
        logger = logging.getLogger(__name__)
        equation_options = self.equation_options
        num_fields = fields.shape[0]

        grid_sizes = equation_options.grid_sizes
        real_sizes = equation_options.grid_domains
        periodicity = equation_options.grid_periodic
        methods = equation_options.eom_derivative_methods

        fields = fields.reshape((num_fields, *grid_sizes))

        all_derivatives = get_all_derivatives(equation_options.dims, self._max_deriv)

        if output_preallocated is None:
            output_preallocated = np.zeros((num_fields, len(all_derivatives), np.prod(grid_sizes)),
                                           dtype=equation_options.field_dtype)

        # if np.iscomplexobj(equation_options.field_dtype(0)):
        #     logger.info("Taking derivatives using complex numbers")
        #     for field in range(len(fields)):
        #         # no need to act with unity
        #         output_preallocated[field, 0, :] = fields[field]
        #
        #         for deriv in range(1, len(self._diff_matrices)):
        #             output_preallocated[field, deriv, :] = self._diff_matrices[deriv].dot(fields[field])
        # else:
        # logger.info("Real number derivative, using cosine transforms")

        for f in range(num_fields):
            for i, deriv in enumerate(all_derivatives):
                field_initial = fields[f]
                for dim, (method, current_deriv, real_size) in enumerate(zip(methods, deriv, real_sizes)):
                    if current_deriv == 0:
                        continue
                    elif method == "chebspectral":
                        field_initial = np.apply_along_axis(ChebyshevSpectralDifferentialOperator.take_fft_deriv, dim,
                                                            field_initial, current_deriv, real_size)
                    elif method == "fft":
                        funged_diff_op = [0 for i in deriv]
                        funged_diff_op[dim] = current_deriv
                        field_initial = FourierSpectralDifferentialOperator.take_fft_deriv(
                            field_initial, funged_diff_op,grid_sizes,real_sizes)
                    elif method == "fdd":
                        field_initial = np.apply_along_axis(
                            lambda fvalues: self._finite_diff_matrices[dim][current_deriv].dot(fvalues),
                        dim,field_initial)
                    else:
                        raise ValueError(f"Incorrect option for derivative method,"
                                         f" expected fdd, fft or chebspectral, got: {method}")


                output_preallocated[f, i, :] = field_initial.flatten()

        return output_preallocated

    @classmethod
    def construct_differentiation_matrices(cls,
                                           equation_options: HolographicLattices.Options.Options.EquationOptions,
                                           max_deriv: int = 2):

        # TODO: PseudoSpectral Diff Mat Testing
        grid_sizes = equation_options.grid_sizes
        spacings = equation_options.grid_spacings
        diff_order = equation_options.diff_order
        periodicities = equation_options.grid_periodic
        domains = equation_options.grid_domains
        methods = equation_options.eom_derivative_methods

        diff_matrices, finite_diff_matrices_separate = HolographicLattices.FiniteDifferences.FiniteDifferenceMatrices.\
            get_diff_matrices_mixed(grid_sizes, methods=methods,
                                            difference_orders=diff_order,
                                            periodic_boundaries=periodicities, spacings=spacings,
                                            domains=domains, max_derivative=max_deriv)

        return cls(_diff_matrices=diff_matrices, equation_options=equation_options,
                   _finite_diff_matrices=finite_diff_matrices_separate, _max_deriv=max_deriv)

    def get_matrix_representation(self):
        return self._diff_matrices


@dataclass
class MixedFDDSparseSpectralDifferentialOperator(BaseDifferentialOperator):
    # THis is it. The big one. This is the one that makes it all come together (or not, it might be the one
    # that is not the final version, and that the finite difference one mixed with this is going to be
    # the true final version of this but we'll see.

    _diff_matrices: List[Union[np.ndarray, scipy.sparse.csr_matrix]]
    equation_options: HolographicLattices.Options.Options.EquationOptions
    _finite_diff_matrices: List[Union[np.ndarray, scipy.sparse.csr_matrix]]
    _max_deriv: int

    def __call__(self, fields: np.ndarray, output_preallocated: np.ndarray = None):
        """
        Differentation using FFT methods
        :param data: Data in 1d format
        :param sizes: Lenght of each interval, in usual definitions 2pi but can differ
        :param dimensions: number of gridpoints in each dimension
        :param periodic: Whether a dimension is periodic
        :param derivatives: Derivative to take. E.g. [0,1,2] corresponds to dy d2z
        :return: Derivative taken using pyfftw
        """
        logger = logging.getLogger(__name__)
        equation_options = self.equation_options
        num_fields = fields.shape[0]

        grid_sizes = equation_options.grid_sizes
        real_sizes = equation_options.grid_domains
        periodicity = equation_options.grid_periodic
        methods = equation_options.eom_derivative_methods

        fields = fields.reshape((num_fields, *grid_sizes))

        all_derivatives = get_all_derivatives(equation_options.dims, max_derivative =self._max_deriv )

        if output_preallocated is None:
            output_preallocated = np.zeros((num_fields, len(all_derivatives), np.prod(grid_sizes)),
                                           dtype=equation_options.field_dtype)

        # if np.iscomplexobj(equation_options.field_dtype(0)):
        #     logger.info("Taking derivatives using complex numbers")
        #     for field in range(len(fields)):
        #         # no need to act with unity
        #         output_preallocated[field, 0, :] = fields[field]
        #
        #         for deriv in range(1, len(self._diff_matrices)):
        #             output_preallocated[field, deriv, :] = self._diff_matrices[deriv].dot(fields[field])
        # else:
        # logger.info("Real number derivative, using cosine transforms")

        for f in range(num_fields):
            for i, deriv in enumerate(all_derivatives):
                field_initial = fields[f]
                for dim, (method, current_deriv, real_size) in enumerate(zip(methods, deriv, real_sizes)):
                    if current_deriv == 0:
                        continue
                    elif method == "chebspectral":
                        field_initial = np.apply_along_axis(ChebyshevSpectralDifferentialOperator.take_fft_deriv, dim,
                                                            field_initial, current_deriv, real_size)
                    elif method == "fft":
                        funged_diff_op = [0 for i in deriv]
                        funged_diff_op[dim] = current_deriv
                        field_initial = FourierSpectralDifferentialOperator.take_fft_deriv(
                            field_initial, funged_diff_op, grid_sizes, real_sizes)
                    elif method == "fdd":
                        field_initial = np.apply_along_axis(
                            lambda fvalues: self._finite_diff_matrices[dim][current_deriv].dot(fvalues),
                            dim, field_initial)
                    else:
                        raise ValueError(f"Incorrect option for derivative method,"
                                         f" expected fdd, fft or chebspectral, got: {method}")

                output_preallocated[f, i, :] = field_initial.flatten()

        return output_preallocated

    @classmethod
    def construct_differentiation_matrices(cls,
                                           equation_options: HolographicLattices.Options.Options.EquationOptions,
                                           max_deriv: int = 2):
        logger = logging.getLogger(__name__)
        # TODO: PseudoSpectral Diff Mat Testing
        grid_sizes = equation_options.grid_sizes
        spacings = equation_options.grid_spacings
        diff_order = equation_options.diff_order
        periodicities = equation_options.grid_periodic
        domains = equation_options.grid_domains

        # Here we make the actual differentiation matrices sparse
        methods = ["fdd" for i in range(equation_options.dims)]
        logger.debug(f"Methods used in finite difference matrices: {methods}")
        diff_matrices, finite_diff_matrices_separate = HolographicLattices.FiniteDifferences.FiniteDifferenceMatrices. \
            get_diff_matrices_mixed(grid_sizes, methods=methods,
                                    difference_orders=diff_order,
                                    periodic_boundaries=periodicities, spacings=spacings,
                                    domains=domains, max_derivative=max_deriv)

        return cls(_diff_matrices=diff_matrices, equation_options=equation_options,
                   _finite_diff_matrices=finite_diff_matrices_separate, _max_deriv =max_deriv)

    def get_matrix_representation(self):
        return self._diff_matrices