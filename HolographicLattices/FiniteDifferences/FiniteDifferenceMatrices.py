import functools
import logging

import numpy as np
from scipy import sparse
import scipy
import scipy.linalg
from collections.abc import Iterable
import itertools

import HolographicLattices.Utilities.GridUtilities

"""
The code in this file has all possible (uniform, non-uniform, etc) differentiation matrices
prescribed in it. The algorithms have been tested, and appear to work quite generically.
"""

# TODO: WRITE TESTS FOR ALL ALGORITHMS


def make_iterable(length, *args):
    """
    If it's iterable, we are ok, because then it can be extended to the entire boundary.
    Otherwise, make a tuple and return that
    :param args: Vararg of arguments to check and extend. Returned in the same order.
    :return: List of all arguments, either the same or all iterable
    """
    return [obj if isinstance(obj, Iterable) else [obj for _ in range(length)] for obj in args ]

def centred_finite_difference_weights(points, deriv_centre, max_deriv=2):
    """
    Gets the finite difference weights for derivatives on the set of gridpoints, to a maximum
    derivative order of max_deriv, centred at point x0. This point x0 can even be a non-gridpoint
    Credit for this goes to some paper I found. TODO: find this paper for proper credit
    :param points: Locations of gridpoints to evaluate derivative over
    :param max_deriv: maximum order of derivative to take
    :param deriv_centre: CCentre of the derivative
    :return: 0th, 1st, .., max_deriv derivatives on the gridpoints
    """
    N = len(points) - 1
    c1 = 1.0
    ret = [[0 for _ in range(N + 1)] for _ in range(max_deriv + 1)]
    c4 = points[0] - deriv_centre
    ret[0][0] = 1.0

    for i in range(1, N + 1):
        mn = min(i, max_deriv)
        c2 = 1.0
        c5 = c4
        c4 = points[i] - deriv_centre

        for j in range(i):
            c3 = points[i] - points[j]
            c2 = c2 * c3

            if j + 1 == i:
                for k in np.arange(mn, 0, -1):
                    ret[k][i] = c1 * (k * ret[k - 1][i - 1] - c5 * ret[k][i - 1]) / c2
                ret[0][i] = -c1 * c5 * ret[0][i - 1] / c2

            for k in np.arange(mn, 0, -1):
                ret[k][j] = (c4 * ret[k][j] - k * ret[k - 1][j]) / c3
            ret[0][j] = c4 * ret[0][j] / c3
        c1 = c2
    return ret


def differentiation_matrix_1d_uniform(grid_dimension, difference_order=4, periodic_boundary=False, grid_domain=1.0,
                                      derivative_order=2):

    """
    Differentation matrices in one dimension for a uniform grid. Only the number of points and the domain matter,
    not the actual spacing of the points
    :param grid_dimension: Number of points on the grid
    :param difference_order: Number of neighbouring points for finite differences
    :param periodic_boundary: Whether the boundary is periddic or not
    :param grid_domain: Physical size of the domain
    :param derivative_order: Maximum order of derivative to create. Usually either 1 or 2.
    :return: List of derivative operators up to derivative_order (in order 0,1,2,...)
    """
    N = grid_dimension
    logger = logging.getLogger(__name__)

    if difference_order % 2 != 0:
        logger.critical("Error: Difference order is not even. This does not work")
        raise ValueError("Wrong order finite differences")

    if N <= difference_order:
        raise ValueError("Need to bigger grid than difference  order")

    if periodic_boundary:

        dx = grid_domain / N
        points = np.array([dx * i for i in range(N)])
        middle_grid = points[:difference_order + 1]
        middle_x0 = points[int(difference_order / 2)]

        periodic_weights = centred_finite_difference_weights(middle_grid, middle_x0, derivative_order)

        nearest_neighbours = int(difference_order / 2)

        diag_pos = [i for i in range(-N + 1, -N + 1 + nearest_neighbours)] + \
                   [i for i in range(-int(difference_order / 2), int(difference_order / 2) + 1)] + \
                   [i for i in range(N - nearest_neighbours, N)]

        diagonals = [periodic_weights[j][nearest_neighbours + 1:] + periodic_weights[j] + periodic_weights[j][:nearest_neighbours] for j
                     in range(derivative_order + 1)]

        return [sparse.diags(diagonals[j], diag_pos, shape=(N, N), format="csr") for j in range(derivative_order+1)]

    else:
        dx = grid_domain / (N - 1)
        points = np.array([dx * i for i in range(N)])
        return_diff_matrices = [sparse.csr_matrix((N, N), dtype=np.float64) for _ in range(derivative_order + 1)]

        # Special-case this on the
        for i in range(int(difference_order / 2)):
            localgrid = points[:difference_order + 1]
            x0 = points[i]
            weights = centred_finite_difference_weights(localgrid, x0, derivative_order)
            for j in range(derivative_order + 1):
                return_diff_matrices[j][i, :difference_order + 1] = weights[j]

        for i in range(N - int(difference_order / 2), N):
            localgrid = points[-(difference_order + 1):]
            x0 = points[i]
            weights = centred_finite_difference_weights(localgrid, x0, derivative_order)
            for j in range(derivative_order + 1):
                return_diff_matrices[j][i, -(difference_order + 1):] = weights[j]

        middle_grid = points[:difference_order + 1]
        middle_x0 = points[int(difference_order / 2)]

        finite_diff_weights_bulk = centred_finite_difference_weights(middle_grid, middle_x0, derivative_order)

        for i in range(int(difference_order / 2), N - int(difference_order / 2)):
            for j in range(derivative_order + 1):
                return_diff_matrices[j][i, i-int(difference_order/2):i + 1 + int(difference_order/2)] = finite_diff_weights_bulk[j]
        for mat in return_diff_matrices:
            mat.eliminate_zeros()
        return return_diff_matrices


def differentiation_matrix_1d_fdd_chebyshev(grid_size, derivative_order=2, difference_order=4, domain=1.0):
    """
    Returns the differentiation matrix for a chebyshev grid. The grid is not period
    :param grid_size: Number of points to take the derivative on
    :param derivative_order: Highest order of derivative to take
    :param difference_order: Difference order. This needs to be a multiple of 2
    :return: 1d differentiation matrices on a chebyshev grid, for the order [-1,..,1]
    """
    N = grid_size
    if grid_size < 1 + difference_order:
        raise ValueError("The stencil cannot be larger than the grid size")
    if derivative_order < 0:
        raise ValueError("Cannot take derivative less than order 0")
    if difference_order % 2 != 0 or difference_order <= 0:
        raise ValueError("Difference order has to be an even positive integer")

    points = np.cos(np.pi * np.arange(0, N) / (N - 1))
    # Reverses direction since the cos(0) = 1, cos(pi) = -1.
    points = (1+points[::-1])/2
    points = points*domain
    return differentiation_matrix_1d_arbitrary(points, derivative_order, difference_order, periodic_boundary=False)


def differentiation_matrices_1d_spectral_uniform(num_points, derivative_order, domain=1.0):
    """
    Full dense spectral (uniform periodic) differentation matrix for one dimension, up to order 2.
    :param points: Grid to make it on
    :param derivative_order: Order of derivative to generate
    :param domain: Physical size of domain
    :return: Dense np.ndarray representing full differentiation matrix
    """
    logger=logging.getLogger(__name__)
    if num_points % 2 != 0:
        raise ValueError("Need to have even number of grid points for this spectral operator (not sure why!)")

    return_mats = []
    if derivative_order >= 0:
        return_mats.append( np.identity(num_points))
    if derivative_order >= 1:
        dx = 2*np.pi/num_points
        rangeN = np.arange(0,num_points,1)
        column = np.zeros_like(rangeN, dtype=np.float64)
        row = np.zeros_like(column)
        column[1:] = (2*np.pi/domain)*(0.5*np.power((-1),(rangeN[1:])))/(np.tan(rangeN[1:]*dx/2.0))
        row[1:] = column[-1:0:-1]
        DMat = scipy.linalg.toeplitz(column,row)
        return_mats.append(DMat)

    if derivative_order >= 2:
        dx = 2*np.pi/num_points
        rangeN = np.arange(0,num_points,1)
        column = np.zeros_like(rangeN, dtype=np.float64)
        row = np.zeros_like(column)
        column[0] = - np.pi**2/(3*dx**2) - (1/6)
        column[1:] = (0.5*np.power((-1),(rangeN[1:]+1)))/(np.sin(rangeN[1:]*dx/2.0)**2)
        row[0] = column[0]
        row[1:] = column[-1:0:-1]
        DMat = scipy.linalg.toeplitz(column,row)*(2*np.pi/domain)**2
        return_mats.append(DMat)
    if derivative_order > 2:
        logger.debug("Diff order > 2 not really supported, be careful with this, as it is currently using matrix powers")
        return_mats.append(np.linalg.matrix_power(return_mats[1], derivative_order%2).dot(np.linalg.matrix_power(return_mats[2], derivative_order//2)))
    return return_mats


def chebyshev_trefethen_mat_1d(num_points, deriv_order, grid_real_size=1.0):
    """
    See J. A . C . Weideman, S . C . Reddy 1998.. This is a numerically stable version that
    does well under rounding effects.
    @param num_points: number of points in the grid
    @param deriv_order: maximum order of derivative to evaluate to. This is stable to high order (npoints - 1)
    """
    assert num_points > 3, f"Need more than 3 points. Received: {num_points= }"
    assert deriv_order >= 0, f"Need positive derivative order. Received: {deriv_order= }"

    N = num_points
    import math
    n1 = math.floor(N / 2)
    n2 = math.ceil(N / 2)

    k = np.arange(0, N)[:, np.newaxis]  # Modes labeled along the circle

    theta = (k * np.pi / (N - 1))  # angles of those modes

    T = np.repeat(theta / 2, N, axis=1)  # difference matrix of those

    DX = 2 * np.sin(T.transpose() + T) * np.sin(T.transpose() - T)  # distances between each pair of points

    DX = np.vstack((DX[0:n1, :], -np.fliplr(np.flipud(DX[0:n2, :]))))  # use trick to increase FP accuracy
    # this is because sin(x) for small x is much more accurate than sin(pi - x) while they give the same value

    # Set this to one so the inversion for Z is correct
    np.fill_diagonal(DX, 1)

    # 1.0 is important here: otherwise integer math will take over silently
    import scipy
    C = scipy.linalg.toeplitz(np.power((-1.0), k.flatten()))

    C[0, :] *= 2.0
    C[-1, :] *= 2.0

    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    Z = 1.0 / DX

    # Recover zero distance after inversion of coordinate distances
    np.fill_diagonal(Z, 0)

    D = np.identity(N)

    # Always have at least one derivative, namely the identity
    return_arrays = [np.copy(D)]

    for diff_order in range(1, deriv_order + 1):
        # This isi all explainedin  J.. A . C . Weideman, S . C . Reddy 1998.
        D = diff_order * Z * (C * np.repeat(np.diag(D)[:, np.newaxis], N, axis=1) - D)
        np.fill_diagonal(D, -np.sum(D, axis=1))
        return_arrays.append(np.copy(D))

    # since they are defined on grid 1....-1 typically, we fix this by doing minus them and then multiplying them
    return [(-(2.0 / grid_real_size)) ** i * mat for i, mat in enumerate(return_arrays)]


def differentiation_matrix_1d_arbitrary(points, derivative_order: int = 2, difference_order: int = 4, periodic_boundary: bool = False):
    """
    Differentation matrices for an arbitrary grid in one dimension. Periodic boundary conditions
    :param points: Points to make the derivative grid on
    :param derivative_order: Hghest order of derivative to take. >=0
    :param difference_order: Number of nearest neighbour derivatives to take. Needs to be even positive integer
    :param periodic_boundary: If the problem is defined on a periodic boundary, the last element is taken to be the
    first element of the repeated array. Therefore if you give x0,....,xN, where xN = x0, it returns the NxN
    differentiation matrix over that grid with the endpoints identified. You _NEED_ the extra point because otherwise
    the identification is impossible to make
    :return: List of CSR sparse differentiation matrices, in the order [I, D^1, D^2, .., D^max]
    """
    N = len(points)
    diff_ord_2 = int(difference_order / 2)
    ret = [np.zeros((N, N)) for _ in range(derivative_order + 1)]
    if not periodic_boundary:
        # If we don't have a periodic boundary, we can just assign all these variables in the usual order:
        # First the endpoints, then the interior
        for i in range(diff_ord_2):
            localgrid = points[:difference_order + 1]
            x0 = points[i]
            weights = centred_finite_difference_weights(localgrid, x0, derivative_order)
            for j in range(derivative_order + 1):
                ret[j][i, :difference_order + 1] = weights[j]
        for i in range(N - int(difference_order / 2), N):
            localgrid = points[-(difference_order + 1):]
            x0 = points[i]
            weights = centred_finite_difference_weights(localgrid, x0, derivative_order)
            for j in range(derivative_order + 1):
                ret[j][i, -(difference_order + 1):] = weights[j]


        for i in range(diff_ord_2, N - diff_ord_2):
            localgrid = points[i - diff_ord_2: i + diff_ord_2 + 1]
            x0 = points[i]
            weights = centred_finite_difference_weights(localgrid, x0, derivative_order)
            for j in range(derivative_order + 1):
                ret[j][i, i - diff_ord_2:i + 1 + diff_ord_2] = weights[j]
    else:
        # If we are in a periodic situation, we need to identify the points x0 = xN, and remove
        # xN from the list of points to include. N+1 poitns need to be given for an N point array because
        # otherwise the spacing until the repeated point (x(N-1) -> x0) can not be made
        endpoints = points[-1]
        points = points[:-1]

        # Do the left side first
        for i in range(diff_ord_2):
            localgrid = list(points[:i + diff_ord_2 + 1])
            # We need to wrap the points by placing the far-end points by their continuation on the iother side of x0
            localgrid_wrap = points[-(diff_ord_2 - i):]
            localgrid_wrap = [loc_point - (endpoints - points[0]) for loc_point in localgrid_wrap]
            localgrid = localgrid_wrap + localgrid
            x0 = points[i]
            weights = centred_finite_difference_weights(localgrid, x0, derivative_order)
            for j in range(derivative_order + 1):
                ret[j][i, :i + diff_ord_2 + 1] = weights[j][diff_ord_2 - i:]
                ret[j][i, -(diff_ord_2 - i):] = weights[j][:diff_ord_2 - i]
        for i in range(diff_ord_2, N - diff_ord_2):
            # The interior is just straightforward
            localgrid = points[i - diff_ord_2: i + diff_ord_2 + 1]
            x0 = points[i]
            weights = centred_finite_difference_weights(localgrid, x0, derivative_order)
            for j in range(derivative_order + 1):
                ret[j][i, i - diff_ord_2:i + 1 + diff_ord_2] = weights[j]
        for i in range(N - diff_ord_2, N):
            # Again we need to wrap, but now by adding the size of the periodic domain on the right hand side.
            localgrid = list(points[i - diff_ord_2:])
            localgrid_wrap = points[:diff_ord_2 + i + 1 - N]
            localgrid_wrap = [loc_point + endpoints - points[0] for loc_point in localgrid_wrap]
            localgrid = localgrid + localgrid_wrap
            x0 = points[i]
            weights = centred_finite_difference_weights(localgrid, x0, derivative_order)
            for j in range(derivative_order + 1):
                ret[j][i, :(diff_ord_2 + i + 1) % N] = weights[j][N - i - diff_ord_2 - 1:]
                ret[j][i, i - diff_ord_2:] = weights[j][:N - i - diff_ord_2 - 1]

    return [sparse.csr_matrix(mat) for mat in ret]


def differentiation_matrices_uniform(grid_sizes, difference_orders=4, periodic_boundaries=(False,), grid_domains=(1.0,),
                                     max_derivatives=2):
    """
    Get all the differentiation matrices in the usual order for uniform grids
    :param grid_sizes: Size of each grid
    :param difference_orders: Difference order
    :param periodic_boundaries: Either a common value for all boundaries, or specified per dimension
    :param grid_domains: Extents of each domain
    :param max_derivatives: Highest derivative needed. This is almost always 2 in our problems.
    :return:
    """
    dim = len(grid_sizes)

    difference_orders, periodic_boundaries = make_iterable(dim, difference_orders, periodic_boundaries)

    differentiation_matrices = [
        differentiation_matrix_1d_uniform(size, difference_order=diff_ord, periodic_boundary=per_bound,
                                          grid_domain=domain, derivative_order=max_derivatives)
        for size, diff_ord, per_bound, domain in
        zip(grid_sizes, difference_orders, periodic_boundaries, grid_domains)]

    derivs = HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dim, max_derivatives)
    ret = []

    for i, mat in enumerate(differentiation_matrices):
        differentiation_matrices[i] = sparse.csr_matrix(mat)
        differentiation_matrices[i].eliminate_zeros()

    for deriv in derivs:
        matrices = [differentiation_matrices[direction][order] for direction, order in enumerate(deriv)]
        ret.append(functools.reduce(lambda x, y: sparse.kron(x, y, format="csr"), matrices))

    return ret

def differentiation_matrices_spectral(grid_sizes, periodic_boundaries=(True,), grid_domains=(1.0,),
                                     max_derivative=2):
    """
    Get all the differentiation matrices in the usual order for uniform grids
    :param grid_sizes: Size of each grid
    :param difference_orders: Difference order
    :param periodic_boundaries: Either a common value for all boundaries, or specified per dimension
    :param grid_domains: Extents of each domain
    :param max_derivatives: Highest derivative needed. This is almost always 2 in our problems.
    :return:
    """
    dim = len(grid_sizes)
    if not all(periodic_boundaries):
        raise ValueError(f"Periodic boundaries found were {periodic_boundaries}, need all to be periodic")

    differentiation_matrices = [
        differentiation_matrices_1d_spectral_uniform(size, derivative_order = 2, domain=domain)
        for size, domain in
        zip(grid_sizes, grid_domains)]

    derivs = HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dim, max_derivative)
    ret = []

    for deriv in derivs:
        matrices = [differentiation_matrices[direction][order] for direction, order in enumerate(deriv)]
        ret.append(functools.reduce(lambda x, y: sparse.kron(x, y, format="csr"), matrices))

    return ret

def diff_matrices_chebyshev_spectral(grid_sizes, methods, max_deriv=2):
    dimension = len(grid_sizes)
    diff_mats = []

    for dim, method in enumerate(methods):
        if method == "chebspectral":
            dimderivs = chebyshev_trefethen_mat_1d(grid_sizes[dim], max_deriv)
            diff_mats.append(dimderivs)
        else:
            raise ValueError(f"Wrong grid type: {method}. Expected: \"fdd\" or \"cheb\"")

    derivs = HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dimension, max_deriv)

    ret = []

    for deriv in derivs:
        matrices = [diff_mats[direction][order] for direction, order in enumerate(deriv)]
        ret.append(functools.reduce(lambda x, y: np.kron(x, y), matrices))

    return ret



def get_diff_matrices_mixed(grid_sizes, methods, difference_orders,periodic_boundaries, spacings,
                                            domains, max_derivative):

    logger = logging.getLogger(__name__)
    dim = len(grid_sizes)
    finite_diff_direction_mats = [list() for _ in range(len(grid_sizes))]
    diff_mats = []
    diff_orders_it, = make_iterable(dim, difference_orders)
    if all((method == "fdd" for method in methods)):
        logger.info("Using finite difference approximations for the jacobian")
        kron_method = lambda a,b: scipy.sparse.kron(a,b, format="csr")
    else:
        logger.info("Using fully dense matrices for the jacobian")
        kron_method = np.kron
    for i,(method, grid_size, periodicity, diff_order, spacing, domain) in \
            enumerate(zip(methods, grid_sizes, periodic_boundaries, diff_orders_it, spacings, domains)):
        if method == "chebspectral":
            matrices = chebyshev_trefethen_mat_1d(grid_size, max_derivative, domain)
            diff_mats.append(matrices)
        elif method == "fft":
            matrices = differentiation_matrices_1d_spectral_uniform(grid_size, max_derivative, domain)
            diff_mats.append(matrices)
        elif method == "fdd":
            if spacing == "cheb":
                matrices = differentiation_matrix_1d_fdd_chebyshev(grid_size,
                                                                   derivative_order=max_derivative,
                                                                   difference_order=diff_order,
                                                                   domain=domain)
                diff_mats.append(matrices)
                finite_diff_direction_mats[i] = matrices
            elif spacing == "unif":
                matrices = differentiation_matrix_1d_uniform(grid_size, diff_order, periodicity, domain, max_derivative)
                diff_mats.append(matrices)
                finite_diff_direction_mats[i] = matrices
    derivative_ordering = HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dim, max_derivative)

    ret = []

    for deriv in derivative_ordering:
        matrices = [diff_mats[direction][order] for direction, order in enumerate(deriv)]
        ret.append(functools.reduce(kron_method, matrices))

    return ret, finite_diff_direction_mats

def differentiation_matrices_structured(grid_sizes, difference_orders=4, periodic_boundaries=False, spacings="unif",
                                        domains=1.0, max_derivative=2):
    """
    Most general way of getting sparse (NON-SPECTRAL) matrices. Spectral matrices could be handled here, but
    but they are large and cumbersome. Returns all differentiation matrices as a list of sparse CSR matrices
    :param grid_sizes: list of 1d grids to evaluate over
    :param difference_orders: finite-difference orders to use for each direction (can be different!)
    :param periodic_boundaries: boundary conditions on each side
    :param spacings: Uniform or nonuniform spacings of the grid (options: cheb, unif so far)
    :param domains: Domain
    :param max_derivative:
    :return:
    """

    logger = logging.getLogger(__name__)
    dim = len(grid_sizes)

    difference_orders, periodic_boundaries,spacings,domains = make_iterable(dim, difference_orders, periodic_boundaries,spacings, domains)

    if not all((i == "cheb" or i == "unif" for i in spacings)):
        logger.error(f"This method expects all uniform or chebyshev spacings for the grid. Received: {spacings}")
        raise ValueError(f"This method expects all uniform or chebyshev spacings for the grid. Received: {spacings}")

    diff_mats = []


    for grid, periodicity, diff_order, spacing,domain in \
        itertools.zip_longest(grid_sizes, periodic_boundaries, difference_orders, spacings, domains):

        if spacing == "unif":

            diff_mats.append(differentiation_matrix_1d_uniform(grid, diff_order, periodicity, domain, max_derivative))
        elif spacing == "cheb":
            raise NotImplementedError("This uses old implementation! Remove")
            #diff_mats.append(differentiation_matrix_1d_chebyshev(grid, max_derivative, diff_order, domain))
        else:
            raise ValueError("Can only make structured grid for unif or cheb spacing")
            # This is kind of the special case
            # diff_mats.append(differentiation_matrix_1d_arbitrary(grid, max_derivative, diff_order, periodicity))

    diff_mats_sparse = [[sparse.csr_matrix(row) for row in mat] for mat in diff_mats]
    for row in diff_mats_sparse:
        for mat in row:
            mat.eliminate_zeros()

    derivative_ordering = HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dim, max_derivative)

    ret = []

    for deriv in derivative_ordering:
        matrices = [diff_mats[direction][order] for direction, order in enumerate(deriv)]
        ret.append(functools.reduce(lambda x, y: sparse.kron(x, y, format="csr"), matrices))
    return ret


def differentiation_matrices_unstructured(grids, difference_orders=4, periodic_boundaries=False, max_deriv=2):
    logger = logging.getLogger(__name__)
    dim = len(grids)
    difference_orders, periodic_boundaries = make_iterable(dim, difference_orders, periodic_boundaries)
    diff_mats = [
        sparse.csr_matrix(differentiation_matrix_1d_arbitrary(grid, derivative_order=max_deriv, periodic_boundary=periodic, difference_order=diff_ord) for
        grid, periodic, diff_ord in
        itertools.zip_longest(grids, periodic_boundaries, difference_orders))]

    for mat in diff_mats:
        mat.eliminate_zeros()

    derivs = HolographicLattices.Utilities.GridUtilities.get_all_derivatives(dim, max_deriv)

    ret = []

    for deriv in derivs:
        matrices = [diff_mats[direction][order] for direction, order in enumerate(deriv)]
        ret.append(functools.reduce(lambda x, y: sparse.kron(x, y, format="csr"), matrices))
    return ret
