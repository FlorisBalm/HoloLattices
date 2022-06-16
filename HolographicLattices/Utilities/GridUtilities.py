from functools import reduce
from itertools import permutations
from dataclasses import dataclass, field
import numpy as np
import logging
from typing import List
from scipy import sparse
import HolographicLattices.Options.Options


def integer_partition(n):
    """
    Generator for integer partitions, used to define all derivatives that we might want to take
    :param n: Number to partition
    :return: Generator that contains all partitions, sorted.
    """
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        m = k + 1
        while x <= y:
            a[k] = x
            a[m] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]


def default_derivatives_sort(a, b):
    """
    Sorting function that sorts the derivative in the predefined order
    :param a: First param to  compare
    :param b: Second to compare
    :return: +1 if a > b, -1 if a < b, 0 if a == b
    """
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a < sum_b:
        return -1
    elif sum_a > sum_b:
        return +1
    else:
        sumsq_a = sum([x ** 2 for x in a])
        sumsq_b = sum([x ** 2 for x in b])
        if sumsq_a > sumsq_b:
            return -1
        elif sumsq_b > sumsq_a:
            return +1
        else:
            if a < b:
                return +1
            elif b < a:
                return -1
            else:
                return 0


def sort_by_key(comp):
    """
    Helper function to turn comp into a key for python sorting, python3 no longer supports cmp sorting.
    :param comp: Function with 2 arguments to use in comparison
    :return: Key-class to compare with
    """
    class K:
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return comp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return comp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return comp(self.obj, other.obj) == 0

        def __le__(self, other):
            return comp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return comp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return comp(self.obj, other.obj) != 0
    return K


def get_boundaries_and_bulk(grid_sizes, periodicity):
    """
    Returns matrices that indicate where the boundaries of an (dim0 x ...) problem are located.
    :param grid_sizes: Sizes for each of the dimensions
    :param periodicity: Perioicity of each of the dimensions
    :return: 1st param: for each of the dimensions, in the order dim0, dim1, dim2 a list of 2 values:
            First is the "x=0" or "front" side of each, then is the "x=1" or "back"
    """
    identities = [sparse.identity(size, format="csr") for size in grid_sizes]
    dim = len(grid_sizes)
    boundaries = [[[], []] for _ in range(dim)]
    # For each of the dimensions,  we only need the start or the end. In the other dimensions, it stays the identity
    for i, periodic in enumerate(periodicity):
        if periodic:
            boundaries[i][0] = None
            boundaries[i][1] = None

        else:
            # temp is a copy of all identity matrices
            # where we one by one take only the first or last element of the matrix.
            temp = list(identities)
            temp_end = list(temp)
            # Here we set one of the matrices to have a non-zero index in only the first or last
            # diagonal element, then when taking the kronecker product we get only the boundary where
            # that index is at the end (e.g. x = 0 gives a yz plane)
            temp[i] = sparse.csr_matrix((grid_sizes[i], grid_sizes[i]))
            temp[i][0, 0] = 1
            temp_end[i] = sparse.csr_matrix((grid_sizes[i], grid_sizes[i]))
            temp_end[i][-1, -1] = 1

            # Take the product of all matrices together
            boundaries[i][0] = reduce(lambda a, b: sparse.kron(a, b, format="csr"), temp)
            boundaries[i][1] = reduce(lambda a, b: sparse.kron(a, b, format="csr"), temp_end)
    # The bulk is the product of identities with each of the endpoints removed
    for i, periodic in enumerate(periodicity):
        if not periodic:
            identities[i][0, 0] = 0
            identities[i][-1, -1] = 0
        else:
            pass

    bulk = reduce(lambda a, b: sparse.kron(a, b, format="csr"), identities)
    return boundaries, bulk


def get_boundaries_and_bulk_indices(grid_sizes, periodicity):
    """
    This function is the same as get_boundaries_and_bulk above, but now it returns the indices that the boundary
    and bulk would have in the vector of all indices
    :param grid_sizes: Sizes for each of the dimensions
    :return: 1st param: for each of the dimensions, in the order dim0, dim1, dim2 a list of 2 values:
                First is the "x=0" or "front" side of each, then is the "x=1" or "back"
                2nd param is the same but for the bulk
        """
    bounds, bulk = get_boundaries_and_bulk(grid_sizes, periodicity)
    retbound = [[[] for _ in range(2)] for _ in range(len(grid_sizes))]
    for i, dim in enumerate(bounds):
        for j in range(2):
            b = bounds[i][j]
            if b is not None:
                retbound[i][j] = b.nonzero()[0]
            else:
                retbound[i][j] = None
    retbulk = bulk.nonzero()[0]
    return retbound, retbulk


@dataclass
class GridInformation:
    options: HolographicLattices.Options.Options.EquationOptions

    grids: List[np.ndarray] = field(init=False, default_factory=lambda: None)

    def __post_init__(self):
        self.grids = self.get_1d_grids()

    def get_1d_grids(self, force_reevaluate_grids=False):
        """
        Computes the grids as defined by the parameters of this setup.
        :return: List of grids in each direction, as indicated by the order in the setup.
        """
        logger = logging.getLogger(__name__)
        if self.grids is not None and not force_reevaluate_grids:
            return self.grids

        def cheb_spacing(a, b, N):
            points = np.cos(np.pi * np.linspace(0, 1, N))
            points = points + 1
            points = points * (b - a) / 2
            points = points + a
            points = points[::-1]
            return points

        # TODO : Allow for dynamic selection of grid type. e.g. FDD might still want a non-uniform grid for example.

        grids = []


        for grid_size, periodicity, spacing, domain in zip(self.options.grid_sizes, self.options.grid_periodic,
                                                           self.options.grid_spacings, self.options.grid_domains):
            # For finite differences, can have any spacing
            if spacing == "unif":
                if periodicity:
                    points = np.linspace(0.0, domain, grid_size + 1)[:-1]
                else:
                    points = np.linspace(0.0, domain, grid_size)
            elif spacing == "cheb":
                if periodicity:
                    points = cheb_spacing(0.0, domain, grid_size + 1)[:-1]
                else:
                    points = cheb_spacing(0.0, domain, grid_size)
            else:
                # It's quite likely an error, however it still possible to recover while making sense.
                logger.error(f"Unknown grid spacing of type: {spacing}!")
                points = np.linspace(0.0, domain, grid_size)

            grids.append(points)

        return grids

def get_all_derivatives(dimensions:int = 1, max_derivative: int = 2):
    """
        Returns all possible derivatives in the specified number of dimensions up to a maximum derivative
        :param max_derivative: highest order derivative to take
        :return: A list of all derivatives. E.g. for 2 derivatives in 2d, it returns
                    [[0,0],[1,0],[0,1],[2,0],[0,2],[1,1]]
        """

    derivs = []

    # all possible partitions of all possible orders of derivatives, up to deriv_order
    for deriv in range(max_derivative + 1):
        for partition in integer_partition(deriv):
            derivs.append(partition)

    # filter out the ones that are across too many dimensions
    derivs = [der for der in derivs if len(der) <= dimensions]
    derivs = list(map(lambda x: [0] * (dimensions - len(x)) + x, derivs))
    ret = []

    # Here we take all possible unique permutations, and sort those in the desired order
    for deriv in derivs:
        for perm in set(permutations(deriv)):
            ret.append(list(map(int, perm)))
    return sorted(ret, key=sort_by_key(default_derivatives_sort))
