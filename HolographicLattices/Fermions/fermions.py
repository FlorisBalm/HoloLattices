import os
import sys

HOLOHOME = os.getenv('HOLOLATTICES_HOME', "")
sys.path.append(HOLOHOME)
import HolographicLattices.control_observables
import HolographicLattices.derivatives
import HolographicLattices.Utilities.GridUtilities
import HolographicLattices.Options.Options
import HolographicLattices.import_equations
import numpy as np
import h5py
import HolographicLattices.nonlinear_solver
import time
import itertools
import scipy
import scipy.sparse
import scipy.sparse.linalg
import logging


def linearFermionOperators(coefficients, setup: 'FermionSetup'):
    logger = logging.getLogger("fermions")
    diffMats = setup.finite_diff_matrices
    # Empty matrices to construct the differentiation matrices in
    # We can see the diff matrix as a huge num_eqs_of_motion x nField block of gridvolume x gridvolume blocks, which
    # can be inverted to find the derivative.

    boundaryIndices, bulkIndices = setup.boundaryIndices, setup.bulkIndices
    boundaryVariables, bulkVariables = setup.getBoundaryBulkInput()
    # Go from Field - Deriv - EOM to field,eom - deriv
    # For most grid sizes, it is the internal evaluation that is by far the slowest, and parallelizing
    # over the data
    numDerivs = len(HolographicLattices.Utilities.GridUtilities.all_derivatives(setup.dims + 1, setup.max_deriv))
    bigDiffMatrix = [[[scipy.sparse.csr_matrix((setup.gridVolume, setup.gridVolume)) for _ in range(setup.num_fields)]
                      for _ in range(setup.num_eqs_of_motion)] for _ in range(numDerivs)]
    for b, block in enumerate(coefficients.internal):
        for i, field in enumerate(block):
            for j, deriv in enumerate(field):
                for k, eom in enumerate(deriv):
                    term_eval = eom(**bulkVariables)
                    import numbers
                    if isinstance(term_eval, numbers.Number):
                        term_eval = np.zeros(len(bulkIndices), dtype=np.complex128) + complex(term_eval)

                    mat = scipy.sparse.coo_matrix((setup.gridVolume, setup.gridVolume))
                    mat.row = bulkIndices
                    mat.col = bulkIndices
                    mat.data = term_eval
                    mat.eliminate_zeros()
                    bigDiffMatrix[b][i][k] += mat.dot(diffMats[j])

    for axis, periodic in enumerate(setup.grid_periodic):
        if not periodic:
            for end in range(2):
                idx = boundaryIndices[axis][end]

                # Evaluate these equations only where they apply. This should reduce
                # computation time somewhat
                variables = boundaryVariables[axis][end]
                coeffs = coefficients.boundary[axis][end]

                # The coefficients are stored as field : deriv : eom for each.
                for b, block in enumerate(coeffs):
                    for i, field in enumerate(block):
                        for j, deriv in enumerate(field):
                            for k, eom in enumerate(deriv):

                                term_eval = eom(**variables)
                                import numbers
                                if isinstance(term_eval, numbers.Number):
                                    term_eval = np.zeros(len(idx), dtype=np.complex128) + complex(term_eval)

                                mat = scipy.sparse.coo_matrix((setup.gridVolume, setup.gridVolume))
                                mat.row = idx
                                mat.col = idx
                                mat.data = term_eval
                                mat.eliminate_zeros()
                                bigDiffMatrix[b][i][k] += mat.dot(diffMats[j])

    # Stack the matrices into a nFxnF form so that it can be inverted
    fermionOperators = [scipy.sparse.csr_matrix((setup.gridVolume, setup.gridVolume)) for _ in range(numDerivs)]
    for b, bigMatrix in enumerate(bigDiffMatrix):
        hStacked = [scipy.sparse.hstack(row) for row in bigMatrix]
        fermionOperators[b] = scipy.sparse.vstack(hStacked, format="csr")
        fermionOperators[b].eliminate_zeros()

    return fermionOperators


# This should just work flawlessly

# @logcall()
def fermionRHS(RHS: 'HolographicLattices.import_equations.Equations', setup: 'FermionSetup'):
    """
    Evaluate the equations of motion for a given set of fields. This does not include the
    right-hand side
    :param fields: Fields to evaluate
    :param eoms_internal: Equations of motion for the points that are not on a boundary
    :param eoms_boundary: boundary equations of motion
    :param setup: Setup of the problem, defined in holo_setup.py
    :return: equations of motion evaluated at each point for each field.
    """
    logger = logging.getLogger("fermions")
    # Empty fields to fill up with equations of motion
    numDerivs = len(HolographicLattices.Utilities.GridUtilities.all_derivatives(setup.dims + 1, setup.max_deriv))
    eomsEvaluated = np.array(
        [[np.zeros(setup.gridVolume, dtype=np.complex128) for _ in range(setup.num_fields)] for _ in range(numDerivs)])

    # These are the indices of the boundaries for 1-d flattened coordinates
    # These are used for indexing into the arrays of points
    # The boundary variable is a list of pairs that for the "0" and "1" end of the boundary,
    # you can interpret that for "left" and "right" or whatever seems appropriate. For periodic
    # boundaries, the indices are None
    boundaryIndices, bulkIndices = setup.boundaryIndices, setup.bulkIndices
    boundaryVariables, bulkVariables = setup.getBoundaryBulkInput()

    # Evaluate and assign the equations of motion to the internal points
    for b, block in enumerate(RHS.internal):
        for i, eom in enumerate(block):
            term_eval = eom(**bulkVariables)
            import numbers
            if isinstance(term_eval, numbers.Number):
                term_eval = np.zeros(len(bulkIndices), dtype=np.complex128) + complex(term_eval)
            eomsEvaluated[b][i][bulkIndices] = term_eval

    # Evaluate and assign the equations of motion to the non-periodic boundary points
    for axis, periodic in enumerate(setup.grid_periodic):
        if not periodic:
            # Each non-periodic direction has two boundaries
            for end in range(2):
                idx = boundaryIndices[axis][end]
                variables = boundaryVariables[axis][end]
                eoms = RHS.boundary[axis][end]
                # Evaluate these equations only where they apply. This should reduce
                # computation time somewhat
                for b, block in enumerate(eoms):
                    for k, eom in enumerate(block):

                        term_eval = eom(**variables)
                        import numbers
                        if isinstance(term_eval, numbers.Number):
                            term_eval = np.zeros(len(idx), dtype=np.complex128) + complex(term_eval)
                        eomsEvaluated[b][k][idx] = term_eval
    return eomsEvaluated


# @logcall()
def readFermionRHS(filename, setup: 'FermionSetup'):
    """
    Read the nonlinear-eoms file.
    :param filename:
    :param setup_data:
    :return:
    """
    try:
        with open(filename, "r") as infile:
            lines = []

            numBlocks = len(HolographicLattices.Utilities.GridUtilities.all_derivatives(setup.dims + 1, setup.max_deriv))
            for line in infile.readlines():
                params = ", ".join((setup.backgroundName, *setup.grid_names))
                l = " ".join(("lambda", params, ":", line))
                fun = eval(l, {**setup.constants, **setup.functions})
                lines.append(fun)

            lines = np.array(lines)
            lines = lines.reshape(numBlocks, setup.num_fields)
            return lines
    except IOError as e:
        logger = logging.getLogger("fermions")
        logger.error(f"Error reading file: {filename}. Error:{e}")
        raise e


# @logcall()
def readFermionCoefficients(filename, setup: 'FermionSetup'):
    """
    Rewritten import_expressions that is more readable

    :param filename: Name of file to read
    :return: Array of N_eom x num_fields x N_derivs elements with the equations in them
    """
    logger = logging.getLogger("fermions")
    try:
        with open(filename, "r") as infile:

            # Strip any whitespace for parsing
            lines = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

            nEOM = setup.num_eqs_of_motion
            nFields = setup.num_fields
            maxDeriv = setup.max_deriv
            dims = setup.dims

            # All possible different derivatives
            numBlocks = len(HolographicLattices.Utilities.GridUtilities.all_derivatives(dims + 1, maxDeriv))
            numDerivs = len(HolographicLattices.Utilities.GridUtilities.all_derivatives(dims, maxDeriv))

            terms_in_eom = [[[[lambda: 0 for _ in range(nFields)] for _ in range(numDerivs)] for _ in range(nEOM)] for _
                            in range(numBlocks)]
            # this is for iterating over all eoms, fields and derivatives
            indices = itertools.product(range(numBlocks), range(nEOM), range(numDerivs), range(nFields))
            eval_dict = {**setup.constants, **setup.functions}
            for (block, eom, deriv, field), line in zip(indices, lines):
                params = ", ".join((setup.backgroundName, *setup.grid_names))
                l = " ".join(("lambda", params, ":", line))
                terms_in_eom[block][eom][deriv][field] = eval(l, eval_dict)
            return np.array(terms_in_eom)

    except IOError as e:
        logger.error(f"Error reading file: {filename}. Error:{e}")
        raise e


##@logcall()
def loadFermionEquations(setup: 'FermionSetup'):
    coefficientFolder = os.path.realpath(setup.coefficient_folder)
    coefsInternalFName = os.path.join(coefficientFolder, "FermionCoefs_I.txt")
    fermionRHSFName = os.path.join(coefficientFolder, "FermionRHS_I.txt")

    coefsInternal = readFermionCoefficients(coefsInternalFName, setup)
    fermionRHSInternal = readFermionRHS(fermionRHSFName, setup)
    # Everything might be periodic, so initialize to None for each direction
    coefsBdy = [[None, None] for _ in range(setup.dims)]
    fermionBdyRHS = [[None, None] for _ in range(setup.dims)]
    for i, periodic in enumerate(setup.grid_periodic):
        if not periodic:
            for j in range(2):
                bdyCoefsFName = os.path.join(coefficientFolder, f"FermionCoefs_{i}_{j}.txt")
                fermionRHSBdy = os.path.join(coefficientFolder, f"FermionRHS_{i}_{j}.txt")

                boundaryCoefs = readFermionCoefficients(bdyCoefsFName, setup)
                fermionRHSBdy = readFermionRHS(fermionRHSBdy, setup)

                coefsBdy[i][j] = boundaryCoefs
                fermionBdyRHS[i][j] = fermionRHSBdy

    FermionCoefs = HolographicLattices.import_equations.Equations(coefsInternal, coefsBdy)
    FermionRHS = HolographicLattices.import_equations.Equations(fermionRHSInternal, fermionBdyRHS)

    return FermionCoefs, FermionRHS


def getFermionOperator(momentumValues, linearOperatorBlocks, fermionRHSValues, setup: 'FermionSetup'):
    # From the not kx, ky, omega dependent operator blocks and righthand side, contruct the full problem.

    derivs = HolographicLattices.Utilities.GridUtilities.all_derivatives(setup.dims + 1, setup.max_deriv)

    # Have this many blocks
    multiplyingFactors = np.zeros(len(derivs))

    # The first block is always multiplied by 1
    multiplyingFactors[0] = 1
    for i in range(1, len(derivs)):
        deriv = derivs[i]
        if all(x == 0 for x in momentumValues):
            continue
        else:
            prod = 1
            for j, val in enumerate(deriv):
                if val != 0:
                    prod = prod * np.power(momentumValues[j], val)
            multiplyingFactors[i] = prod
    finalOperator = scipy.sparse.csr_matrix((setup.num_fields * setup.gridVolume, setup.num_fields * setup.gridVolume))
    finalRHS = np.zeros(setup.gridVolume * setup.num_fields, dtype=np.complex128)

    for i, factor in enumerate(multiplyingFactors):
        finalOperator += factor * linearOperatorBlocks[i]
        finalRHS += factor * fermionRHSValues[i].ravel()

    return finalOperator, finalRHS


# Example step hook:

##@logcall("INFO")
def exportFermionsHDF5(filename, fermions, setup: 'FermionSetup', blockValues,
                       observables: HolographicLattices.control_observables.FieldDependentHook):
    """
    Export the fermions to a HDF file. This does __not__ handle the loop correctly yet, but that is because it would be too
    big to use.
    :param filename:
    :param fermions:
    :param setup:
    :param blockValues:
    :param observables:
    :return:
    """
    # This is
    logger = logging.getLogger("fermions")
    with h5py.File(filename, "w") as outfile:
        # Ensure the shape is correct
        logger.info("Writing data to %s", filename)
        fermions = fermions.reshape((*fermions.shape[:2], *setup.grid_sizes))
        fermionData = outfile.create_dataset(u"fermions", data=fermions, dtype=np.complex128)
        for i, b in enumerate(setup.blocks):
            value = blockValues[i]
            fermionData.attrs[b] = value

        fermionData.attrs[u"Number of Fields"] = setup.num_fields
        fermionData.attrs.create(u"periodic", setup.grid_periodic, (setup.dims,))
        fermionData.attrs[u"tolerance"] = setup.tolerance
        fermionData.attrs[u"Number of Background fields"] = setup.nBackgroundFields

        import json
        fermionData.attrs[u"constants"] = json.dumps(setup.constants)
        fermionData.attrs[u"Finite Diff Methods"] = json.dumps(setup.eom_derivative_methods)
        fermionData.attrs[u"Finite Diff Order"] = setup.diff_order

        fermionData.attrs[u"Update Parameter"] = setup.nonlinear_update_step

        bdyVariables, bulkVariables = setup.getBoundaryBulkInput(fermions)
        bdyShapes, bulkShape = setup.boundaryBulkShapes()

        if observables is not None:
            observ = outfile.create_group(u"observables")
            if observables.interiorEquations:
                interior = observ.create_group(u"interior")
                for k, v in observables.interiorEquations.items():
                    interior.create_dataset(k, data=v(**bulkVariables).reshape(bulkShape))

            for i in range(setup.dims):
                for j in range(2):
                    eqns = observables.boundaryEquations[i][j]
                    variables = bdyVariables[i][j]
                    bdyShape = bdyShapes[i]
                    boundaryGroup = None
                    if eqns:
                        if boundaryGroup is None:
                            boundaryGroup = observ.create_group(u"boundaries")
                        for k, v in eqns.items():
                            boundaryGroup.create_dataset(k, data=v(**variables).reshape(bdyShape))
            if observables.globalEquations:
                glob = observ.create_group(u"global")
                allVariables = {**setup.background, setup.field: fermions, **setup.coordinates}
                for k, v in observables.globalEquations.items():
                    glob.create_dataset(k, data=v(**allVariables).reshape(setup.grid_sizes))


# def importBackgroundHDF5
class FermionSolver(HolographicLattices.nonlinear_solver.Solver):
    def __init__(self, setup: 'FermionSetup'):
        super().__init__()
        self.setup = setup
        self.result = None
        self.preSolveHook = lambda *args: None
        self.postSolveHook = lambda *args: None
        self.currentFunction = None

    def sourceMomenta(self):
        setup = self.setup
        sourceMomentum = setup.sourceMomentum
        sourceMomentum = sourceMomentum / np.product(setup.grid_sizes[setup.grid_periodic])
        return sourceMomentum

    def solve(self):
        setup = self.setup
        import copy
        logger = logging.getLogger("fermions")

        setupSource1 = copy.deepcopy(setup)
        setupSource1.constants["source"] = np.array([1, 0], dtype=np.complex128)

        setupSource2 = copy.deepcopy(setup)
        setupSource2.constants["source"] = np.array([0, 1], dtype=np.complex128)

        fermionCoefsSource1, RHSSource1 = loadFermionEquations(setupSource1)
        fermionCoefsSource2, RHSSource2 = loadFermionEquations(setupSource2)

        try:
            obsPath = os.path.join(setup.observablesFolder, "FermionObservables.ini")
            if not os.path.exists(obsPath):
                observable = None
                logger.warning("No control observable file found, assuming averages.")
            else:
                observable = FermionObservables("FermionObservables.ini", setup)
        except IOError as e:
            observable = None
            logger.warning("Error reading observables file.")

        fermionOperators = linearFermionOperators(fermionCoefsSource2, setupSource2)

        fermRHSS1 = fermionRHS(RHSSource1, setupSource1)
        fermRHSS2 = fermionRHS(RHSSource2, setupSource2)

        points = np.array(list(itertools.product(*setup.blockRanges.values())))
        allData = np.zeros((len(points), 4))
        if setup.allFourier:
            logger.info("All fourier components are being used")
            totalSize = 1
            for d, n in zip(setup.grid_periodic, setup.grid_sizes):
                if d:
                    totalSize *= n
            fullResponses = np.zeros((len(points), 4, totalSize), dtype=np.complex128)
        else:
            fullResponses = None
        allData[:, :-1] = points / (setup.constants["mu"])
        sigma2 = np.array([[0, -1.J], [1.J, 0]], dtype=np.complex128)
        nextPercent = 0
        if observable is not None:
            import copy
            observableSetup1 = copy.deepcopy(setupSource1)
            observableSetup2 = copy.deepcopy(setupSource2)
            observableSetup1.max_deriv = 2
            observableSetup2.max_deriv = 2
            observableSetup1.__finalize_init__()
            observableSetup2.__finalize_init__()

        for i, point in enumerate(itertools.product(*setup.blockRanges.values())):

            fullOperator1, RHS1 = getFermionOperator(point, fermionOperators, fermRHSS1, setupSource1)
            fullOperator2, RHS2 = getFermionOperator(point, fermionOperators, fermRHSS2, setupSource2)
            logger.info(f"Starting solve {i}")

            result1 = scipy.sparse.linalg.spsolve(fullOperator1, RHS1).reshape((setup.num_fields, *setup.grid_sizes))
            result2 = scipy.sparse.linalg.spsolve(fullOperator2, RHS2).reshape((setup.num_fields, *setup.grid_sizes))
            if observable is not None:
                if setup.allFourier:
                    Response_1_1_full, Response_1_2_full = observable(result1, point, observableSetup1)
                    Response_2_1_full, Response_2_2_full = observable(result2, point, observableSetup2)
                    fullResponses[i, 0] = Response_1_1_full
                    fullResponses[i, 1] = Response_1_2_full
                    fullResponses[i, 2] = Response_2_1_full
                    fullResponses[i, 3] = Response_2_2_full
                    # with h5py.File(f"testfolder/Fermion_fourier_observable_{point}.h5") as fullObservable:
                    #    Point =  fullObservable.create_dataset("Point", data=point/setup.constants["mu"])
                    #    Greenfunctions= fullObservable.create_dataset("GreenFunction", data=Grs)

                Response_1_1, Response_1_2 = observable(result1, point, observableSetup1, average=True)
                Response_2_1, Response_2_2 = observable(result2, point, observableSetup2, average=True)
            else:
                logger.warning("Observables not found! using average of 0 component")
                Response_1_1 = np.average(result1[0, :, 0])
                Response_1_2 = np.average(result1[2, :, 0])
                Response_2_1 = np.average(result2[0, :, 0])
                Response_2_2 = np.average(result2[2, :, 0])

            R = np.array([[Response_1_1, Response_2_1], [Response_1_2, Response_2_2]])

            # Reverse on 2nd axis == multiply by {{0,1}{1,0}}
            M = R[:, ::-1]
            Gr = - M.dot(sigma2)
            rho = np.trace(np.imag(Gr))
            allData[i, -1] = rho
            logger.debug(f"Evaluated point {point / (setup.constants['mu'])}, rho = {rho}")

            if i * 100 / len(allData) > nextPercent:
                logger.info(f"Currently at {i * 100 / len(allData)} percent")
                while i * 100 / len(allData) > nextPercent:
                    nextPercent += 1
        return allData, fullResponses


class FermionObservables(HolographicLattices.control_observables.FieldDependentHook):
    def __call__(self, fermionFunctions: np.ndarray, blockValues: np.array, setup: 'FermionSetup', average=False):
        logger = logging.getLogger("fermions")
        fieldsDerivatives = HolographicLattices.derivatives.take_derivative_mixed(fermionFunctions, setup)
        boundaryVariables, _ = setup.getBoundaryBulkInput(fieldsDerivatives)
        blockDict = {k: v for k, v in zip(setup.blocks, blockValues)}
        R1 = self.boundaryEquations[1][0]["r1"](**{**boundaryVariables[1][0], **blockDict})
        R2 = self.boundaryEquations[1][0]["r2"](**{**boundaryVariables[1][0], **blockDict})
        if average:
            return np.average(R1), np.average(R2)
        else:
            return R1, R2


class FermionSetup(HolographicLattices.Options.Options.Setup):
    def __init__(self, filename, args=None):
        logger = logging.getLogger("fermions")
        super().__init__(filename, args)
        if filename is not None:
            try:
                import configparser
                import json

                with open(filename, "r") as infile:
                    self.fileLines = infile.readlines()
                    self.fileLines = "\n".join([x.strip() for x in self.fileLines if len(x.strip()) > 0])

                config = configparser.ConfigParser()
                config.read(filename)

                fermions = config["Fermions"]
                self.nBackgroundFields = fermions.getint("N_BACKGROUND_FIELDS", 4)
                self.backgroundFile = fermions.get("BACKGROUND_FILE", "FermionBackgrounds.csv")
                self.backgroundName = fermions.get("BACKGROUND_NAME", "f")
                self.reinterpolate = fermions.getboolean("REINTERPOLATE", False)
                self.constants["sourceMomentum"] = json.loads(fermions.get("SOURCE_MOMENTUM", "[0]"))

                self.background = {self.backgroundName: self.__loadBackground__(self.backgroundFile)}
                self.allFourier = fermions.getboolean("ALL_FOURIER", False)

                T = self.constants["t"]
                mu = 2 * (-4 * np.pi * T + np.sqrt(3 + 16 * np.pi ** 2 * T ** 2))
                self.constants["mu"] = mu
                self.constants["mu1"] = mu
                self.constants["p0"] = mu * self.constants["plattice"] / self.constants["nperiods"]

                self.blocks = json.loads(fermions.get("BLOCKS", '["ω", "kx"]'))
                self.blockRanges = {}
                for block in self.blocks:
                    self.blockRanges[block] = np.linspace(*eval(fermions.get(block, "[1,1,1]")))
                    # This multiplies by the chemical potential to be able to evaluate mu in terms of chemical potential
                    self.blockRanges[block] = self.constants["mu"] * self.blockRanges[block]


            except IOError as e:
                logger.error(f"Error reading file: {filename}. Assuming default options:{e}")

        if args is not None:
            # Command line arguments are parsed if they are given. These override anything set in the settings file.
            import argparse
            parser = argparse.ArgumentParser(add_help=False)

            # Add options here
            parsed, _ = parser.parse_known_args(args)

            # Add saving of parsing here
        self.__finalize_init__()

    def __loadBackground__(self, filename):
        """
        Loads the background in from a file. Since this is highly nontrivial, it assumes the following structure:
        nx * ny * nz ... for num_fields fields, but it currently does not do e.g. interpolation or evaluation, it is just
        reading it in.
        :return:
        """
        logger = logging.getLogger("fermions")
        #    raise ValueError("Need to specify absolute path for this. To be fixed")
        with h5py.File(filename, "r") as inFile:
            background = inFile["/field"]
            import json
            for k, v in json.loads(background.attrs["constants"]).items():
                self.constants[k] = v

            if (np.abs(self.constants["m"]) < 1e-6 or np.abs(self.constants["m"] - 1) < 1e-6) and self.reinterpolate:
                logger.info("Mass is 0 or 1, using mass = 0 transform")
                oldData = background[:, 0, :].reshape((background.shape[0], *self.grid_sizes))
                oldGrid = self.grids["y"]
                import invertz

                newData = invertz.invertz_m0(oldData, oldGrid)
                derivNewData = HolographicLattices.derivatives.take_deriv_bg(newData, self)

                return derivNewData
            elif np.abs(self.constants["m"] - (1 / 4)) < 1e-6 and self.reinterpolate:
                logger.info("Mass is 1/4, using m = 1/4 transform")
                oldData = background[:, 0, :].reshape((background.shape[0], *self.grid_sizes))
                oldGrid = self.grids["y"]
                import invertz
                newData = invertz.invertz_m14(oldData, oldGrid)
                derivNewData = HolographicLattices.derivatives.take_deriv_bg(newData, self)
                return derivNewData


            else:
                return background[:]

    def __finalize_init__(self):
        # This method does the usual post-initialization, but it also takes the derivatives of the background field, so
        # that they can be used easily in further evaluations.
        super().__finalize_init__()

    def getVariableNames(self):
        return [*self.grid_names, self.field, self.backgroundName, *self.blocks]

    def getBoundaryBulkInput(self, fermionFields=None, *args):
        """ This gets the boundary and bulk input for a given problem, which needs to be
        overridden for the Fermion setup"""
        bulkVariables = {**self.coordinates}
        bulkVariables[self.backgroundName] = self.background[self.backgroundName].reshape(
            *self.background[self.backgroundName].shape[:2], self.gridVolume)

        for ax in self.grid_names:
            bulkVariables[ax] = bulkVariables[ax][self.bulkIndices]
        bulkVariables[self.backgroundName] = bulkVariables[self.backgroundName][:, :, self.bulkIndices]
        if fermionFields is not None:
            bulkVariables[self.field] = fermionFields.reshape((*fermionFields.shape[:2], self.gridVolume))[:, :,
                                        self.bulkIndices]

        retBoundary = [[None, None] for _ in range(self.dims)]
        for i in range(self.dims):
            if not self.grid_periodic[i]:
                for j in range(2):
                    idx = self.boundaryIndices[i][j]
                    boundaryVariables = {**self.background, **self.coordinates}
                    boundaryVariables[self.backgroundName] = self.background[self.backgroundName].reshape(
                        *self.background[self.backgroundName].shape[:2], self.gridVolume)
                    for ax in self.grid_names:
                        boundaryVariables[ax] = boundaryVariables[ax][idx]
                    boundaryVariables[self.backgroundName] = boundaryVariables[self.backgroundName][:, :, idx]

                    if fermionFields is not None:
                        boundaryVariables[self.field] = fermionFields.reshape(
                            (*fermionFields.shape[:2], self.gridVolume))[:,
                                                        :, idx]
                    retBoundary[i][j] = boundaryVariables

        return retBoundary, bulkVariables


def main():
    timeStart = time.time()
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", type=str, help="Setup file to use")
    parser.add_argument("--outfile", type=str, help="Where to output as hdf5 file")
    parser.add_argument("--logfile", type=str, help="Where to log to")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level to use. Pick one of: DEBUG, INFO, "
                                                                "WARN, ERROR")
    parsed, other_args = parser.parse_known_args()
    setupFile = parsed.setup
    print("Using setup file", setupFile)
    loggingLevel = getattr(logging, parsed.log.upper())
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", filename=parsed.logfile,
                        level=loggingLevel)
    logger = logging.getLogger("fermions")
    logger.info(f"Started running at {timeStart}")

    setup = FermionSetup(setupFile)

    solver = FermionSolver(setup)

    logger.info("Loaded fermions successfully")

    if parsed.outfile is not None:
        f = h5py.File(parsed.outfile, "w")
    else:
        f = h5py.File(setup.outfile, "w")

    logger.info(f"Saving to {f.filename}")
    result, resultFullMat = solver.solve()
    import json
    sF = f.create_dataset("SpectralFunction", data=result)
    sF.attrs["shape"] = np.array([len(setup.blockRanges[block]) for block in setup.blocks])
    sF.attrs["constants"] = json.dumps(setup.constants)
    sF.attrs["setup"] = setup.fileLines
    logger.info("Created spectral function")
    if resultFullMat is not None:
        logger.info("Also outputting full matrix")
        resp = f.create_dataset("FullResponses", data=resultFullMat, dtype=np.complex128)
        resp.attrs["shape"] = np.array([len(setup.blockRanges[block]) for block in setup.blocks])
        resp.attrs["constants"] = json.dumps(setup.constants)
        resp.attrs["setup"] = setup.fileLines
    f.close()
    timeEnd = time.time()
    logger.info(f"Time taken: {timeEnd - timeStart}")


if __name__ == "__main__":
    main()
