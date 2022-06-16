import logging
from pathlib import Path
import yaml
import h5py
import numpy as np

from HolographicLattices.Options.Options import ConstantOptions, EquationOptions
from HolographicLattices.Utilities.GridUtilities import GridInformation


def load_seed_algebraic(path: Path, constant_options: ConstantOptions, grid_information : GridInformation) -> np.ndarray:
    path = Path(path)
    with path.open("r") as infile:
        logger = logging.getLogger(__name__)
        lines = infile.readlines()

        grid_shape = grid_information.options.grid_sizes
        grid_volume = np.product(grid_shape)
        num_fields = grid_information.options.num_fields

        full_grids = [grid.ravel() for grid in np.meshgrid(*grid_information.get_1d_grids(), indexing="ij")]
        variables = {k:v for k, v in zip(grid_information.options.coordinates, full_grids)}
        seed_return = np.zeros((num_fields,grid_volume),dtype=grid_information.options.field_dtype)
        # Evaluate each term. If it was a constant, assume that it takes that constant value
        # at each of the gridpoints
        for i, line in enumerate(lines):
            field_seed_evaluated = eval(line, {**constant_options.constants, **constant_options.functions, **variables})

            if not hasattr(field_seed_evaluated, "shape"):
                field_seed_evaluated = np.zeros(grid_volume,dtype=seed_return.dtype) + float(field_seed_evaluated)
            seed_return[i] = field_seed_evaluated
        return seed_return


def load_seed(path: Path, equation_options : EquationOptions):
    logger = logging.getLogger(__name__)
    shape = equation_options.grid_sizes
    with h5py.File(path, "r") as infile:
        if "seed" in infile:
            seed = infile["seed"]
            logger.info(f"Seed found, Legacy style, shape={shape}")
            return seed[:].reshape((shape[0], np.product(shape[1:])))
        elif "field" in infile:
            seed = infile["field"]
            logger.info(f"Seed found, New style, shape={shape}")
            constant_options = yaml.load(infile["options/constant_options"][()], Loader=yaml.Loader)
            return seed[:, 0, :].reshape((shape[0], np.product(shape[1:]))).astype(equation_options.field_dtype), constant_options
        else:
            raise ValueError("No valid seed found")