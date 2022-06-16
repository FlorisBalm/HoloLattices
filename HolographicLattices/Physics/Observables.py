import logging 
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Callable

import numpy as np
import yaml
from yaml import Loader as loader

import HolographicLattices.Options.Options
import HolographicLattices.Utilities.GridUtilities


@dataclass
class Observables:
    equation_options: HolographicLattices.Options.Options.EquationOptions

    interiorObservables: Dict[str, Union[Callable, None]] = field(default_factory=list)
    globalObservables: Dict[str, Union[Callable, None]] = field(default_factory=list)
    boundaryObservables: List[List[Dict[str, Union[Callable, None]]]] \
        = field(default_factory=lambda: [[{}, {}]])

    def evaluate(self, fields_and_derivatives: np.ndarray,
                 grid_information: HolographicLattices.Utilities.GridUtilities.GridInformation):

        evaluated_dict = {}
        equation_options = self.equation_options
        field_name = equation_options.field
        coordinates = equation_options.coordinates

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

        if self.globalObservables:
            for k, v in self.globalObservables.items():
                evaluated_dict[k] = v(**input_data)

        if self.interiorObservables:
            for k, v in self.interiorObservables.items():
                evaluated_dict[k] = v(**bulk_variables)

        if self.boundaryObservables:
            for axis in range(grid_information.options.dims):
                for end in range(2):
                    if not grid_information.options.grid_periodic[axis]:
                        idx = boundary_indices[axis][end]
                        boundary_variables = {}

                        for ax in coordinates:
                            boundary_variables[ax] = input_data[ax][idx]

                        boundary_variables[field_name] = input_data[field_name][:, :, idx]
                        if self.boundaryObservables[axis][end]:
                            for k, v in self.boundaryObservables[axis][end].items():
                                evaluated_dict[k] = v(**boundary_variables)
        return evaluated_dict

    @classmethod
    def parse_observables(cls, infile, constants: HolographicLattices.Options.Options.ConstantOptions, equationOptions:
    HolographicLattices.Options.Options.EquationOptions):
        logger = logging.getLogger(__name__)

        # Gather all input as usual
        constant_parameters = constants.get_all_defined_parameters()
        field_input = ", ".join([*equationOptions.coordinates, equationOptions.field])

        infile_as_path = Path(infile)

        try:
            with infile_as_path.open("r") as observables_file:
                observables = yaml.load(observables_file, Loader=loader)
                # Go through all global/interior/boundary observables (if they exist)
                construction_dict = {"globalObservables": {}, "interiorObservables": {},
                                     "boundaryObservables": [[{}, {}] for _ in range(equationOptions.dims)]}
                if observables is not None:

                    if "Global" in observables:
                        construction_dict["globalObservables"] = {}

                        for observable_name, equation in observables["Global"].items():
                            observable_as_lambda = " ".join(("lambda", field_input, ":", equation))
                            construction_dict["globalObservables"][observable_name] = eval(observable_as_lambda,
                                constant_parameters)

                    if "Interior" in observables:
                        construction_dict["interiorObservables"] = {}

                        for observable_name, equation in observables["Interior"].items():
                            observable_as_lambda = " ".join(("lambda", field_input, ":", equation))
                            construction_dict["interiorObservables"][observable_name] = eval(observable_as_lambda,
                                constant_parameters)

                    for d in range(equationOptions.dims):
                        for end in (0, 1):
                            config_str = f"Bdy_{d}{end}"

                            if config_str in observables:
                                for observable_name, equation in observables[config_str].items():
                                    observable_as_lambda = " ".join(("lambda", field_input, ":", equation))
                                    construction_dict["boundaryObservables"][d][end][observable_name] = eval(
                                               observable_as_lambda, constant_parameters)

                else:
                    logger.warn("No observables found in observables file. Is this intentional?")
                construction_dict["equation_options"] = equationOptions
                return cls(**construction_dict)

        except IOError as e:
            logger.error(f"Could not load observables from file {infile}. Reason: {e}")
            import sys 
            
