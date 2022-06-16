import itertools
import logging
import dataclasses
import numpy as np
import scipy.sparse
import typing
import yaml

# My own
import HolographicLattices.Equations.EquationsBase
import HolographicLattices.SystemOnBackground.BackgroundOptions
import HolographicLattices.Utilities.GridUtilities
import HolographicLattices.Options.Options


from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Dict, Callable, Union

PathLike = typing.Union[str, Path]

@dataclass
class PerturbationObservables:

    equation_options: HolographicLattices.Options.Options.EquationOptions
    background_options: HolographicLattices.SystemOnBackground.BackgroundOptions.BackgroundOptions
    constant_options : HolographicLattices.Options.Options.ConstantOptions
    interiorObservables: Dict[str, Union[Callable, None]] = field(default_factory=list)
    globalObservables: Dict[str, Union[Callable, None]] = field(default_factory=list)
    boundaryObservables: List[List[Dict[str, Union[Callable, None]]]] \
        = field(default_factory=lambda: [[{}, {}]])

    def evaluate(self, fields_and_derivatives: np.ndarray, bg_fields_derivs,
                 grid_information: HolographicLattices.Utilities.GridUtilities.GridInformation):

        evaluated_dict = {}
        field_name = self.equation_options.field

        bg_field_name=self.background_options.background_field_name

        coordinates = self.equation_options.coordinates

        boundary_indices, bulk_indices = HolographicLattices.Utilities.GridUtilities.get_boundaries_and_bulk_indices(
            grid_information.options.grid_sizes,
            grid_information.options.grid_periodic)

        # Create a local dict of variables to only evaluate in the bulk points
        full_grids = [grid.ravel() for grid in np.meshgrid(*grid_information.get_1d_grids(), indexing="ij")]

        input_data = {field_name: fields_and_derivatives,
                      bg_field_name: bg_fields_derivs,
                      **{name: grid for name, grid in zip(coordinates,
                                                          full_grids)}}

        bulk_variables = {}

        for ax in coordinates:
            bulk_variables[ax] = input_data[ax][bulk_indices]

        bulk_variables[field_name] = input_data[field_name][:, :, bulk_indices]
        bulk_variables[bg_field_name] = input_data[bg_field_name][:, :, bulk_indices]

        if self.globalObservables:
            for k, v in self.globalObservables.items():
                evaluated_dict[k] = v(**input_data)

        if self.interiorObservables:
            for k, v in self.interiorObservables.items():
                evaluated_dict[k] = v(**input_data)

        if self.boundaryObservables:
            for axis in range(grid_information.options.dims):
                for end in range(2):
                    idx = boundary_indices[axis][end]
                    boundary_variables = {}

                    for ax in coordinates:
                        boundary_variables[ax] = input_data[ax][idx]

                    boundary_variables[field_name] = input_data[field_name][:, :, idx]
                    boundary_variables[bg_field_name] = input_data[bg_field_name][:, :, idx]

                    if self.boundaryObservables[axis][end]:
                        for k, v in self.boundaryObservables[axis][end].items():
                            evaluated_dict[k] = v(**boundary_variables)
        return evaluated_dict

    @classmethod
    def parse_observables(cls, infile,
                          constant_options: HolographicLattices.Options.Options.ConstantOptions,
                          background_options: HolographicLattices.SystemOnBackground.BackgroundOptions.BackgroundOptions,
                          equation_options: HolographicLattices.Options.Options.EquationOptions):

        logger = logging.getLogger()

        constant_parameters = {**constant_options.get_all_defined_parameters(), **background_options.get_all_defined_parameters()}


        field_input = ", ".join([*equation_options.coordinates, equation_options.field, background_options.background_field_name])
        infile_as_path = Path(infile)
        try:
            with infile_as_path.open("r") as observables_file:
                loader = yaml.Loader
                observables = yaml.load(observables_file, Loader=loader)

                construction_dict = {"globalObservables": {}, "interiorObservables": {},
                                     "boundaryObservables": [[{}, {}] for _ in range(equation_options.dims)]}

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

                for d in range(equation_options.dims):
                    for end in (0, 1):
                        config_str = f"Bdy_{d}{end}"

                        if config_str in observables:
                            for observable_name, equation in observables[config_str].items():
                                observable_as_lambda = " ".join(("lambda", field_input, ":", equation))
                                construction_dict["boundaryObservables"][d][end][observable_name] = eval(
                                    observable_as_lambda, constant_parameters)

                construction_dict["equation_options"] = equation_options
                construction_dict["constant_options"] = constant_options
                construction_dict["background_options"] = background_options
                return cls(**construction_dict)

        except IOError as e:
            logger.error(f"Could not load observables from file {infile}. Reason: {e}")
