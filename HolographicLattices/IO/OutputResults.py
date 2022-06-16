import dataclasses
import logging
from pathlib import Path

import h5py
import numpy as np
import yaml
from typing import Callable

from HolographicLattices.Options import Options


class Result:
    """Simple class that collects all the information about a solve"""

    def __init__(self, observables, fields, last_update, eom_residual,
                 equation_options: Options.EquationOptions, constant_options: Options.ConstantOptions):
        self.observables = observables
        self.fields = fields
        self.last_update = last_update.reshape((equation_options.num_fields, *equation_options.grid_sizes))
        self.eom_residual = eom_residual.reshape((equation_options.num_fields, *equation_options.grid_sizes))
        self.equation_options = equation_options
        self.constant_options = constant_options

    def write_to_file(self, outfile: Path):
        logger = logging.getLogger(__name__)
        try:
            if Path(outfile).exists():
                pass

            with h5py.File(outfile, "w") as output:
                field_output = output.create_dataset("field", data=self.fields)
                field_output.attrs["grid_size"] = self.equation_options.grid_sizes
                # # This is just legacy support, it does not do much
                # for k, v in {**dataclasses.fields(self.grid_options), **dataclasses.fields(self.equation_options),
                #              **dataclasses.fields(self.constant_options)}.items():
                #     field_output.attrs[k] = np.string_(v)

                if self.observables:
                    observableGroup = output.create_group("observables")
                    for k, v in self.observables.items():
                        obs = observableGroup.create_dataset(k, data=v)

                        # keep scalars as constants
                        if np.isscalar(v):
                            self.constant_options.constants[k] = np.array(v).item()
                        elif isinstance(v, (np.ndarray, np.generic)) and v.size == 1:
                            self.constant_options.constants[k] = v.item(0)

                last_update_output = output.create_dataset("last_update", data=self.last_update)
                eom_output = output.create_dataset("eom_residual", data=self.eom_residual)
                options_group = output.create_group("options")
                eq_opts = options_group.create_dataset("equation_options", data=yaml.dump(self.equation_options))
                grid_opts = options_group.create_dataset("constant_options",
                                                         data=yaml.dump(self.constant_options.constants))
            logger.info(f"Successfully wrote output to {outfile}")

        except IOError as e:
            logger.error(f"Could not write to file {outfile}. Reason: {e}.")
            logger.debug(yaml.dump(self))
            raise e
