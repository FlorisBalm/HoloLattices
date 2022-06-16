from dataclasses import dataclass, field
from pathlib import Path

import h5py
from typing import Callable, Dict, Union, List, Any
import numpy as np
import HolographicLattices.Utilities.GridUtilities
import HolographicLattices.Options.Options

import yaml
import logging


@dataclass
class BackgroundOptions(HolographicLattices.Options.Options.OptionsBase):

    background_field_name: str =field(default_factory=lambda: "f")
    background_dim: int = 2
    max_background_deriv: int = 2
    petsc_ordering: bool = True
    background_filename: str = field(default_factory=lambda: "Background.h5")
    background_constant_ordering: list = field(default_factory=lambda: [])

    constants: Dict[str, Union[float, complex]] = field(default_factory=lambda: {})
    functions: Dict[str, Callable] = field(default_factory=lambda: {
        "real": np.real,
        "imag": np.imag
    })
    background_field : np.ndarray = field(init=False)

    def __post_init__(self):
        self._load_bg_from_file(self.background_filename)
        pass

    def _load_bg_from_file(self, background_filename):
        logger = logging.getLogger(__name__)
        with h5py.File(background_filename, "r") as background:
            if "result" in background:
                result = background["result"][:]
                if self.petsc_ordering:
                    logger.info("Using petsc ordering")
                    result = result.transpose(range(len(result.shape))[::-1])

                self.background_field = result
                parameters = background["parameters"][:]
                self.constants = {**self.constants,
                                  **{k: v for k, v in zip(self.background_constant_ordering, parameters)}}
                #print(self.constants)

            elif "field" in background:
                result = background["field"][:, 0, :]
                self.background_field = result

                import yaml
                opts = background["options/constant_options"][()]

                self.constants = {**self.constants,
                                  **yaml.load(background["options/constant_options"][()], Loader=yaml.SafeLoader)}
            else:
                raise ValueError("No valid file found")

    def get_all_defined_parameters(self):
        return {**self.constants, **self.functions}

    def slice_background(self, indexing_slice: tuple):
        self.background_field = np.squeeze(self.background_field[indexing_slice])

    @classmethod
    def _class_opt_name(cls):
        return "BackgroundOptions"
