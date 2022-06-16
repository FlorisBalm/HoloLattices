import dataclasses
from dataclasses import dataclass
from typing import List, Dict

import HolographicLattices.Options.Options
from pathlib import Path

@dataclass
class FermionOptions(HolographicLattices.Options.Options.EquationOptions):
    # TODO: Implement
    field: str = dataclasses.field(default_factory=lambda : "psi")
    field_dtype : str = dataclasses.field(default_factory=lambda : "complex")
    background_field: str = dataclasses.field(default_factory=lambda : "f")
    fermion_params: List[str] = dataclasses.field(default_factory=lambda : ["omega", "kx", "ky"])
    max_bg_deriv : int = 2
    omega : List = dataclasses.field(default_factory=lambda : [0.01, 0.01, 1])
    kx : List = dataclasses.field(default_factory=lambda : [0.01, 0.01, 1])
    ky : List = dataclasses.field(default_factory=lambda : [0.01, 0.01, 1])


@dataclass
class FermionConstantOptions(HolographicLattices.Options.Options.ConstantOptions):
    read_options_from_background_file : bool = True

    def read_constants_from_background_file(self, file_path) -> Dict[str, float]:
        import h5py
        import yaml

        with h5py.File(file_path, "r") as data_file:
            consts = yaml.load(data_file["/options/constant_options"][()], Loader=yaml.Loader)
            return consts.constants

def load_background_field(file_path):
    import h5py
    with h5py.File(file_path, "r") as data_file:
        return data_file["field"][:]
