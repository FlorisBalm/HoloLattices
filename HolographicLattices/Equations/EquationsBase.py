from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Any

import HolographicLattices.Options.Options


@dataclass
class EquationsBase(metaclass=ABCMeta):
    constant_options: HolographicLattices.Options.Options.ConstantOptions
    equation_options: HolographicLattices.Options.Options.EquationOptions

    internal_equations: Any = field(init=False)
    boundary_equations: Any = field(init=False)

    @abstractmethod
    def evaluate(self, fields_and_derivatives,grid_information, output_prealloc):
        pass

    @abstractmethod
    def load_from_folder(self, folder: Path, periodicities, file_base):
        pass

    def _load_from_folder_impl(self, folder: Path, periodicities, file_base: str = None):
        dimensions = len(periodicities)
        file_base_full = str((Path(folder) / file_base).resolve())
        internal_file_name = Path("_".join((file_base_full,"I.txt")))
        self.internal_equations = self.read_file(internal_file_name, self.equation_options, self.constant_options)

        boundary_file_names: List[Union[List[Path], List[type(None)]]] = \
            [[Path("_".join((file_base_full,f"{dim}_{end}.txt"))) for end in [0, 1]] if not periodicities[dim]
             else [None, None] for dim in range(dimensions)]

        self.boundary_equations = [[None,None] for dim in range(dimensions)]
        for dim in range(dimensions):
            for end in (0, 1):
                if boundary_file_names[dim][end] is not None:
                    self.boundary_equations[dim][end] = self.read_file(boundary_file_names[dim][end],
                                                                       self.equation_options, self.constant_options)

    @classmethod
    @abstractmethod
    def read_file(cls, path, equation_options, constant_options):
        pass

