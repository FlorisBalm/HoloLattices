"""
The New-Style options work as follows:

implemented as a dataclass, it uses the name of the class to look in a setup file for the relevant options.
It then tries
"""
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Union, Callable,Any

import numpy as np
import yaml
from yaml import SafeLoader as SafeLoader


@dataclass
class OptionsBase:

    """
    This is where the magic happens. Subclassing this allows it to be trivially read from a file.
    The way it works is that it looks for the names that are given as data members of the objects,
    and then tries to read those from the input file. It works quite well, and multiple inheritance
    makes it so that you can combine classes in options (see e.g. IOOptions <- (InputOptions | OutputOptions)
    It emits warnings if options are provided that are not available, or if options are available
    that are not provided (Remember: Specificity is better as it causes fewer surprises)
    """

    @classmethod
    def load_from_file(cls, infile: str):
        """
        This is the bulk method that does all the magic in the loading
        :param infile:
        :return:
        """
        logger = logging.getLogger(__name__)

        # Basically filename.class. Useful if you're subclassing somewhere else.
        qualified_class_name = '.'.join([cls.__module__, cls.__name__])

        try:
            with open(infile, "r") as option_file:
                # The string class opt name (e.g. equations, nonlinearsolver etc...) is used to find which section
                # of the yaml file to load
                options_dict = yaml.load(option_file, Loader=SafeLoader)[cls._class_opt_name()]

                # Options that can be set in the current class
                variables_named_in_class = set(f.name for f in dataclasses.fields(cls))

                # If options are given in the yaml file that are not available, warn about it
                for option, value in options_dict.items():
                    if option not in variables_named_in_class:
                        logger.debug(
                            f"Option {option} was set to {value} in configuration file {infile}, but is not a "
                            f"valid option for {qualified_class_name}. "
                            f"Option was ignored.")

                # Conversely, if there are options missing in the yaml file, warn about that here
                for option in variables_named_in_class:
                    if option not in options_dict:
                        logger.debug(f"Option {option} was not provided in configuration file {infile} for "
                                       f"{qualified_class_name}. Default was used")

                # Only pass those that can actually be set
                valid_options = {option: value for option, value in options_dict.items()
                                 if option in variables_named_in_class}

                # Constructor specified by @dataclass decorator
                return cls(**valid_options)

        except IOError as e:
            # If the file can't be read, that's almost surely unintended, and would otherwise run
            # with the wrong options. This is most likely unrecoverable, so we rethrow.
            logger.error(f"Could not read {qualified_class_name} from file {infile}. Reason: {e}.")
            raise e

    @classmethod
    def _class_opt_name(cls):
        """
        Part of the magic: Subclasses replace cls.__name__ with their own name, making it that they can be easily
        imported. This should be overridden for deeper subclasses than the basic list provided here. These are currently
         - EquationOptions
         - GridOptions
         - SolverOptions
         - InputOptions
         - OutputOptions
         - ConstantOptions
         - ...

         If you want your options to look for something that isn't their class name (for whatever reason), this
         will indicate what will be sought for.
        :return: Name that will be used in parsing the YAML file to find the relevant options.
        """
        return cls.__name__


@dataclass
class EquationOptions(OptionsBase):
    dims: int = 1
    num_fields: int = 1
    num_eqs_of_motion: int = 1
    max_deriv: int = 2
    diff_order: int = 4

    grid_sizes: List[int] = field(default_factory=lambda: [10])
    grid_domains: List[float] = field(default_factory=lambda: [1.0])
    grid_spacings: List[str] = field(default_factory=lambda: ["unif"])

    grid_periodic: List[bool] = field(default_factory=lambda: [True])
    eom_derivative_methods: List[str] = field(default_factory=lambda: ["fdd"])

    field: str = dataclasses.field(default_factory=lambda: "f")
    coordinates: List[str] = dataclasses.field(default_factory=lambda: ["x"])
    field_dtype : Union[str,Any] = dataclasses.field(default_factory=lambda: "float")

    def __post_init__(self): # Verification and initialisation
        for spacing, method, periodicity in zip(self.grid_spacings, self.eom_derivative_methods, self.grid_periodic):
            # These test for all the valid combination, these are all that I can think of. Some of
            # the combinations may not be the most logical, but they should still give some indication
            # of where the problem is.
            if spacing == "cheb" and (method != "fdd" and method != "chebspectral"):
                raise ValueError(f"For grid spacing {spacing}, expect methods fdd or chebspectral. Got: {method}")
            if method == "fft" and spacing != "unif":
                raise ValueError(f"For fft derivatives, require uniform spacing. Got: {spacing}")
            if method == "fft" and not periodicity:
                raise ValueError(f"For fft derivatives, require periodic domain. Got: {periodicity}")
            if method == "chebspectral" and spacing != "cheb":
                raise ValueError(f"For chebyshev spectral derivatives, require chebyshev spacing. Got: {spacing}")
            if method == "cheb" and periodicity:
                raise ValueError(f"For chebyshev spectral derivatives, require non-periodic domain. Got: {periodicity}")

        dtype = self.field_dtype
        if dtype == "float" or dtype == "real" or dtype == "float64":
            self.field_dtype = np.float64
        elif dtype == "float128" or dtype == "longdouble":
            self.field_dtype = np.longdouble

        elif dtype == "cpx" or dtype == "complex" or dtype == "complex128" or dtype=="complex128":
            self.field_dtype = np.complex128
        else:
            raise ValueError(f"Wrong option for {self.__name__}")

@dataclass
class SolverOptions(OptionsBase):
    tolerance: float = 1e-6

@dataclass
class NonLinearSolverOptions(SolverOptions):
    nonlinear_update_step: float = 1.0
    max_nonlinear_steps: int = 20
    @classmethod
    def _class_opt_name(cls):
        return "SolverOptions"


@dataclass
class PreconditionedSolverOptions(SolverOptions):
    preconditioning: bool = True
    preconditioning_diff_order: int = 4

    @classmethod
    def _class_opt_name(cls):
        return "SolverOptions"


@dataclass
class NonlinearPreconditionedSolverOptions(PreconditionedSolverOptions, NonLinearSolverOptions):
    @classmethod
    def _class_opt_name(cls):
        return "SolverOptions"


@dataclass
class InputOptions(OptionsBase):
    coefficient_folder: str = "."
    observables_file: str = "Observables.yaml"
    seed_algebraic: bool = False
    seed_file: str = "seed.txt"
    use_constants_from_seed: bool = False


@dataclass
class OutputOptions(OptionsBase):
    output_file: str = "Temp.h5"


@dataclass
class IOOptions(InputOptions, OutputOptions):

    @classmethod
    def _class_opt_name(cls):
        return "IOOptions"


@dataclass
class ConstantOptions(OptionsBase):
    """
    Has two fields: constants and functions. Function are things like sin, cos, pow, exp etc,
    where constants are pi and others, such as
    """
    constants: Dict[str, Union[float, complex]] = field(
        default_factory=lambda: {"pi": np.pi})

    functions: Dict[str, Callable] = field(
        default_factory=lambda: {"cos" : np.cos,
                                 "sin" : np.sin,
                                 "tan" : np.tan,
                                 "cosh": np.cosh,
                                 "sinh": np.sinh,
                                 "tanh": np.tanh,
                                 "sec" : lambda x: 1.0/np.cos(x),
                                 "csc" : lambda x: 1.0/np.sin(x),
                                 "cot" : lambda x: 1.0/np.tan(x),
                                 "sech": lambda x: 1.0/np.cosh(x), # numpy does not support reciprocal functions out of the box
                                 "csch": lambda x: 1.0/np.sinh(x),
                                 "coth": lambda x: 1.0/np.tanh(x),
                                 "sqrt": np.sqrt,
                                 "exp" : np.exp,
                                 "pow" : np.power,
                                 "log" : np.log,
                                 "rand": np.random.rand
                                 })

    def get_all_defined_parameters(self):
        return {**self.constants, **self.functions}

    def __post_init__(self):
        self.constants["pi"] = np.pi

    @classmethod
    def load_from_file_unsafe(cls, infile: str):
        """
        This method works slightly differently: Since we expressly do NOT want to have to specify all functions
        again, we just add or replace in the default set of functions instead of expecting something more.
        The normal set of functions should be sufficient, however it is of course possible that other functions will
        be needed.
        :param infile:
        :return:
        """
        logger = logging.getLogger(__name__)
        try:
            with open(infile, "r") as option_file:
                options_dict = yaml.load(option_file, Loader=SafeLoader)[cls._class_opt_name()]
                constants = options_dict["constants"]
                return_instance = cls(constants=constants)

                if "functions" in options_dict:
                    for k, v in options_dict["functions"].items():
                        return_instance.functions[k] = eval(v)

                return return_instance

        except IOError as e:
            logger.error(f"Could not read IO Options from file {infile}. Reason: {e}.")
            raise e

    @classmethod
    def load_from_file_safe(cls, infile: str):
        logger = logging.getLogger(__name__)
        try:
            with open(infile, "r") as option_file:
                options_dict = yaml.load(option_file, Loader=SafeLoader)[cls._class_opt_name()]
                constants = options_dict["constants"]
                return_instance = cls(constants=constants)
                return return_instance

        except IOError as e:
            logger.error(f"Could not read IO Options from file {infile}. Reason: {e}.")
            raise e

    @classmethod
    def load_from_file(cls, infile: str):
        logger = logging.getLogger(__name__)
        logger.info("Initializing ConstantOptions safely from file using load_from_file_safe."
                       " Custom defined functions CAN NOT be parsed"
                       " this way due to safety in eval. Use load_from_file_unsafe to get expected behaviour")
        return cls.load_from_file_safe(infile)

