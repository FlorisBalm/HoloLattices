import logging

import numpy as np
import scipy.interpolate

"""
ChangeCoordinates.py 


This file contains a way to change the coordinates of the system in case they have been changed.
The changes would usually appear due to compactness and/or singularities at the boundaries. The invertz 
function can take any function relating the two coordinates, e.g. y = (1 - (1-z**2)**2) or variations. 
"""


def change_coordinates(data: np.ndarray, old_grid, coordinate_relation):
    new_grid = coordinate_relation(old_grid)
    logger = logging.getLogger(__name__)
    logger.debug(f'Reinterpolating. Old grid: {old_grid}, new grid: {new_grid}')

    def reinterpolate(values):
        interpolation = scipy.interpolate.interp1d(old_grid, values, kind="cubic", assume_sorted=True)
        return interpolation(new_grid)

    result = np.apply_along_axis(reinterpolate, axis=-1, arr=data)
    return result


def invert_z_to_r_m_0(data: np.ndarray, old_grid):
    return change_coordinates(data, old_grid, lambda z: 1 - np.sqrt(1 - z))


def invert_z_to_r_m_14(data: np.ndarray, old_grid):
    return change_coordinates(data, old_grid, lambda z: 1 - np.sqrt(1 - np.sqrt(z)))
