import math
import numbers
import sys
import os
import numpy as np
import pandas as pd

from .qm import qm_spinsystem
import numba as nb
import scipy

# Add this at the top of your file, before importing jax
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'  # Use only 50% of GPU memory

import jax.numpy as jnp
from jax import jit, vmap


@nb.njit(parallel=True)
def _createLineshape_numba(
    peaklist_array, points, l_limit, r_limit, function_type
):
    """
    Numba-compiled core function for createLineshape.

    Parameters
    ----------
    peaklist_array : numpy.array
        Array of shape (N, 3) with (frequency, intensity, width) data.
    points : int
        Number of data points.
    l_limit : float
        Left frequency limit.
    r_limit : float
        Right frequency limit.
    function_type : int
        0 for lorentzian, 1 for gaussian.

    Returns
    -------
    x, y : numpy.array
        Arrays for frequency (x) and intensity (y) for the simulated lineshape.
    """
    x_inv = np.linspace(l_limit, r_limit, points)
    x = x_inv[::-1]  # reverses the x axis

    if len(peaklist_array) > 0:
        if function_type == 0:  # lorentzian
            y = add_lorentzians(x, peaklist_array)
        elif function_type == 1:  # gaussian
            y = add_gaussians(x, peaklist_array)
        else:
            y = np.zeros(points)
    else:
        y = np.zeros(points)
    return x, y


def createLineshape(
    peaklist, points=65536, limits=None, function='lorentzian'
):
    peaklist_array = np.array(peaklist)
    if len(peaklist_array) > 0:
        sort_indices = np.argsort(peaklist_array[:, 0])
        peaklist_array = peaklist_array[sort_indices]

    if limits:
        l_limit, r_limit = validate_and_sort_limits(limits)
    else:
        if len(peaklist_array) > 0:
            l_limit = peaklist_array[0, 0] - 0.5
            r_limit = peaklist_array[-1, 0] + 0.5
        else:
            l_limit = -0.5
            r_limit = 0.5

    function_type = 0 if function == 'lorentzian' else 1

    # Return the parameters needed for _createLineshape_numba
    # return peaklist_array, points, l_limit, r_limit, function_type

    return _createLineshape_numba(
        peaklist_array, points, l_limit, r_limit, function_type
    )


def validate_and_sort_limits(t):
    try:
        m, n = t
        if not isinstance(m, numbers.Real) or not isinstance(n, numbers.Real):
            raise TypeError
        return (min(m, n), max(m, n))
    except Exception:
        raise TypeError('limits must be a tuple of two real numbers.')


@nb.njit(parallel=True, fastmath=True)
def add_lorentzians(linspace, peaklist):
    result = np.zeros(linspace.shape, dtype=np.float64)
    for i in nb.prange(len(linspace)):
        v = linspace[i]
        total = 0.0
        for j in range(len(peaklist)):
            v0, I, w = peaklist[j]
            hw = 0.5 * w
            total += I * (hw * hw) / (hw * hw + (v - v0) ** 2)
        result[i] = total
    return result


@nb.njit(parallel=True, fastmath=True)
def add_gaussians(linspace, peaklist):
    result = np.zeros(linspace.shape, dtype=np.float64)
    wf = 0.4246609
    for i in nb.prange(len(linspace)):
        v = linspace[i]
        total = 0.0
        for j in range(len(peaklist)):
            v0, I, w = peaklist[j]
            total += I * np.exp(-((v - v0) ** 2) / (2 * ((w * wf) ** 2)))
        result[i] = total
    return result


def peakListFromSpinSystemMatrix(spinSystemMatrix, frequency, width):
    """
    Numpy Arrays of the simulated lineshape for a peaklist.
    Parameters
    ----------
    peaklist : [(float, float, float)...]
        A list of (frequency, intensity, width) tuples.
    y_min : float or int
        Minimum intensity for the plot.
    y_max : float or int
        Maximum intensity for the plot.
    points : int
        Number of data points.
    limits : (float, float)
        Frequency limits for the plot.
    function: string
        Plotting function for the peak shape, either lorentzian or gaussian.
    Returns
    -------
    x, y : numpy.array
        Arrays for frequency (x) and intensity (y) for the simulated lineshape.
    """
    breaks = [
        index + 1
        for index in range(len(spinSystemMatrix) - 1)
        if not np.any(spinSystemMatrix[: index + 1, index + 1 :] != 0)
    ]
    breaks = [0] + breaks + [len(spinSystemMatrix)]
    matrices = [
        spinSystemMatrix[
            breakpoint : breaks[index + 1], breakpoint : breaks[index + 1]
        ]
        for index, breakpoint in enumerate(breaks[:-1])
    ]
    # Suppress the printing from qm
    output = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    peaklist = [
        item
        for sublist in [
            qm_spinsystem(
                [ssm[index][index] * frequency for index in range(len(ssm))],
                ssm,
            )
            for ssm in matrices
        ]
        for item in sublist
    ]
    sys.stdout = output
    peaklist_out = [(peak[0] / frequency, peak[1], width) for peak in peaklist]
    # df_out = pd.DataFrame(peaklist_out, columns=['chemical_shift', 'height', 'width', 'multiplet_id'])
    return peaklist_out
