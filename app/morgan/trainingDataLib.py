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


@nb.njit
def lorentz(v, v0, I, w):
    """
    A lorentz function that takes linewidth at half intensity (w) as a
    parameter.
    When `v` = `v0`, and `w` = 0.5 (Hz), the function returns intensity I.
    Arguments
    ---------
    v : float
        The frequency (x coordinate) in Hz at which to evaluate intensity (y
        coordinate).
    v0 : float
        The center of the distribution.
    I : float
        the relative intensity of the signal
    w : float
        the peak width at half maximum intensity
    Returns
    -------
    float
        the intensity (y coordinate) for the Lorentzian distribution
        evaluated at frequency `v`.
    """
    return I * ((0.5 * w) ** 2 / ((0.5 * w) ** 2 + (v - v0) ** 2))


@nb.njit
def add_lorentzians(linspace, peaklist):
    """
    Adapted from nmrsim
    Given a numpy linspace, a peaklist of (frequency, intensity, width)
    tuples, and a linewidth, returns an array of y coordinates for the
    total line shape.
    Arguments
    ---------
    linspace : array-like
        Normally a numpy.linspace of x coordinates corresponding to frequency
        in Hz.
    peaklist : [(float, float)...]
        A list of (frequency, intensity) tuples.
    w : float
        Peak width at half maximum intensity.
    Returns
    -------
    [float...]
        an array of y coordinates corresponding to intensity.
    """
    result = np.zeros(linspace.shape)
    for i in range(len(peaklist)):
        result += lorentz(
            linspace, peaklist[i, 0], peaklist[i, 1], peaklist[i, 2]
        )
    return result


@nb.njit
def gauss(v, v0, I, w):
    """
    A gaussian function that takes linewidth at half intensity (w) as a
    parameter.

    Arguments
    ---------
    v : float
        The frequency (x coordinate) in Hz at which to evaluate intensity (y
        coordinate).
    v0 : float
        The center of the distribution.
    I : float
        the relative intensity of the signal
    w : float
        the peak width at half maximum intensity
    Returns
    -------
    float
        the intensity (y coordinate) for the Gaussian distribution
        evaluated at frequency `v`.
    """
    wf = 0.4246609
    return I * np.exp(-((v - v0) ** 2) / (2 * ((w * wf) ** 2)))


@nb.njit
def add_gaussians(linspace, peaklist):
    """
    Given a numpy linspace and a peaklist of (frequency, intensity, width)
    tuples, returns an array of y coordinates for the total line shape.

    Arguments
    ---------
    linspace : array-like
        Normally a numpy.linspace of x coordinates corresponding to frequency
        in Hz.
    peaklist : numpy.array
        A 2D array of shape (N, 3) with (frequency, intensity, width) data.

    Returns
    -------
    numpy.array
        an array of y coordinates corresponding to intensity.
    """
    result = gauss(linspace, peaklist[0, 0], peaklist[0, 1], peaklist[0, 2])
    for i in range(1, len(peaklist)):
        result += gauss(
            linspace, peaklist[i, 0], peaklist[i, 1], peaklist[i, 2]
        )
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
    breaks = [index + 1 for index in range(len(spinSystemMatrix) - 1) if not np.any(spinSystemMatrix[:index + 1, index + 1:] != 0)]
    breaks = [0] + breaks + [len(spinSystemMatrix)]
    matrices = [spinSystemMatrix[breakpoint:breaks[index + 1], breakpoint:breaks[index + 1]] for index, breakpoint in enumerate(breaks[:-1])]
    # Suppress the printing from qm
    output = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    peaklist = [item for sublist in [qm_spinsystem([ssm[index][index] * frequency for index in range(len(ssm))], ssm) for ssm in matrices] for item in sublist]
    sys.stdout = output
    peaklist_out = [(peak[0] / frequency, peak[1], width) for peak in peaklist]
    # df_out = pd.DataFrame(peaklist_out, columns=['chemical_shift', 'height', 'width', 'multiplet_id'])
    return peaklist_out
