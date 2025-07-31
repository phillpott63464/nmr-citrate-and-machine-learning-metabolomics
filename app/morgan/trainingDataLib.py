import math
import numbers
import sys
import os
import numpy as np
import pandas as pd
from nmrsim.qm import qm_spinsystem


def createLineshape(
    peaklist, points=65536, limits=None, function='lorentzian'
):
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
    peaklist.sort()
    if limits:
        l_limit, r_limit = low_high(limits)
    else:
        l_limit = peaklist[0][0] - 0.5
        r_limit = peaklist[-1][0] + 0.5
    x_inv = np.linspace(l_limit, r_limit, points)
    x = x_inv[::-1]  # reverses the x axis
    if len(peaklist) > 0:
        if function == 'lorentzian':
            y = add_lorentzians(x, peaklist)
        elif function == 'gaussian':
            y = add_gaussians(x, peaklist)
    else:
        y = np.zeros(points)
    return x, y


def is_number(n):
    if isinstance(n, numbers.Real):
        return n
    else:
        raise TypeError('Must be a real number.')


def is_tuple_of_two_numbers(t):
    m, n = t
    return is_number(m), is_number(n)


def low_high(t):
    two_numbers = is_tuple_of_two_numbers(t)
    return min(two_numbers), max(two_numbers)


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
    result = lorentz(linspace, peaklist[0][0], peaklist[0][1], peaklist[0][2])
    for v, i, w in peaklist[1:]:
        result += lorentz(linspace, v, i, w)
    return result


def gauss(v, v0, I, w):
    wf = 0.4246609
    return I * (math.e ** ((-((v - v0) ** 2)) / (2 * ((w * wf) ** 2))))


def add_gaussians(linspace, peaklist, w):
    result = gauss(linspace, peaklist[0][0], peaklist[0][1], w)
    for v, i, w in peaklist[1:]:
        result += gauss(linspace, v, i, w)
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
