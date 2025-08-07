import math
import numbers
import sys
import os
import numpy as np
import pandas as pd

# from nmrsim.qm import qm_spinsystem
import numba as nb
import scipy


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
    return peaklist_array, points, l_limit, r_limit, function_type

    # return _createLineshape_numba(
    #     peaklist_array, points, l_limit, r_limit, function_type
    # )


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


def getMatrices(spinSystemMatrix):
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

    return matrices

def generatePeaklists(matrices_top, frequency, width):
    # Suppress printing from qm
    output = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Execute the three functions in sequence
    hamiltonians_and_tm = generate_hamiltonians_and_transition_moments(matrices_top, frequency)
    peaklists_raw_data = calculate_eigenvalues_and_intensities(matrices_top, hamiltonians_and_tm)
    peaklists_top = normalize_and_process_peaklists(peaklists_raw_data, frequency, width)

    sys.stdout = output
    return peaklists_top


def generate_hamiltonians_and_transition_moments(matrices_top, frequency):
    """Function 1: Generate Hamiltonians and transition moments for all matrices"""
    hamiltonians_and_tm = []
    
    for matrices_row in matrices_top:
        row_data = []
        for matrices in matrices_row:
            matrices_data = []
            for ssm in matrices:
                freqs = [ssm[index][index] * frequency for index in range(len(ssm))]
                nspins = len(freqs)
                
                # Function 1
                H = hamiltonian_sparse(freqs, ssm).todense()
                T = _tm_cache(nspins)
                
                matrices_data.append({
                    'H': H,
                    'T': T,
                    'nspins': nspins,
                    'freqs': freqs
                })
            row_data.append(matrices_data)
        hamiltonians_and_tm.append(row_data)
    
    return hamiltonians_and_tm


def calculate_eigenvalues_and_intensities(matrices_top, hamiltonians_and_tm):
    """Function 2: Eigenvalue decomposition and intensity calculations"""
    peaklists_raw_data = []
    
    for row_idx, matrices_row in enumerate(matrices_top):
        row_peaklists = []
        for col_idx, matrices in enumerate(matrices_row):
            for mat_idx, ssm in enumerate(matrices):
                data = hamiltonians_and_tm[row_idx][col_idx][mat_idx]
                H = data['H']
                T = data['T']
                cutoff = 0.001
                
                # Function 2
                E, V = scipy.linalg.eigh(H.todense())
                V = V.real
                I = np.square(V.T.dot(T.dot(V)))
                I_upper = np.triu(I)
                E_matrix = np.abs(E[:, np.newaxis] - E)
                E_upper = np.triu(E_matrix)
                combo = np.stack([E_upper, I_upper])
                iv = combo.reshape(2, I.shape[0] ** 2).T
                peaklist = iv[iv[:, 1] >= cutoff]
                
                row_peaklists.append({
                    'peaklist': peaklist,
                    'nspins': data['nspins']
                })
        peaklists_raw_data.append(row_peaklists)
    
    return peaklists_raw_data


def normalize_and_process_peaklists(peaklists_raw_data, frequency, width):
    """Function 3: Normalization and final processing"""
    peaklists_top = []
    
    for row_idx, row_peaklists in enumerate(peaklists_raw_data):
        peaklists_processed = []
        for peaklist_data in row_peaklists:
            peaklist = peaklist_data['peaklist']
            nspins = peaklist_data['nspins']
            
            # Function 3 - Normalization
            normalized_peaklist = normalize_peaklist(peaklist, nspins)
            
            # Final processing (same as original)
            peak_array = np.array(normalized_peaklist, dtype=np.float64)  # shape (N, 3)
            width_column = np.full((peak_array.shape[0], 1), width)
            peak_array = np.hstack((peak_array, width_column))

            # Normalize first column
            peak_array[:, 0] /= frequency

            # Replace third column by width
            peak_array[:, 2] = width

            peaklists_processed.append(peak_array)
        
        peaklists_top.append(peaklists_processed)
    
    return peaklists_top


##### NMRSIMM #####
import sys

import scipy.sparse

if sys.version_info >= (3, 7):
    from importlib import resources
else:
    import importlib_resources as resources

import numpy as np  # noqa: E402
import sparse  # noqa: E402

import nmrsim.bin  # noqa: E402
from nmrsim.math import normalize_peaklist  # noqa: E402

CACHE = True  # saving of partial solutions is allowed
SPARSE = True  # the sparse library is available


def secondorder_sparse(freqs, couplings, normalize=True, **kwargs):
    nspins = len(freqs)
    ###Function 1
    H = hamiltonian_sparse(freqs, couplings)
    T = _tm_cache(nspins)
    cutoff = 0.001

    ###Function 2
    E, V = scipy.linalg.eigh(H.todense())
    V = V.real
    I = np.square(V.T.dot(T.dot(V)))
    I_upper = np.triu(I)
    E_matrix = np.abs(E[:, np.newaxis] - E)
    E_upper = np.triu(E_matrix)
    combo = np.stack([E_upper, I_upper])
    iv = combo.reshape(2, I.shape[0] ** 2).T
    peaklist = iv[iv[:, 1] >= cutoff]
    ###

    ###Function 3
    if normalize:
        peaklist = normalize_peaklist(peaklist, nspins)
    return peaklist


def hamiltonian_sparse(v, J):
    """
    Calculate the spin Hamiltonian as a sparse array.

    Parameters
    ----------
    v : array-like
        list of frequencies in Hz (in the absence of splitting) for each
        nucleus.
    J : 2D array-like
        matrix of coupling constants. J[m, n] is the coupling constant between
        v[m] and v[n].

    Returns
    -------
    H : sparse.COO
        a sparse spin Hamiltonian.
    """
    nspins = len(v)
    Lz, Lproduct = _so_sparse(nspins)  # noqa
    # TODO: remove the following lines once tests pass
    assert isinstance(Lz, (sparse.COO, np.ndarray, scipy.sparse.spmatrix))
    # On large spin systems, converting v and J to sparse improved speed of
    # sparse.tensordot calls with them.
    # First make sure v and J are a numpy array (required by sparse.COO)
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if not isinstance(J, np.ndarray):
        J = np.array(J)
    H = sparse.tensordot(sparse.COO(v), Lz, axes=1)
    scalars = 0.5 * sparse.COO(J)
    H += sparse.tensordot(scalars, Lproduct, axes=2)
    return H

def _so_sparse(nspins):
    """
    Either load a presaved set of spin operators as numpy arrays, or
    calculate them and save them if a presaved set wasn't found.

    Parameters
    ----------
    nspins : int
        the number of spins in the spin system

    Returns
    -------
    (Lz, Lproduct) : a tuple of:
        Lz : 3d sparse.COO array of shape (n, 2^n, 2^n) representing
            [Lz1, Lz2, ...Lzn]
        Lproduct : 4d sparse.COO array of shape (n, n, 2^n, 2^n), representing
            an n x n array (cartesian product) for all combinations of
            Lxa*Lxb + Lya*Lyb + Lza*Lzb, where 1 <= a, b <= n.

    Side Effect
    -----------
    Saves the results as .npz files to the bin directory if they were not
    found there.
    """
    # TODO: once nmrsim demonstrates installing via the PyPI *test* server,
    # need to determine how the saved solutions will be handled. For example,
    # part of the final build may be generating these files then testing.
    # Also, need to consider different users with different system capabilities
    # (e.g. at extreme, Raspberry Pi). Some way to let user select, or select
    # for user?
    filename_Lz = f'Lz{nspins}.npz'
    filename_Lproduct = f'Lproduct{nspins}.npz'
    bin_path = _bin_path()
    path_Lz = bin_path.joinpath(filename_Lz)
    path_Lproduct = bin_path.joinpath(filename_Lproduct)
    # with path_context_Lz as p:
    #     path_Lz = p
    # with path_context_Lproduct as p:
    #     path_Lproduct = p
    try:
        Lz = sparse.load_npz(path_Lz)
        Lproduct = sparse.load_npz(path_Lproduct)
        return Lz, Lproduct
    except FileNotFoundError:
        print('no SO file ', path_Lz, ' found.')
        print(f'creating {filename_Lz} and {filename_Lproduct}')
    Lz, Lproduct = _so_dense(nspins)
    Lz_sparse = sparse.COO(Lz)
    Lproduct_sparse = sparse.COO(Lproduct)
    sparse.save_npz(path_Lz, Lz_sparse)
    sparse.save_npz(path_Lproduct, Lproduct_sparse)

    return Lz_sparse, Lproduct_sparse

def _compile_peaklist(I, E, cutoff=0.001):
    """
    Generate a peaklist from intensity and energy matrices.

    Parameters
    ----------
    I : numpy.ndarray (2D)
        matrix of relative intensities
    E : numpy.ndarray (1D)
        array of energies
    cutoff : float, optional
        The intensity cutoff for reporting signals.

    Returns
    -------
    numpy.ndarray (2D)
        A [[frequency, intensity]...] peaklist.
    """
    I_upper = np.triu(I)
    E_matrix = np.abs(E[:, np.newaxis] - E)
    E_upper = np.triu(E_matrix)
    combo = np.stack([E_upper, I_upper])
    iv = combo.reshape(2, I.shape[0] ** 2).T
    return iv[iv[:, 1] >= cutoff]


def _bin_path():
    """Return a Path to the nmrsim/bin directory."""
    init_path_context = resources.path(nmrsim.bin, '__init__.py')
    with init_path_context as p:
        init_path = p
    bin_path = init_path.parent
    return bin_path


def _tm_cache(nspins):
    """
    Loads a saved sparse transition matrix if it exists, or creates and saves
    one if it is not.

    Parameters
    ----------
    nspins : int
        The number of spins in the spin system.

    Returns
    -------
    T_sparse : sparse.COO
        The sparse transition matrix.

    Side Effects
    ------------
    Saves a sparse array to the bin folder if the required array was not
    found there.
    """
    # Speed tests indicated that using sparse-array transition matrices
    # provides a modest speed improvement on larger spin systems.
    filename = f'T{nspins}.npz'
    # init_path_context = resources.path(nmrsim.bin, '__init__.py')
    # with init_path_context as p:
    #     init_path = p
    # print('path to init: ', init_path)
    # bin_path = init_path.parent
    bin_path = _bin_path()
    path = bin_path.joinpath(filename)
    try:
        T_sparse = sparse.load_npz(path)
        return T_sparse
    except FileNotFoundError:
        print(f'creating {filename}')
        T_sparse = _transition_matrix_dense(nspins)
        T_sparse = sparse.COO(T_sparse)
        print('_tm_cache will save on path: ', path)
        sparse.save_npz(path, T_sparse)
        return T_sparse
