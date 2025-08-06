import math
import numbers
import sys
import os
import numpy as np
import pandas as pd

# from nmrsim.qm import qm_spinsystem
import numba as nb
import scipy


@nb.njit
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
    # Convert peaklist to numpy array and sort by frequency
    peaklist_array = np.array(peaklist)
    if len(peaklist_array) > 0:
        # Sort by frequency (first column)
        sort_indices = np.argsort(peaklist_array[:, 0])
        peaklist_array = peaklist_array[sort_indices]

    # Handle limits
    if limits:
        l_limit, r_limit = validate_and_sort_limits(limits)
    else:
        if len(peaklist_array) > 0:
            l_limit = peaklist_array[0, 0] - 0.5
            r_limit = peaklist_array[-1, 0] + 0.5
        else:
            l_limit = -0.5
            r_limit = 0.5

    # Convert function string to integer
    function_type = 0 if function == 'lorentzian' else 1

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

    peak_array = np.array(peaklist, dtype=np.float64)  # shape (N, 3)
    width_column = np.full((peak_array.shape[0], 1), width)
    peak_array = np.hstack((peak_array, width_column))

    # Divide first column by frequency
    peak_array[:, 0] = peak_array[:, 0] / frequency

    # Replace third column by width
    peak_array[:, 2] = width

    peaklist_out = peak_array
    # df_out = pd.DataFrame(peaklist_out, columns=['chemical_shift', 'height', 'width', 'multiplet_id'])
    return peaklist_out


"""qm contains functions for the quantum-mechanical (second-order)
calculation of NMR spectra.

The qm module provides the following attributes:

* CACHE : bool (default True)
    Whether saving to disk of partial solutions is allowed.
* SPARSE : bool (default True)
    Whether the sparse library can be used.

The qm module provides the following functions:

* qm_spinsystem: The high-level function for computing a second-order
  simulation from frequency and J-coupling data.
* hamiltonian_dense: Calculate a spin Hamiltonian using dense arrays
  (slower).
* hamiltonian_sparse: Calculate a spin Hamiltonian using cached sparse arrays
  (faster).
* solve_hamiltonian: Calculate a peaklist from a spin Hamiltonian.
* secondorder_dense: Calculate a peaklist for a second-order spin system,
  using dense arrays (slower).
* secondorder_sparse: Calculate a peaklist for a second-order spin system,
  using cached sparse arrays (faster).

Notes
-----
Because numpy.matrix is marked as deprecated, starting with Version 0.2.0 the
qm code was refactored to a) accommodate this deprecation and b) speed up the
calculations. The fastest calculations rely on:

1. the pydata/sparse library. SciPy's sparse depends on numpy.matrix,
and they currently recommend that pydata/sparse be used for now.

2. Caching partial solutions for spin operators and transition matrices as
.npz files.

If the pydata/sparse package is no longer available, and/or if distributing
the library with .npz files via PyPI is problematic, then a backup is
required. The qm module for now provides two sets of functions for
calculating second-order spectra: one using pydata/sparse and caching,
and the other using neither.
"""
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


def _bin_path():
    """Return a Path to the nmrsim/bin directory."""
    init_path_context = resources.path(nmrsim.bin, '__init__.py')
    with init_path_context as p:
        init_path = p
    bin_path = init_path.parent
    return bin_path


def _so_dense(nspins):
    """
    Calculate spin operators required for constructing the spin hamiltonian,
    using dense (numpy) arrays.

    Parameters
    ----------
    nspins : int
        The number of spins in the spin system.

    Returns
    -------
    (Lz, Lproduct) : a tuple of:
        Lz : 3d array of shape (n, 2^n, 2^n) representing [Lz1, Lz2, ...Lzn]
        Lproduct : 4d array of shape (n, n, 2^n, 2^n), representing an n x n
            array (cartesian product) for all combinations of
            Lxa*Lxb + Lya*Lyb + Lza*Lzb, where 1 <= a, b <= n.
    """
    sigma_x = np.array([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.array([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.array([[1 / 2, 0], [0, -1 / 2]])
    unit = np.array([[1, 0], [0, 1]])

    L = np.empty(
        (3, nspins, 2**nspins, 2**nspins), dtype=np.complex128
    )  # TODO: consider other dtype?
    for n in range(nspins):
        Lx_current = 1
        Ly_current = 1
        Lz_current = 1

        for k in range(nspins):
            if k == n:
                Lx_current = np.kron(Lx_current, sigma_x)
                Ly_current = np.kron(Ly_current, sigma_y)
                Lz_current = np.kron(Lz_current, sigma_z)
            else:
                Lx_current = np.kron(Lx_current, unit)
                Ly_current = np.kron(Ly_current, unit)
                Lz_current = np.kron(Lz_current, unit)

        L[0][n] = Lx_current
        L[1][n] = Ly_current
        L[2][n] = Lz_current

    # ref:
    # https://stackoverflow.com/questions/47752324/matrix-multiplication-on-4d-numpy-arrays
    L_T = L.transpose(1, 0, 2, 3)
    Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)

    return L[2], Lproduct


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


def hamiltonian_dense(v, J):
    """
    Calculate the spin Hamiltonian as a dense array.

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
    H : numpy.ndarray
        a sparse spin Hamiltonian.
    """
    nspins = len(v)
    Lz, Lproduct = _so_dense(nspins)  # noqa
    H = np.tensordot(v, Lz, axes=1)
    if not isinstance(J, np.ndarray):
        J = np.array(J)
    scalars = 0.5 * J
    H += np.tensordot(scalars, Lproduct, axes=2)
    return H


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
    print('From hamiltonian_sparse:')
    print('Lz is type: ', type(Lz))
    print('Lproduct is type: ', type(Lproduct))
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


def _transition_matrix_dense(nspins):
    """
    Creates a matrix of allowed transitions, as a dense array.

    The integers 0-`n`, in their binary form, code for a spin state
    (alpha/beta). The (i,j) cells in the matrix indicate whether a transition
    from spin state i to spin state j is allowed or forbidden.
    See the ``is_allowed`` function for more information.

    Parameters
    ---------
    nspins : number of spins in the system.

    Returns
    -------
    numpy.ndarray
        a transition matrix that can be used to compute the intensity of
    allowed transitions.

    Notes
    -----
    The integers 0-`n`, in their binary form, code for a pure spin state
    (alpha/beta). For example, for a three-spin system:
    0 = 000 = alpha-alpha-alpha,
    1 = 001 = alpha-alpha-beta,
    ⋮
    7 = 111 = beta-beta-beta.
    A transition between two of these states is allowed if only one spin flips.
    This is equal to a single bit change in the binary representation of the
    index.

    The (i,j) cells in the transition matrix indicate whether a transition
    from spin state i to spin state j is allowed or forbidden (1 = allowed,
    0 = forbidden).
    """
    # function was optimized by only calculating upper triangle and then adding
    # the lower.
    n = 2**nspins
    T = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if bin(i ^ j).count('1') == 1:
                T[i, j] = 1
    T += T.T
    return T


def secondorder_dense(freqs, couplings, normalize=True, **kwargs):
    """
    Calculates second-order spectral data (freqency and intensity of signals)
    for *n* spin-half nuclei.

    Parameters
    ---------
    freqs : [float...]
        a list of *n* nuclei frequencies in Hz
    couplings : array-like
        an *n, n* array of couplings in Hz. The order
        of nuclei in the list corresponds to the column and row order in the
        matrix, e.g. couplings[0][1] and [1]0] are the J coupling between
        the nuclei of freqs[0] and freqs[1].
    normalize: bool
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    peaklist : [[float, float]...]
        numpy 2D array of [frequency, intensity] pairs.

    Other Parameters
    ----------------
    cutoff : float
        The intensity cutoff for reporting signals (default is 0.001).
    """
    nspins = len(freqs)
    H = hamiltonian_dense(freqs, couplings)
    E, V = scipy.linalg.eigh(H, driver='evd')
    V = V.real
    T = _transition_matrix_dense(nspins)
    I = np.square(V.T.dot(T.dot(V)))
    peaklist = _compile_peaklist(I, E, **kwargs)
    if normalize:
        peaklist = normalize_peaklist(peaklist, nspins)
    return peaklist


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


def _intensity_and_energy(H, nspins):
    """
    Calculate intensity matrix and energies (eigenvalues) from Hamiltonian.

    Parameters
    ----------
    H :  numpy.ndarray
        Spin Hamiltonian
    nspins : int
        number of spins in spin system

    Returns
    -------
    (I, E) : (numpy.ndarray, numpy.ndarray) tuple of:
        I : (relative) intensity 2D array
        V : 1-D array of relative energies.
    """
    E, V = scipy.linalg.eigh(H)
    V = V.real
    T = _tm_cache(nspins)
    I = np.square(V.T.dot(T.dot(V)))
    return I, E


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


def solve_hamiltonian(H, nspins, **kwargs):
    """
    Calculates frequencies and intensities of signals from a spin Hamiltonian
    and number of spins.

    Parameters
    ----------
    H : numpy.ndarray (2D)
        The spin Hamiltonian
    nspins : int
        The number of spins in the system

    Returns
    -------
    [[float, float]...] numpy 2D array of frequency, intensity pairs.

    Other Parameters
    ----------------
    cutoff : float
        The intensity cutoff for reporting signals (default is 0.001).
    """
    I, E = _intensity_and_energy(H, nspins)
    return _compile_peaklist(I, E, **kwargs)


def secondorder_sparse(freqs, couplings, normalize=True, **kwargs):
    """
    Calculates second-order spectral data (frequency and intensity of signals)
    for *n* spin-half nuclei.

    Parameters
    ---------
    freqs : [float...]
        a list of *n* nuclei frequencies in Hz
    couplings : array-like
        an *n, n* array of couplings in Hz. The order
        of nuclei in the list corresponds to the column and row order in the
        matrix, e.g. couplings[0][1] and [1]0] are the J coupling between
        the nuclei of freqs[0] and freqs[1].
    normalize: bool
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    peaklist : [[float, float]...] numpy 2D array
        of [frequency, intensity] pairs.

    Other Parameters
    ----------------
    cutoff : float
        The intensity cutoff for reporting signals (default is 0.001).
    """
    nspins = len(freqs)
    H = hamiltonian_sparse(freqs, couplings)
    peaklist = solve_hamiltonian(H.todense(), nspins, **kwargs)
    if normalize:
        peaklist = normalize_peaklist(peaklist, nspins)
    return peaklist


def qm_spinsystem(*args, cache=CACHE, sparse=SPARSE, **kwargs):
    """
    Calculates second-order spectral data (frequency and intensity of signals)
    for *n* spin-half nuclei.

    Currently, n is capped at 11 spins.

    Parameters
    ----------
    freqs : [float...]
        a list of *n* nuclei frequencies in Hz.
    couplings : array-like
        An *n, n* array of couplings in Hz. The order of nuclei in the list
        corresponds to the column and row order in the matrix, e.g.
        couplings[0][1] and [1]0] are the J coupling between the nuclei of
        freqs[0] and freqs[1].
    normalize: bool (optional keyword argument; default = True)
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    peaklist : [[float, float]...] numpy 2D array
        of [frequency, intensity] pairs.

    Other Parameters
    ----------------
    cache: bool (default = nmrsim.qm.CACHE)
        Whether caching of partial solutions (for acceleration) is allowed.
        Currently CACHE = True, but this provides a hook to modify nmrsim for
        platforms such as Raspberry Pi where storage space is a concern.
    sparse: bool (default = nmrsim.qm.SPARSE)
        Whether the pydata sparse library for sparse matrices is available.
        Currently SPARSE = True, but this provides a hook to modify nmrsim
        should the sparse library become unavailable (see notes).
    cutoff : float
        The intensity cutoff for reporting signals (default is 0.001).

    Notes
    -----
    With numpy.matrix marked for deprecation, the scipy sparse array
    functionality is on shaky ground, and the current recommendation is to use
    the pydata sparse library. In case a problem arises in the numpy/scipy/
    sparse ecosystem, SPARSE provides a hook to use a non-sparse-dependent
    alternative.
    """
    if not (cache and sparse):
        return secondorder_dense(*args, **kwargs)
    return secondorder_sparse(*args, **kwargs)
