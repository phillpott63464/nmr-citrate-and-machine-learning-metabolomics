import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from trainingDataLib import createLineshape, peakListFromSpinSystemMatrix

referenceData = pd.DataFrame(
    {
        'multiplets': [
            pd.DataFrame(
                {
                    'center': 0.0,
                    'scale': 1.0,
                    'ssm_indices': [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
                }
            ),
            None,
            None,
        ],
        'ssm': [np.zeros([9, 9]), None, None],
        'offset': [0.0, 0.0, 0.0],
    },
    index=['tsp', 'dss', 'tms'],
)


"""Establish the simulation data sources"""
dirPath = Path(os.path.dirname(os.path.realpath(__file__)))
metabolites = pd.read_csv(
    dirPath.joinpath('Casmdb_Data/metabolites.csv'), index_col=0
)
samples = pd.read_csv(dirPath.joinpath('Casmdb_Data/samples.csv'), index_col=0)
spectra = pd.read_csv(dirPath.joinpath('Casmdb_Data/spectra.csv'), index_col=0)
multiplets = pd.read_csv(
    dirPath.joinpath('Casmdb_Data/multiplets.csv'), index_col=0
)
couplings = pd.read_csv(
    dirPath.joinpath('Casmdb_Data/couplings.csv'), index_col=0
)


def createTrainingData(
    substanceSpectrumIds=[],
    sampleNumber=1,
    frequency=600.00,
    points=2**15,
    limits=(10, -2),
    highLimits=(9, 11),
    lowLimits=(-1, -3),
    peakWidth=1.5,
    scale=1.0,
    rondomlyScaleSubstances=True,
    includeNoise=True,
    noiseRange=(-2, -3),
    referenceSubstanceSpectrumId='tsp',
    randomlyOffsetMultiplets=True,
    multipletOffsetCap=0.01,
):
    """
    Training data and annotations for a ML model.
    ----------
    substanceSpectrumIds : [str...]
        A list of strings for collecting metabolite sims from the database.
    sampleNumber : int
        Number of sample spectra to be simulated.
    frequency : float or int
        Spectrometer frequency to simulate at.
    points : int
        Number of data points.
    limits : (float, float)
        Frequency limits for the plot.
    peakWidth: float or int
        Peak widths in Hz.
    scale: float or int
        Global scaling factor for all the simulated sample spectra.
    rondomlyScaleSubstances: bool
        Flag to determine whether to randomise component metabolite spectra scales.
    includeNoise: bool
        Flag to determine whether to add an array of noise to the sample simulations.
    noiseRange: (float or int, float or int)
        A pair of values for the upper and lower powers for the noise range.
    referenceSubstanceSpectrumId: str
        ID of the reference compound.
    randomlyOffsetMultiplets: bool
        Flag to determine whether to apply a random offset to the multiplets of the component metabolite spectra.
    multipletOffsetCap: float or int
         The absolute value for the maximum multiplet offset in the component metabolite spectra.
    Returns
    -------
    scales, positions, intensities, components: pandas.DataFrame, numpy.array, numpy.array, numpy.array
        Data on the scales (aka true values of the components), position array of all spectra (x-axis), intensity arrays of the simulated samples (y-axes), intensity arrays of the simulated components before any transformations (y-axes).
    """
    scalesDict = {
        ssid: [
            scale * (10 ** random.uniform(0.0, 1.0))
            if rondomlyScaleSubstances
            else scale
            for samNum in range(sampleNumber)
        ]
        for ssid in substanceSpectrumIds + [referenceSubstanceSpectrumId]
    }
    untransformedComponentsList = []
    transformedComponentsList = []
    intensitiesList = []
    peakWidth = peakWidth / frequency
    for sampleNumber in tqdm(range(sampleNumber)):
        # Make the reference signal first
        positions, y = generateSignal(
            referenceData.loc[referenceSubstanceSpectrumId, 'ssm'],
            peakWidth,
            frequency,
            points,
            limits,
            scalesDict[referenceSubstanceSpectrumId][sampleNumber],
        )
        for spectrumId in substanceSpectrumIds:
            if sampleNumber == 0:
                ssm = getSsmData(
                    spectrumId=spectrumId,
                    referenceOffset=referenceData.loc[
                        referenceSubstanceSpectrumId, 'offset'
                    ],
                    transform=False,
                    multipletOffsetCap=multipletOffsetCap,
                )
                x, substanceY = generateSignal(
                    ssm,
                    peakWidth,
                    frequency,
                    points,
                    limits,
                    scalesDict[spectrumId][sampleNumber],
                )
                untransformedComponentsList.append(substanceY)
            ssm = getSsmData(
                spectrumId=spectrumId,
                referenceOffset=referenceData.loc[
                    referenceSubstanceSpectrumId, 'offset'
                ],
                transform=randomlyOffsetMultiplets,
                multipletOffsetCap=multipletOffsetCap,
            )
            x, substanceY = generateSignal(
                ssm,
                peakWidth,
                frequency,
                points,
                limits,
                scalesDict[spectrumId][sampleNumber],
            )
            # transformedComponentsList.append(substanceY)
            y += substanceY
        if includeNoise:
            noise = 10 ** random.uniform(noiseRange[0], noiseRange[-1])
            y += np.random.normal(y, scale * noise)
        intensitiesList.append(y)
    return {
        'scales': pd.DataFrame(scalesDict),
        'positions': positions,
        'intensities': np.vstack(intensitiesList),
        'components': np.vstack(untransformedComponentsList),
    }


def generateSignal(ssm, peakWidth, frequency, points, limits, scale):
    peakList = peakListFromSpinSystemMatrix(ssm, frequency, peakWidth)
    x, y = createLineshape(peakList, points=points, limits=limits)
    return x, y * scale


def getSsmData(spectrumId, referenceOffset, transform, multipletOffsetCap):
    multipletsData = multiplets.loc[multiplets['spectrum_id'] == spectrumId]
    couplingsData = couplings.loc[couplings['spectrum_id'] == spectrumId]
    ssm = np.zeros((len(multipletsData), len(multipletsData)))
    for index, row in couplingsData.iterrows():
        ssm[
            multipletsData.index.get_loc(row['multiplet_id1']),
            multipletsData.index.get_loc(row['multiplet_id2']),
        ] = float(row['j'])
    ssm = ssm + ssm.T
    for index, center in enumerate(multipletsData['center']):
        chemShift = (
            float(center)
            + referenceOffset
            + random.uniform(-multipletOffsetCap, multipletOffsetCap)
            if transform
            else float(center) + referenceOffset
        )
        ssm[index, index] = float(chemShift)
    return ssm


def createSpinSystemMatrix(splitting_data, multiplet_data, frequency):
    shifts = [float(shift) * frequency for shift in multiplet_data.center]
    dimension = len(multiplet_data.index)
    ss_matrix = np.zeros((dimension, dimension))
    for cp_index, line in splitting_data.iterrows():
        j = float(line.j)
        x = int(line.multiplet_id1.split('.')[-1]) - 1
        y = int(line.multiplet_id2.split('.')[-1]) - 1
        # x = multiplet_data.index[multiplet_data['multiplet_id'] == line.multiplet_id1].tolist()[0]
        # y = multiplet_data.index[multiplet_data['multiplet_id'] == line.multiplet_id2].tolist()[0]
        ss_matrix[x, y] = j
    ss_matrix = ss_matrix + ss_matrix.T
    # if len(ss_matrix) > 11:
    #     print('spin system matrix is too large')
    #     return None
    for index, shift in enumerate(shifts):
        ss_matrix[index][index] = round(shift / frequency, 4)
    return ss_matrix
