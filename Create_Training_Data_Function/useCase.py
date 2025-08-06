from pathlib import Path
from createTrainingData import createTrainingData
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import cProfile
import pstats

# Name to Spectrum ID Dictionary.
# # We can expand this to any of the GISSMO simulated spectra. All the data is in the Casmdb_Data directory.
substanceDict = {
    'L-Alanine': ['SP:3208'],
    'L-Arginine': ['SP:3212', 'SP:3285', 'SP:3360', 'SP:3388'],
    'L-Asparagine': ['SP:3408', 'SP:3597'],
    'L-Aspartic_Acid': ['SP:3526', 'SP:3603'],
    'L-Cysteine': ['SP:3723', 'SP:3725'],
    'L-Glutamic_Acid': ['SP:3412', 'SP:3690'],
    'L-Glutamine': ['SP:3108'],
    'L-Histidine': ['SP:3099', 'SP:3684'],
    'L-Isoleucine': ['SP:3390', 'SP:3502'],
    'L-Leucine': ['SP:3551'],
    'L-Lysine': ['SP:3506', 'SP:3560'],
    'L-Methionine': ['SP:3456', 'SP:3509'],
    'L-Proline': ['SP:3140', 'SP:3406'],
    'L-Phenylalanine': ['SP:3326', 'SP:3462', 'SP:3507'],
    'L-Serine': ['SP:3324', 'SP:3427', 'SP:3732'],
    'L-Threonine': ['SP:3327', 'SP:3437'],
    'L-Tryptophan': ['SP:3342', 'SP:3455'],
    'L-Tyrosine': ['SP:3464'],
    'L-Valine': ['SP:3413', 'SP:3490'],
    'Glycine': ['SP:3365', 'SP:3682'],
    }

# # Amino acid list. This is just a list of names for convenience.
# aminoAcids = ['L-Alanine', 'L-Arginine', 'L-Asparagine', 'L-Aspartic_Acid', 'L-Cysteine', 'L-Glutamic_Acid', 'L-Glutamine', 'L-Histidine', 'L-Isoleucine', 'L-Leucine', 'L-Lysine', 'L-Methionine', 'L-Proline', 'L-Phenylalanine', 'L-Serine', 'L-Threonine', 'L-Tryptophan', 'L-Tyrosine', 'L-Valine', 'Glycine']

# # Collect the IDs from the substanceDict. We use the last id as this is collected at the highest Spectrometer Frequency and is probably the best resolution.
# substanceSpectrumIds = [substanceDict[aminoAcid][-1] for aminoAcid in aminoAcids]

# Citric Acid: SP:3368

# substanceDict = {
#     'Citric acid': ['SP:3368'],
# }

substanceSpectrumIds = [substanceDict[substance][-1] for substance in substanceDict]

# The function for actually creating the simulations. See createTrainingData.py for details on the function.
sampleNumber = 10

# cProfile.run('createTrainingData(substanceSpectrumIds=substanceSpectrumIds, sampleNumber=sampleNumber, scale=0.5)', 'output.prof')

with open('results.txt', 'w') as f:
    stats = pstats.Stats('output.prof', stream=f)
    stats.sort_stats('cumulative')
    stats.print_stats()

print(positions.shape)
print(intensities.shape)


# Save data to a new timestamped directory in the current working directory.
cwd = Path(os.path.dirname(os.path.realpath(__file__)))
outDir = cwd.joinpath(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if not os.path.isdir(outDir):
    os.makedirs(outDir)
scales.to_csv(outDir.joinpath('scales.npy'), sep=',', index=True, encoding='utf-8')
# Save as .npy files
np.save(outDir.joinpath('positions.npy'), positions)
np.save(outDir.joinpath('intensities.npy'), intensities)
np.save(outDir.joinpath('components.npy'), components)

#Save as transposed csv files
np.savetxt(outDir.joinpath('positions.csv'), np.transpose(positions), delimiter=',', header='Positions')
np.savetxt(outDir.joinpath('intensities.csv'), np.transpose(intensities), delimiter=',', header=','.join([f'Sample_{i}' for i in range(sampleNumber)]))
np.savetxt(outDir.joinpath('components.csv'), np.transpose(components), delimiter=',', header=','.join([str(substance) for substance in substanceDict]))

# Quick plot to see an example sample.
plt.plot(positions, intensities[0])
plt.xlim(max(positions), min(positions))
plt.savefig((outDir.joinpath('sampleplot.png')))
# Quick plot to see an example component.
plt.plot(positions, components[0])
plt.xlim(max(positions), min(positions))
plt.savefig((outDir.joinpath('componentplot.png')))
