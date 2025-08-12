import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    torch.set_float32_matmul_precision('high')

    # Check hardware capabilities for GPU acceleration
    hip_version = torch.version.hip
    cuda_built = torch.backends.cuda.is_built()
    gpu_count = torch.cuda.device_count()

    return cuda_built, gpu_count, hip_version, mo, torch


@app.cell(hide_code=True)
def _(cuda_built, gpu_count, hip_version, mo):
    mo.md(
        f"""
    ## PyTorch GPU Setup Information

    - **HIP Version:** {hip_version}
    - **CUDA Built:** {cuda_built}
    - **GPU Device Count:** {gpu_count}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# NMR Sim Concentration Model""")
    return


@app.cell
def _():
    # global variables

    count = 100
    trials = 1000
    combo_number = 30
    notebook_name = 'randomisation_hold_back'
    MODEL_TYPE = 'mlp'  # or 'mlp' or 'transformer'
    downsample = None  # Target resolution for ML model
    reverse = False
    cache_dir = f'./data_cache/{notebook_name}/{MODEL_TYPE}/{"time" if reverse else "freq"}/{downsample}'

    # Define metabolites and their spectrum IDs for NMR simulation
    substanceDict = {
        'Citric acid': ['SP:3368'],
        'Succinic acid': ['SP:3211'],
        'Maleic acid': ['SP:3110'],
        'Lactic acid': ['SP:3675'],
        'L-Methionine': ['SP:3509'],
        'L-Proline': ['SP:3406'],
        'L-Phenylalanine': ['SP:3507'],
        'L-Serine': ['SP:3732'],
        'L-Threonine': ['SP:3437'],
        'L-Tryptophan': ['SP:3455'],
        'L-Tyrosine': ['SP:3464'],
        'L-Valine': ['SP:3490'],
        'Glycine': ['SP:3682'],
    }
    return (
        MODEL_TYPE,
        cache_dir,
        combo_number,
        count,
        downsample,
        reverse,
        substanceDict,
        trials,
    )


@app.cell(hide_code=True)
def _(mo, spectrafigures, substanceDict):
    mo.md(
        rf"""
    ## Data Creation

    Create spectra using morgan's code overlaying nmrsim

    Contains metabolites: {''.join(f"{x}, " for x in substanceDict)}

    {mo.as_html(spectrafigures)}
    """
    )
    return


@app.cell
def _(cache_dir, combo_number, count, substanceDict):
    from morgan.createTrainingData import createTrainingData
    import morgan
    import numpy as np
    import tqdm
    import itertools
    import os
    import pickle
    from pathlib import Path
    import random

    substances = list(substanceDict.keys())

    # Save/load functions for data persistence
    def save_spectra_data(
        spectra, held_back_metabolites, combinations, filename
    ):
        """Save generated spectra data, held-back metabolite, and combinations to pickle file"""
        os.makedirs(cache_dir, exist_ok=True)
        filepath = f'{cache_dir}/{filename}.pkl'

        data_to_save = {
            'spectra': spectra,
            'held_back_metabolites': held_back_metabolites,
            'combinations': combinations,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(
            f"Saved {len(spectra)} spectra, held-back metabolites '{[x for x in held_back_metabolites]}', and {len(combinations)} combinations to {filepath}"
        )

    def load_spectra_data(filename, substanceDict):
        """Load generated spectra data, held-back metabolite, and combinations from pickle file"""
        filepath = f'{cache_dir}/{filename}.pkl'

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Handle both old and new cache formats
            if isinstance(data, list):
                # Old format - just spectra
                print(
                    f'Loaded {len(data)} spectra from {filepath} (old format)'
                )
                return data, None, None
            elif 'combinations' not in data:
                # Medium format - spectra + held_back_metabolites
                spectra = data['spectra']
                try:
                    held_back_metabolites = data['held_back_metabolites']
                except:
                    held_back_metabolites = data['held_back_metabolite']

                # Check if held_back_metabolites is a string and convert to a list
                if isinstance(held_back_metabolites, str):
                    held_back_metabolites = [
                        held_back_metabolites,
                        random.choice(list(substanceDict.keys())),
                    ]

                print(
                    f"Loaded {len(spectra)} spectra and held-back metabolite '{held_back_metabolites}' from {filepath} (medium format)"
                )
                return spectra, held_back_metabolites, None
            else:
                # New format - spectra + held_back_metabolites + combinations
                spectra = data['spectra']
                try:
                    held_back_metabolites = data['held_back_metabolites']
                except:
                    held_back_metabolites = data['held_back_metabolite']
                combinations = data['combinations']

                # Check if held_back_metabolites is a string and convert to a list
                if isinstance(held_back_metabolites, str):
                    held_back_metabolites = [
                        held_back_metabolites,
                        random.choice(list(substanceDict.keys())),
                    ]

                print(
                    f"Loaded {len(spectra)} spectra, held-back metabolite '{held_back_metabolites}', and {len(combinations)} combinations from {filepath}"
                )
                return spectra, held_back_metabolites, combinations
        return None, None, None

    def generate_cache_key(substanceDict, combo_number, count):
        """Generate unique cache key based on parameters"""
        substance_key = '_'.join(sorted(substanceDict.keys()))
        combo_key = f'combos_{combo_number}'
        count_key = f'count_{count}'
        return f'spectra_{substance_key}_{combo_key}_{count_key}'

    def create_batch_data(substances_and_count):
        """Generate training data batch for specific substance combination with random scaling"""
        substances, sample_count = substances_and_count
        return createTrainingData(
            substanceSpectrumIds=substances,
            sampleNumber=sample_count,
            rondomlyScaleSubstances=True,  # Randomize concentrations for training diversity
            referenceSubstanceSpectrumId='tsp',
        )

    def check_loaded_data(spectra, held_back_metabolites, combinations):
        if spectra is None:
            print('No cached data found. Generating new spectra...')

            # Generate all possible combinations of substances (4 to n substances)
            # This creates training data for different metabolite mixtures
            all_combinations = []
            for r in range(len(substances) // 3, len(substances) + 1):
                for combo in itertools.combinations(substances, r):
                    combo_dict = {
                        substance: substanceDict[substance]
                        for substance in combo
                    }
                    all_combinations.append(combo_dict)

            if combo_number is not None:
                combinations = random.sample(all_combinations, combo_number)
            print(f'Generated {len(combinations)} random combinations')

            # Select random metabolite to hold back for testing
            held_back_metabolites = random.sample(list(substanceDict.keys()), 2)
            print(
                f"Selected '{held_back_metabolites}' as held-back metabolite for testing"
            )

            # Extract spectrum IDs for each combination
            substanceSpectrumIds = [
                [combination[substance][-1] for substance in combination]
                for combination in combinations
            ]

            # Prepare arguments for sequential processing
            mp_args = [
                (substances, count) for substances in substanceSpectrumIds
            ]

            # Sequential data generation with progress bar
            print(f'Generating data sequentially...')

            batch_data = []
            for arg in tqdm.tqdm(mp_args, desc="Generating batches"):
                batch_data.append(create_batch_data(arg))

            print(f'Generated {len(batch_data)} batches')

            # Reshape batch data into individual spectrum samples
            # Each spectrum contains intensities, positions, scales, and component information
            spectra = []

            for batch in batch_data:
                for i in range(count):
                    # Extract individual sample scales (concentrations) from batch
                    sample_scales = {
                        key: [values[i]]
                        for key, values in batch['scales'].items()
                    }

                    # Create individual spectrum dictionary
                    spectrum = {
                        'scales': sample_scales,
                        'intensities': batch['intensities'][
                            i : i + 1
                        ],  # Keep 2D structure
                        'positions': batch[
                            'positions'
                        ],  # Chemical shift positions (ppm)
                        'components': batch[
                            'components'
                        ],  # Individual component spectra
                    }
                    spectra.append(spectrum)

            # Save generated data for future use
            save_spectra_data(
                spectra, held_back_metabolites, combinations, cache_key
            )
        else:
            print('Using cached spectra data.')
            if held_back_metabolites is None:
                # If old cache format, select and save new held-back metabolite
                held_back_metabolites = random.choice(
                    list(substanceDict.keys())
                )
                print(
                    f"Cache missing held-back metabolite. Selected '{held_back_metabolites}' and updating cache..."
                )
                save_spectra_data(
                    spectra, held_back_metabolites, combinations, cache_key
                )
            print(f'Using {len(combinations)} combinations from cache')

        return spectra, held_back_metabolites, combinations

    # Generate cache key for current configuration
    cache_key = generate_cache_key(substanceDict, combo_number, count)

    # Try to load existing data first
    spectra, held_back_metabolites, combinations = load_spectra_data(
        cache_key, substanceDict
    )
    spectra, held_back_metabolites, combinations = check_loaded_data(
        spectra, held_back_metabolites, combinations
    )

    # Extract spectrum IDs for each combination (needed for later processing)
    substanceSpectrumIds = [
        [combination[substance][-1] for substance in combination]
        for combination in combinations
    ]

    print(f'Total combinations: {len(combinations)}')

    # Display sample information for verification
    print('Sample scales preview:')
    print(''.join(f"{x['scales']}\n" for x in spectra[:5]))
    print(f"Intensities shape: {spectra[0]['intensities'].shape}")
    print(f"Positions shape: {spectra[0]['positions'].shape}")
    print(f"Components shape: {spectra[0]['components'].shape}")

    # Prepare data for markdown display
    sample_scales_preview = '\n'.join(
        [
            f"Sample {i+1}: {spectrum['scales']}"
            for i, spectrum in enumerate(spectra[:5])
        ]
    )
    intensities_shape = spectra[0]['intensities'].shape
    positions_shape = spectra[0]['positions'].shape
    components_shape = spectra[0]['components'].shape

    return (
        combinations,
        components_shape,
        createTrainingData,
        held_back_metabolites,
        intensities_shape,
        np,
        positions_shape,
        random,
        sample_scales_preview,
        spectra,
        tqdm,
    )


@app.cell(hide_code=True)
def _(
    combinations,
    components_shape,
    count,
    intensities_shape,
    mo,
    positions_shape,
    sample_scales_preview,
):
    mo.md(
        rf"""
    ## Data Generation Results

    **Generated {count} samples of {len(combinations)} combinations successfully**

    **Data Structure:**

    - **Intensities Shape:** {intensities_shape} (Y axis)

    - **Positions Shape:** {positions_shape} (X axis) 

    - **Components Shape:** {components_shape} (Peaks of all separate components)

    **Sample Scales Preview (First 5 samples):**
    ```
    {sample_scales_preview}
    ```
    """
    )
    return


@app.cell
def _(spectra):
    import matplotlib.pyplot as plt

    print(len(spectra))
    graph_count = 3

    # Create visualization grid showing sample spectra
    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for graphcounter in range(1, graph_count**2 + 1):
        plt.subplot(graph_count, graph_count, graphcounter)
        plt.plot(
            spectra[graphcounter]['positions'],
            spectra[graphcounter]['intensities'][0],
        )

    spectrafigures = plt.gca()
    return graph_count, plt, spectrafigures


@app.cell(hide_code=True)
def _(mo, spectra):
    mo.md(
        rf"""
    ## Spectra Overview

    - **Total Spectra Generated:** {len(spectra)}
    """
    )
    return


@app.cell
def _(mo, referencefigure):
    mo.md(
        rf"""
    ## Extract Reference Spectra
    Extract spectra of individual components as a reference for the model

    {mo.as_html(referencefigure)}
    """
    )
    return


@app.cell
def _(createTrainingData, plt, spectra, substanceDict):
    # Generate pure component reference spectra (no random scaling)
    # These serve as templates for identifying substances in mixtures
    referenceSpectrumIds = [
        substanceDict[substance][-1] for substance in substanceDict
    ]

    reference_spectra_raw = createTrainingData(
        substanceSpectrumIds=referenceSpectrumIds,
        sampleNumber=1,
        rondomlyScaleSubstances=False,  # Keep original intensities for references
    )

    # Map substance names to their reference spectra
    reference_spectra = {
        substanceDict[substance][0]: reference_spectra_raw['components'][index]
        for index, substance in enumerate(substanceDict)
    }

    print(reference_spectra)

    # Visualize reference spectra for each substance
    for substance in substanceDict:
        plt.plot(
            spectra[0]['positions'],
            reference_spectra[substanceDict[substance][0]],
        )

    referencefigure = plt.gca()

    return reference_spectra, referencefigure


@app.cell
def _(mo, preprocessedfigure, preprocessedreferencefigure):
    mo.md(
        rf"""
    ## Spectra Preprocessing
    - Extract only the relevant parts of the spectra
    - Add baseline distortion
    - Extract ratio of citrate to reference

    {mo.as_html(preprocessedfigure)}

    {mo.as_html(preprocessedreferencefigure)}
    """
    )
    return


@app.cell
def _(downsample, np, plt, reference_spectra, reverse, spectra, substanceDict):
    ## Preprocessing

    import multiprocessing as mp
    from scipy.signal import resample, hilbert
    from scipy.interpolate import interp1d
    from scipy.fft import ifft, irfft

    def preprocess_peaks(
        intensities,
        positions,
        ranges=[[-100, 100]],
        baseline_distortion=False,
        downsample=None,
        reverse=True,
    ):
        """
        Extract and preprocess spectral regions of interest.

        Args:
            intensities: Spectral intensity data
            positions: Chemical shift positions (ppm)
            ranges: List of [min, max] ppm ranges to extract
            baseline_distortion: Add realistic baseline drift
            downsample: Target number of points for downsampling
            reverse: Boolean to reverse fourier transform data or not
        """

        new_positions = positions #Default
        new_intensities = intensities #Default, in the case of no transformation
    
        if reverse == True:
            fid = ifft(hilbert(intensities))
            fid[0] = 0
            threshold = 1e-16
            fid[np.abs(fid) < threshold] = 0
            fid = fid[fid != 0]
            new_intensities = fid.astype(np.float32)
            new_positions = [0, 0]

        if downsample == None:
            return new_positions, new_intensities

        if len(new_intensities) > downsample:
            step = (
                len(new_intensities) // downsample
            )  # integer division for downsampling factor
            new_len = downsample
            new_nyquist = new_len // 2 + 1

            filtered = np.zeros_like(new_intensities)
            filtered[:new_nyquist] = new_intensities[:new_nyquist]

            time_domain = irfft(filtered, n=len(new_intensities))

            downsampled = new_intensities[::step]

            new_intensities = downsampled

        return new_positions, new_intensities

    def preprocess_ratio(scales, substanceDict):
        """Calculate concentration ratios relative to internal standard (tsp)"""
        ratios = {
            substance: scales[substance][0] / scales['tsp'][0]
            for substance in scales
        }

        return ratios

    def preprocess_spectra(
        spectra,
        ranges,
        substanceDict,
        baseline_distortion=False,
        downsample=None,
        reverse=True,
    ):
        """
        Complete preprocessing pipeline for a single spectrum.
        Extracts regions, adds distortion, calculates ratios.
        """
        new_positions, new_intensities = preprocess_peaks(
            intensities=spectra['intensities'][0],
            positions=spectra['positions'],
            ranges=ranges,
            baseline_distortion=baseline_distortion,
            downsample=downsample,
        )

        ratios = preprocess_ratio(spectra['scales'], substanceDict)

        return {
            'intensities': new_intensities,
            'positions': new_positions,
            'scales': spectra['scales'],
            'components': spectra['components'],
            'ratios': ratios,
        }

    # Preprocessing configuration
    ranges = [[-100, 100]]  # Full spectral range in ppm
    baseline_distortion = True  # Add realistic experimental artifacts

    def process_single_spectrum(spectrum):
        """Worker function for parallel spectrum preprocessing"""
        return preprocess_spectra(
            spectra=spectrum,
            ranges=ranges,
            substanceDict=substanceDict,
            baseline_distortion=baseline_distortion,
            downsample=downsample,  #
            reverse=reverse,
        )

    def process_single_reference(spectrum_key):
        """Worker function for parallel reference preprocessing"""
        pos_int = preprocess_peaks(
            positions=spectra[0]['positions'],
            intensities=reference_spectra[spectrum_key],
            downsample=downsample,
            reverse=reverse,
        )
        return (spectrum_key, pos_int[1])

    def process_spectra(spectra):
        """Parallel preprocessing of all training spectra"""
        num_processes = max(1, mp.cpu_count() - 1)
        print(f'Using {num_processes} processes for spectra preprocessing')

        with mp.Pool(processes=num_processes) as pool:
            preprocessed_spectra = pool.map(process_single_spectrum, spectra)

        return preprocessed_spectra

    def process_references(reference_spectra):
        """Parallel preprocessing of reference spectra"""
        num_processes = max(1, mp.cpu_count() - 1)
        print(f'Using {num_processes} processes for reference preprocessing')

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(
                process_single_reference, reference_spectra.keys()
            )

        preprocessed_reference_spectra = {
            key: intensities for key, intensities in results
        }

        return preprocessed_reference_spectra

    # Execute preprocessing pipelines
    preprocessed_spectra = process_spectra(spectra)
    preprocessed_reference_spectra = process_references(reference_spectra)

    def generate_figure():
        """Create before/after preprocessing comparison plots"""
        plt.figure(figsize=(8, 4))
        for substance in substanceDict:
            plt.subplot(1, 2, 1)
            plt.plot(
                spectra[0]['positions'],
                reference_spectra[substanceDict[substance][0]],
            )
            plt.subplot(1, 2, 2)
            plt.plot(
                preprocessed_reference_spectra[substanceDict[substance][0]],
            )

        return plt.gca()

    preprocessedreferencefigure = generate_figure()

    print(len(preprocessed_spectra[0]['positions']))
    print(len(preprocessed_spectra[0]['intensities']))

    positions_count = len(preprocessed_spectra[0]['positions'])
    intensities_count = len(preprocessed_spectra[0]['intensities'])
    return (
        baseline_distortion,
        intensities_count,
        positions_count,
        preprocessed_reference_spectra,
        preprocessed_spectra,
        preprocessedreferencefigure,
        ranges,
    )


@app.cell(hide_code=True)
def _(baseline_distortion, intensities_count, mo, positions_count, ranges):
    mo.md(
        rf"""
    ## Preprocessing Results

    **Configuration:**

    - **Baseline Distortion:** {'Enabled' if baseline_distortion else 'Disabled'}

    - **Spectral Ranges:** {ranges}

    **Processed Data Dimensions:**

    - **Positions Count:** {positions_count}

    - **Intensities Count:** {intensities_count}
    """
    )
    return


@app.cell
def _(graph_count, plt, preprocessed_spectra, spectra):
    print(len(spectra))

    # Compare original vs preprocessed spectra
    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for graphcounter2 in range(1, graph_count**2 + 1):
        plt.subplot(graph_count, graph_count, graphcounter2)
        # Original spectrum
        # plt.plot(
        #     spectra[graphcounter2]['intensities'][0],
        # )
        # Preprocessed spectrum (downsampled + baseline corrected)
        plt.plot(
            preprocessed_spectra[graphcounter2]['intensities'],
        )

    preprocessedfigure = plt.gca()
    return (preprocessedfigure,)


@app.cell
def _(
    held_back_metabolites,
    np,
    preprocessed_reference_spectra,
    preprocessed_spectra,
    random,
    substanceDict,
    torch,
):
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split

    class NMRDataset(Dataset):
        """Custom dataset for NMR spectral data that loads batches on demand"""

        def __init__(self, data, labels):
            # Keep data on CPU until needed
            self.data = data.cpu() if data.is_cuda else data
            self.labels = labels.cpu() if data.is_cuda else labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Return individual samples - DataLoader will handle batching
            return self.data[idx], self.labels[idx]

    def get_training_data_mlp(
        spectra,
        reference_spectra,
        held_back_metabolites,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        axes=0,
    ):
        """
        Improved data splitting with CPU-based datasets for memory efficiency.
        """
        data = []
        labels = []
        data_test = []
        labels_test = []
        data_val = []
        labels_val = []

        held_back_key_test = substanceDict[held_back_metabolites[0]][0]
        held_back_key_validation = substanceDict[held_back_metabolites[1]][0]
        print(
            f'Using held-back metabolites: {[x for x in held_back_metabolites]} (testkey: {held_back_key_test}, valkey: {held_back_key_validation})'
        )

        # Separate spectra based on held-back metabolite presence
        train_spectra = []
        val_with_holdback = []
        val_without_holdback = []
        test_with_holdback = []
        test_without_holdback = []
        data_train = []
        labels_train = []

        for spectrum in spectra:
            if held_back_key_test in spectrum['ratios']:
                test_with_holdback.append(spectrum)
            elif held_back_key_validation in spectrum['ratios']:
                val_with_holdback.append(spectrum)
            else:
                train_spectra.append(spectrum)

        # Split the non-holdback spectra into train/val and additional test data
        train_size = len(train_spectra)
        test_size = int(train_size * test_ratio)
        val_size = int(train_size * val_ratio)

        all_indices = list(range(train_size))
        random.seed(42)
        random.shuffle(all_indices)

        test_indices = set(all_indices[:test_size])
        val_indices = set(all_indices[test_size : test_size + val_size])
        train_indices = set(all_indices[test_size + val_size :])

        for i, spectrum in enumerate(train_spectra):
            if i in test_indices:
                test_without_holdback.append(spectrum)
            elif i in val_indices:
                val_without_holdback.append(spectrum)
            elif i in train_indices:
                for substance in reference_spectra:
                    if substance not in [
                        held_back_key_test,
                        held_back_key_validation,
                    ]:
                        temp_data = np.concatenate(
                            [
                                spectrum['intensities'],
                                reference_spectra[substance],
                            ]
                        )

                        if substance in spectrum['ratios']:
                            temp_label = [1, spectrum['ratios'][substance]]
                        else:
                            temp_label = [0, 0]

                        data_train.append(temp_data)
                        labels_train.append(temp_label)

        # Create test data from spectra WITH held-back metabolite
        for spectrum in test_with_holdback:
            temp_data = np.concatenate(
                [
                    spectrum['intensities'],
                    reference_spectra[held_back_key_test],
                ]
            )
            temp_label = [1, spectrum['ratios'][held_back_key_test]]
            data_test.append(temp_data)
            labels_test.append(temp_label)

        for spectrum in val_with_holdback:
            temp_data = np.concatenate(
                [
                    spectrum['intensities'],
                    reference_spectra[held_back_key_validation],
                ]
            )
            temp_label = [1, spectrum['ratios'][held_back_key_validation]]
            data_val.append(temp_data)
            labels_val.append(temp_label)

        # Create test data from spectra WITHOUT held-back metabolite
        for spectrum in test_without_holdback:
            temp_data = np.concatenate(
                [
                    spectrum['intensities'],
                    reference_spectra[held_back_key_test],
                ]
            )
            temp_label = [0, 0]
            data_test.append(temp_data)
            labels_test.append(temp_label)

        for spectrum in val_without_holdback:
            temp_data = np.concatenate(
                [
                    spectrum['intensities'],
                    reference_spectra[held_back_key_validation],
                ]
            )
            temp_label = [0, 0]
            data_val.append(temp_data)
            labels_val.append(temp_label)

        print(f'Training spectra: {len(train_spectra) - test_size - val_size}')
        print(
            f'Test spectra with {held_back_metabolites[0]}: {len(test_with_holdback)}'
        )
        print(
            f'Test spectra without {held_back_metabolites[0]}: {len(test_without_holdback)}'
        )
        print(f'Total test samples: {len(data_test)}')
        print(
            f'validation spectra with {held_back_metabolites[1]}: {len(val_with_holdback)}'
        )
        print(
            f'Validation spectra without {held_back_metabolites[1]}: {len(val_without_holdback)}'
        )
        print(f'Total validation samples: {len(data_val)}')

        # Convert to tensors but keep on CPU
        data_train = torch.tensor(data_train, dtype=torch.float32)
        labels_train = torch.tensor(labels_train, dtype=torch.float32)
        data_val = torch.tensor(data_val, dtype=torch.float32)
        labels_val = torch.tensor(labels_val, dtype=torch.float32)
        data_test = torch.tensor(data_test, dtype=torch.float32)
        labels_test = torch.tensor(labels_test, dtype=torch.float32)

        # Create datasets (data stays on CPU)
        train_dataset = NMRDataset(data_train, labels_train)
        val_dataset = NMRDataset(data_val, labels_val)
        test_dataset = NMRDataset(data_test, labels_test)

        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
        }

    training_data = get_training_data_mlp(
        spectra=preprocessed_spectra,
        reference_spectra=preprocessed_reference_spectra,
        held_back_metabolites=held_back_metabolites,
    )

    print([len(training_data[x]) for x in training_data])
    print(len(training_data['train_dataset'][0][0]))

    sample_ratio = preprocessed_spectra[0]['ratios']
    data_length = len(training_data['train_dataset'][0][0])

    return DataLoader, data_length, sample_ratio, training_data


@app.cell(hide_code=True)
def _(data_length, mo, sample_ratio, training_data):
    mo.md(
        rf"""
    ## Training Data Preparation

    **Sample Information:**

    - **Sample Ratios (first spectrum):** {sample_ratio:}

    - **Feature Vector Length:** {data_length}

    **Tensor Shapes:**

    {chr(10).join([f"- **{key}:** {value.shape}" for key, value in training_data.items()])}
    """
    )
    return


@app.cell
def _(DataLoader, np, torch, tqdm):
    import copy
    import torch.optim as optim
    import torch.nn as nn
    import math

    class MLPRegressor(nn.Module):
        def __init__(self, input_size, trial=None, div_size=None):
            """
            Multi-layer perceptron for regression tasks.

            Args:
                input_size: Number of input features
                trial: Optuna trial object for hyperparameter suggestion
                div_size: Division factor for layer size reduction (if trial is None)
            """
            super(MLPRegressor, self).__init__()

            # Get division factor from trial or use provided value
            if trial is not None:
                self.div_size = trial.suggest_float('div_size', 2, 10, step=1)
            elif div_size is not None:
                self.div_size = div_size
            else:
                self.div_size = 4  # Default value

            # Progressive layer size reduction based on division factor
            a = input_size  # Input feature dimension
            b = int(a / self.div_size)
            c = int(b / self.div_size)
            d = int(c / self.div_size)
            e = int(d / self.div_size)

            # Store layer sizes for reference
            self.layer_sizes = [a, b, c, d, e, 1]

            # Define the model architecture
            self.model = nn.Sequential(
                nn.Linear(a, b),
                nn.ReLU(),
                nn.Linear(b, c),
                nn.ReLU(),
                nn.Linear(c, d),
                nn.ReLU(),
                nn.Linear(d, e),
                nn.ReLU(),
                nn.Linear(e, 2),  # Output: concentration
            )

        def forward(self, x):
            return self.model(x)

        def get_architecture_info(self):
            """Return information about the model architecture"""
            return {
                'div_size': self.div_size,
                'layer_sizes': self.layer_sizes,
                'total_parameters': sum(p.numel() for p in self.parameters()),
            }

    class ImprovedTransformerRegressor(nn.Module):
        def __init__(self, input_size, trial=None, **kwargs):
            """
            Improved Transformer model for NMR spectral analysis.
            """
            super(ImprovedTransformerRegressor, self).__init__()

            # Get hyperparameters
            if trial is not None:
                self.d_model = int(trial.suggest_categorical('d_model', [128, 256, 512]))
                self.nhead = int(trial.suggest_categorical('nhead', [8, 16]))
                self.num_layers = int(trial.suggest_int('num_layers', 3, 6))
                self.dim_feedforward = int(trial.suggest_categorical('dim_feedforward', [512, 1024, 2048]))
                # Remove dropout from hyperparameter search
                # Better sequence length strategy
                self.target_seq_len = int(trial.suggest_categorical('target_seq_len', [64, 128, 256]))
            else:
                self.d_model = kwargs.get('d_model', 256)
                self.nhead = kwargs.get('nhead', 8)
                self.num_layers = kwargs.get('num_layers', 4)
                self.dim_feedforward = kwargs.get('dim_feedforward', 1024)
                # Remove dropout parameter
                self.target_seq_len = kwargs.get('target_seq_len', 128)

            # Ensure nhead divides d_model evenly
            while self.d_model % self.nhead != 0:
                self.nhead = max(1, self.nhead - 1)

            # Use linear projection instead of patch embedding to preserve all data
            self.spectrum_projection = nn.Linear(input_size // 2, self.d_model * self.target_seq_len)
            self.reference_projection = nn.Linear(input_size // 2, self.d_model * self.target_seq_len)

            # Set actual sequence length to target (no data reduction)
            self.actual_seq_len = self.target_seq_len

            # Separate embeddings for spectrum and reference (no dropout)
            self.spectrum_pos_encoding = PositionalEncoding(self.d_model, 0.0, self.actual_seq_len)

            # Cross-attention between spectrum and reference (no dropout)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.nhead,
                dropout=0.0,  # Remove dropout
                batch_first=True
            )

            # Self-attention transformer (no dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=0.0,  # Remove dropout
                activation='gelu',
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_layers
            )

            # Multi-scale feature extraction (preserve all data)
            self.multi_scale_conv = nn.ModuleList([
                nn.Conv1d(self.d_model, self.d_model // 4, kernel_size=k, padding=k//2)
                for k in [3, 7, 15]  # Different receptive fields
            ])

            # Calculate enriched feature size
            enriched_feature_size = self.d_model + (3 * self.d_model // 4)

            # Attention-based pooling for enriched features
            self.attention_pool = nn.Sequential(
                nn.Linear(enriched_feature_size, enriched_feature_size // 4),
                nn.Tanh(),
                nn.Linear(enriched_feature_size // 4, 1)
            )

            # Update task-specific heads to use enriched features
            self.presence_head = nn.Sequential(
                nn.Linear(enriched_feature_size, enriched_feature_size // 2),
                nn.LayerNorm(enriched_feature_size // 2),
                nn.GELU(),
                nn.Linear(enriched_feature_size // 2, 1)
            )

            self.concentration_head = nn.Sequential(
                nn.Linear(enriched_feature_size, enriched_feature_size // 2),
                nn.LayerNorm(enriched_feature_size // 2),
                nn.GELU(),
                nn.Linear(enriched_feature_size // 2, enriched_feature_size // 4),
                nn.GELU(),
                nn.Linear(enriched_feature_size // 4, 1)
            )

        def forward(self, x):
            batch_size = x.size(0)

            # Split input into spectrum and reference (assuming concatenated)
            mid_point = x.size(1) // 2
            spectrum = x[:, :mid_point]  # [batch, seq_len]
            reference = x[:, mid_point:]

            # Project to transformer dimensions without data loss
            spectrum_projected = self.spectrum_projection(spectrum)  # [batch, d_model * target_seq_len]
            reference_projected = self.reference_projection(reference)

            # Reshape to sequence format
            spectrum_patches = spectrum_projected.view(batch_size, self.target_seq_len, self.d_model)
            reference_patches = reference_projected.view(batch_size, self.target_seq_len, self.d_model)

            # Add positional encoding (no dropout)
            spectrum_patches = self.spectrum_pos_encoding(spectrum_patches)
            reference_patches = self.spectrum_pos_encoding(reference_patches)

            # Cross-attention between spectrum and reference
            attended_spectrum, _ = self.cross_attention(
                spectrum_patches, reference_patches, reference_patches
            )

            # Self-attention on attended spectrum
            encoded = self.transformer_encoder(attended_spectrum)

            # Multi-scale feature extraction
            encoded_transposed = encoded.transpose(1, 2)  # [batch, d_model, seq_len]
            multi_scale_features = []
            for conv in self.multi_scale_conv:
                multi_scale_features.append(conv(encoded_transposed))

            # Combine multi-scale features
            combined_features = torch.cat(multi_scale_features, dim=1)  # [batch, 3*d_model//4, seq_len]
            combined_features = combined_features.transpose(1, 2)  # [batch, seq_len, 3*d_model//4]

            # FIXED: Concatenate original transformer features with multi-scale features
            enriched_features = torch.cat([encoded, combined_features], dim=2)  # [batch, seq_len, d_model + 3*d_model//4]

            # Attention-based pooling using enriched features
            attention_weights = self.attention_pool(enriched_features)  # [batch, seq_len, 1]
            attention_weights = torch.softmax(attention_weights, dim=1)
            pooled_features = torch.sum(enriched_features * attention_weights, dim=1)  # [batch, d_model + 3*d_model//4]

            # Task-specific predictions
            presence_logits = self.presence_head(pooled_features)  # [batch, 1]
            concentration_pred = self.concentration_head(pooled_features)  # [batch, 1]

            return torch.cat([presence_logits, concentration_pred], dim=1)  # [batch, 2]

    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer input sequences"""

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            # Remove dropout completely
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)

        def forward(self, x):
            # x shape: [batch_size, seq_len, d_model]
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].transpose(0, 1)
            return x

    class HybridEnsembleRegressor(nn.Module):
        def __init__(self, input_size, trial=None, **kwargs):
            super(HybridEnsembleRegressor, self).__init__()

            # Initialize both models
            self.mlp = MLPRegressor(input_size, trial, **kwargs)
            self.transformer = ImprovedTransformerRegressor(input_size, trial, **kwargs)

            # Learnable ensemble weights
            if trial is not None:
                self.classification_weight = trial.suggest_float('class_ensemble_weight', 0.1, 0.9)
                self.concentration_weight = trial.suggest_float('conc_ensemble_weight', 0.1, 0.9)
            else:
                self.classification_weight = kwargs.get('class_ensemble_weight', 0.3)  # Favor transformer
                self.concentration_weight = kwargs.get('conc_ensemble_weight', 0.7)   # Favor MLP

        def forward(self, x):
            mlp_output = self.mlp(x)
            transformer_output = self.transformer(x)

            # Weighted ensemble for each task
            classification_pred = (
                self.classification_weight * mlp_output[:, 0] + 
                (1 - self.classification_weight) * transformer_output[:, 0]
            )

            concentration_pred = (
                self.concentration_weight * mlp_output[:, 1] + 
                (1 - self.concentration_weight) * transformer_output[:, 1]
            )

            return torch.stack([classification_pred, concentration_pred], dim=1)

    def improved_compute_loss(predictions, targets, epoch=0, max_epochs=200):
        """
        Improved loss function with curriculum learning and adaptive weighting.
        """
        presence_logits = predictions[:, 0]
        concentration_pred = predictions[:, 1]
        presence_true = targets[:, 0]
        concentration_true = targets[:, 1]

        # Classification loss
        classification_loss = nn.BCEWithLogitsLoss()(presence_logits, presence_true)

        # Curriculum learning: gradually increase concentration loss weight
        curriculum_weight = min(1.0, epoch / (max_epochs * 0.3))

        # Adaptive concentration loss based on presence confidence
        presence_prob = torch.sigmoid(presence_logits)
        present_mask = presence_true == 1

        if present_mask.sum() > 0:
            # Weight concentration loss by presence confidence
            confidence_weights = presence_prob[present_mask]

            # Huber loss for robustness to outliers
            concentration_diff = concentration_pred[present_mask] - concentration_true[present_mask]
            huber_loss = nn.SmoothL1Loss()(
                concentration_pred[present_mask], 
                concentration_true[present_mask]
            )

            # Calculate MAE and RMSE for monitoring
            concentration_mae = torch.mean(torch.abs(concentration_diff))
            concentration_rmse = torch.sqrt(torch.mean(concentration_diff ** 2))

            # Scale-aware loss: relative error for different concentration ranges
            relative_error = torch.abs(concentration_diff) / (concentration_true[present_mask] + 1e-8)
            concentration_loss = torch.mean(confidence_weights * (huber_loss + 0.1 * relative_error))
        else:
            concentration_loss = torch.tensor(0.0, device=predictions.device)
            concentration_mae = torch.tensor(0.0, device=predictions.device)
            concentration_rmse = torch.tensor(0.0, device=predictions.device)

        # Adaptive loss weighting
        total_loss = classification_loss + curriculum_weight * concentration_loss

        return total_loss, classification_loss, concentration_mae, concentration_rmse

    def train_model(training_data, trial, model_type='mlp'):
        """
        Train a multi-task neural network using DataLoaders for memory efficiency.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        max_epochs = 200
        batch_size = int(trial.suggest_float('batch_size', 10, 100, step=10))
        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        # Get input size from first sample
        sample_data, _ = training_data['train_dataset'][0]
        input_length = len(sample_data)

        if model_type == 'transformer':
            model = ImprovedTransformerRegressor(
                input_size=input_length, trial=trial
            ).to(device)
        elif model_type == 'mlp':
            model = MLPRegressor(input_size=input_length, trial=trial).to(device)
        elif model_type == 'ensemble':
            model = HybridEnsembleRegressor(input_size=input_length, trial=trial).to(device)

        # Create DataLoaders with pin_memory for faster GPU transfer
        train_loader = DataLoader(
            training_data['train_dataset'], 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            training_data['val_dataset'], 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )
        test_loader = DataLoader(
            training_data['test_dataset'], 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )

        # Test model with a small batch to catch memory issues early
        try:
            sample_batch = next(iter(train_loader))
            dummy_data, dummy_labels = sample_batch
            dummy_data = dummy_data.to(device, non_blocking=True)
            dummy_labels = dummy_labels.to(device, non_blocking=True)

            dummy_output = model(dummy_data)
            dummy_loss = torch.nn.MSELoss()(dummy_output, dummy_labels)
            dummy_loss.backward()
            model.zero_grad()

            del dummy_data, dummy_labels, dummy_output, dummy_loss
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            del model
            torch.cuda.empty_cache()
            raise e

        try:
            model = torch.compile(model)
        except Exception:
            print('Compilation failed — using uncompiled model.')

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Early stopping parameters
        early_stop_patience = 15
        min_delta = 1e-6
        validation_interval = 5

        best_val_loss = np.inf
        epochs_without_improvement = 0
        best_weights = None

        for epoch in range(max_epochs):
            model.train()

            # Training loop with DataLoader
            with tqdm.tqdm(train_loader, unit='batch', mininterval=0, disable=True) as bar:
                bar.set_description(f'Epoch {epoch}')
                for data_batch, labels_batch in bar:
                    # Move batch to GPU only when needed
                    data_batch = data_batch.to(device, non_blocking=True)
                    labels_batch = labels_batch.to(device, non_blocking=True)

                    predictions = model(data_batch)
                    total_loss, class_loss, conc_mae, conc_rmse = improved_compute_loss(
                        predictions, labels_batch, epoch, max_epochs
                    )

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    bar.set_postfix(
                        total_loss=float(total_loss),
                        class_loss=float(class_loss),
                        conc_mae=float(conc_mae),
                        conc_rmse=float(conc_rmse),
                    )

            # Validation
            if epoch % validation_interval == 0 or epoch == max_epochs - 1:
                model.eval()
                val_losses = []

                with torch.no_grad():
                    for data_batch, labels_batch in val_loader:
                        data_batch = data_batch.to(device, non_blocking=True)
                        labels_batch = labels_batch.to(device, non_blocking=True)

                        predictions = model(data_batch)
                        val_loss, _, _, _ = improved_compute_loss(
                            predictions, labels_batch, epoch, max_epochs
                        )
                        val_losses.append(float(val_loss))

                avg_val_loss = np.mean(val_losses)

                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    best_weights = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += validation_interval

                if epochs_without_improvement >= early_stop_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # Load best weights
        model.load_state_dict(best_weights)
        model.eval()

        # Compute final metrics using DataLoaders
        def compute_metrics(data_loader):
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for data_batch, labels_batch in data_loader:
                    data_batch = data_batch.to(device, non_blocking=True)
                    labels_batch = labels_batch.to(device, non_blocking=True)

                    predictions = model(data_batch)
                    all_predictions.append(predictions)
                    all_targets.append(labels_batch)

            # Concatenate all batches
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            presence_logits = all_predictions[:, 0]
            concentration_pred = all_predictions[:, 1]
            presence_pred = torch.sigmoid(presence_logits)

            presence_true = all_targets[:, 0]
            concentration_true = all_targets[:, 1]

            # Classification metrics
            presence_binary = (presence_pred > 0.5).float()
            accuracy = torch.mean((presence_binary == presence_true).float())

            # Regression metrics
            present_mask = presence_true == 1
            if present_mask.sum() > 0:
                conc_mae = torch.mean(
                    torch.abs(concentration_pred[present_mask] - concentration_true[present_mask])
                )
                conc_rmse = torch.sqrt(
                    torch.mean((concentration_pred[present_mask] - concentration_true[present_mask]) ** 2)
                )
                ss_res = torch.sum((concentration_true[present_mask] - concentration_pred[present_mask]) ** 2)
                ss_tot = torch.sum((concentration_true[present_mask] - torch.mean(concentration_true[present_mask])) ** 2)
                conc_r2 = 1 - (ss_res / ss_tot)
            else:
                conc_mae = torch.tensor(0.0)
                conc_rmse = torch.tensor(0.0)
                conc_r2 = torch.tensor(0.0)

            return float(accuracy), float(conc_r2), float(conc_mae), float(conc_rmse)

        # Compute validation and test metrics
        val_accuracy, val_conc_r2, val_conc_mae, val_conc_rmse = compute_metrics(val_loader)
        test_accuracy, test_conc_r2, test_conc_mae, test_conc_rmse = compute_metrics(test_loader)

        return (
            val_accuracy, val_conc_r2, val_conc_mae, val_conc_rmse,
            test_accuracy, test_conc_r2, test_conc_mae, test_conc_rmse,
        )

    # Store device info for display
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_info = device
    gpu_name = ''
    if torch.cuda.is_available():
        gpu_name = f' ({torch.cuda.get_device_name(0)})'
    return device_info, gpu_name, train_model


@app.cell(hide_code=True)
def _(device_info, gpu_name, mo):
    mo.md(
        rf"""
    ## Model Training Setup

    **Hardware Configuration:**

    - **Training Device:** {device_info} / {gpu_name}

    **Model Architecture:** Transformer

    - Final output: Classification Logits and concentration regression
    """
    )
    return


@app.cell
def _(MODEL_TYPE, cache_dir, torch, tqdm, train_model, training_data, trials):
    import optuna
    from functools import partial

    def objective(training_data, trial, model_type='transformer'):
        """
        Optuna objective function for hyperparameter optimization with GPU memory handling.
        """
        try:
            (
                val_accuracy,
                val_conc_r2,
                val_conc_mae,
                val_conc_rmse,
                test_accuracy,
                test_conc_r2,
                test_conc_mae,
                test_conc_rmse,
            ) = train_model(training_data, trial, model_type)

            # Store all metrics in trial for later analysis
            trial.set_user_attr('val_accuracy', val_accuracy)
            trial.set_user_attr('val_conc_r2', val_conc_r2)
            trial.set_user_attr('val_conc_mae', val_conc_mae)
            trial.set_user_attr('val_conc_rmse', val_conc_rmse)
            trial.set_user_attr('test_accuracy', test_accuracy)
            trial.set_user_attr('test_conc_r2', test_conc_r2)
            trial.set_user_attr('test_conc_mae', test_conc_mae)
            trial.set_user_attr('test_conc_rmse', test_conc_rmse)
            trial.set_user_attr('model_type', model_type)

            # Calculate classification error (1 - accuracy)
            val_classification_error = 1.0 - val_accuracy

            # Use the new optimization formula
            combined_score = 0.5 * val_classification_error + 0.5 * (
                0.5 * val_conc_mae + 0.5 * val_conc_rmse
            )

            return combined_score

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Clear GPU cache immediately
            torch.cuda.empty_cache()

            # Check if it's specifically a memory error
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                # Log this as a memory failure
                trial.set_user_attr('memory_error', True)
                trial.set_user_attr('error_message', str(e))

                # Return a penalty score that's worse than any reasonable performance
                # but not so bad that it completely dominates the search space
                penalty_score = 2.0  # Worse than worst possible (1.0 classification error + 1.0 regression error)

                return penalty_score
            else:
                # Re-raise other RuntimeErrors as they might be actual bugs
                raise e

        except Exception as e:
            # Clear GPU cache for any other errors too
            torch.cuda.empty_cache()

            # Log the error for debugging
            trial.set_user_attr('general_error', True)
            trial.set_user_attr('error_message', str(e))

            # Return a different penalty for non-memory errors
            penalty_score = 1.5

            return penalty_score

    # Create or load existing Optuna study with persistent SQLite storage
    study = optuna.create_study(
        direction='minimize',  # Maximize the negative combined error (minimize error)
        study_name='study',  # Use cache key for unique study identification
        storage=f'sqlite:///{cache_dir}/database.db',  # Use cache key for database filename
        load_if_exists=True,  # Resume previous optimization if study exists
    )

    # Count completed trials for progress tracking
    completed_trials = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    )

    # Run hyperparameter optimization if more trials are needed
    if trials - completed_trials > 0:
        with tqdm.tqdm(
            total=trials - completed_trials, desc='Optimizing'
        ) as pbar:

            def callback(study, trial):
                """Progress callback for Optuna optimization"""
                pbar.update(1)

            # Resume or start hyperparameter optimization
            study.optimize(
                partial(objective, training_data, model_type=MODEL_TYPE),
                callbacks=[
                    optuna.study.MaxTrialsCallback(
                        trials, states=(optuna.trial.TrialState.COMPLETE,)
                    ),
                    callback,
                ],
            )

    return optuna, study


@app.cell(hide_code=True)
def _(optuna, study):
    # Analyze failed trials for debugging
    failed_trials = [
        t for t in study.trials 
        if t.state in [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED]
    ]

    error_trials = [
        t for t in study.trials
        if t.user_attrs.get('memory_error', False) or t.user_attrs.get('general_error', False)
    ]

    if failed_trials or error_trials:
        error_summary = f"""
    ## Trial Error Analysis

    **Failed Trials Summary:**
    - **Total Failed/Pruned Trials:** {len(failed_trials)}
    - **Memory Error Trials:** {len([t for t in error_trials if t.user_attrs.get('memory_error', False)])}
    - **General Error Trials:** {len([t for t in error_trials if t.user_attrs.get('general_error', False)])}

    **Error Details:**
    """

        # Group errors by type and message
        error_groups = {}
        for trial in error_trials:
            error_type = "Memory Error" if trial.user_attrs.get('memory_error', False) else "General Error"
            error_msg = trial.user_attrs.get('error_message', 'Unknown error')

            # Truncate long error messages
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."

            key = f"{error_type}: {error_msg}"
            if key not in error_groups:
                error_groups[key] = []
            error_groups[key].append(trial.number)

        for error_desc, trial_numbers in error_groups.items():
            error_summary += f"\n**{error_desc}**\n- Trials: {trial_numbers[:10]}{'...' if len(trial_numbers) > 10 else ''} ({len(trial_numbers)} total)\n"

        # Add parameter analysis for memory errors
        memory_error_trials = [t for t in error_trials if t.user_attrs.get('memory_error', False)]
        if memory_error_trials:
            error_summary += "\n**Memory Error Parameter Analysis:**\n"

            # Analyze common parameters in memory errors
            if memory_error_trials[0].params:
                param_ranges = {}
                for trial in memory_error_trials:
                    for param, value in trial.params.items():
                        if param not in param_ranges:
                            param_ranges[param] = []
                        param_ranges[param].append(value)

                for param, values in param_ranges.items():
                    if isinstance(values[0], (int, float)):
                        error_summary += f"- **{param}:** {min(values):.3f} - {max(values):.3f} (avg: {sum(values)/len(values):.3f})\n"
                    else:
                        unique_vals = list(set(values))
                        error_summary += f"- **{param}:** {unique_vals}\n"

        print(error_summary)
    else:
        print("## Trial Error Analysis\n\n✅ **No failed trials detected** - All optimization trials completed successfully!")

    return


@app.cell(hide_code=True)
def _(held_back_metabolites, mo, optuna, study):
    # Determine model type from best trial parameters
    model_type = (
        'Transformer' if 'd_model' in study.best_trial.params else 'MLP'
    )

    # Create model-specific parameter display
    if (model_type == 'Transformer'):
        model_params_md = f"""
    **Transformer Architecture:**
    - **Model Dimension (d_model):** {study.best_trial.params['d_model']}
    - **Number of Attention Heads:** {study.best_trial.params['nhead']}
    - **Number of Encoder Layers:** {study.best_trial.params['num_layers']}
    - **Feedforward Dimension:** {study.best_trial.params['dim_feedforward']}
    - **Target Sequence Length:** {study.best_trial.params['target_seq_len']}

    **Model Architecture:**

    Input Projection → Positional Encoding → Transformer Encoder → Global Average Pooling → Output Projection
    """
    else:  # MLP
        model_params_md = f"""
    **MLP Architecture:**
    - **Division Size (layer reduction factor):** {study.best_trial.params['div_size']:.1f}

    **Model Architecture:**

    Input Layer → Hidden Layers (progressively smaller) → Output Layer (2 outputs)

    *Layer sizes are determined by dividing the previous layer size by the division factor*
    """

    mo.md(
        f"""
    ## Hyperparameter Optimization Results

    **Model Type:** {model_type}

    **Best Trial Performance (Validation Set):**

    - **Combined Score: {study.best_trial.value:.4f}** (0.5 * Classification Error + 0.5 * (0.5*MAE + 0.5*RMSE), optimized metric - lower is better)
    - **Classification Accuracy: {study.best_trial.user_attrs['val_accuracy']:.4f}** (Presence prediction accuracy - higher is better)
    - **Concentration R²: {study.best_trial.user_attrs['val_conc_r2']:.4f}** (Coefficient of determination for concentration - higher is better)
    - **Concentration MAE: {study.best_trial.user_attrs['val_conc_mae']:.6f}** (Mean Absolute Error for concentration - lower is better)
    - **Concentration RMSE: {study.best_trial.user_attrs['val_conc_rmse']:.6f}** (Root Mean Square Error for concentration - lower is better)

    **Final Test Set Performance:**

    - **Classification Accuracy: {study.best_trial.user_attrs['test_accuracy']:.4f}**
    - **Concentration R²: {study.best_trial.user_attrs['test_conc_r2']:.4f}**
    - **Concentration MAE: {study.best_trial.user_attrs['test_conc_mae']:.6f}**
    - **Concentration RMSE: {study.best_trial.user_attrs['test_conc_rmse']:.6f}**

    **Best Hyperparameters:**

    **Training Parameters:**
    - **Batch Size:** {study.best_trial.params['batch_size']:.0f}
    - **Learning Rate:** {study.best_trial.params['lr']:.2e}

    {model_params_md}

    **Multi-Task Learning:**

    - **Task 1:** Binary classification for substance presence (BCEWithLogitsLoss)
    - **Task 2:** Regression for concentration prediction (weighted by presence)
    - **Combined Loss:** 0.5 × Classification Error + 0.5 × (0.5×MAE + 0.5×RMSE)

    **Data Split:**

    - Training: Spectra without held-back metabolite
    - Validation: 15% of training data (used for hyperparameter optimization)
    - Test: Spectra containing held-back metabolite ({held_back_metabolites})

    **Total Trials Completed:** {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}
    """
    )
    return


if __name__ == "__main__":
    app.run()
