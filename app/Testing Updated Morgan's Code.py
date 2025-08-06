import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

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

    import os

    count = 10
    trials = 1000
    combo_number = 30
    notebook_name = 'testing_morgan_updated'
    cache_dir = f'./data_cache/{notebook_name}'

    if os.path.exists(cache_dir) == False:
        os.mkdir(cache_dir)

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
    return cache_dir, combo_number, count, os, substanceDict, trials


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
def _(cache_dir, combo_number, count, os, substanceDict):
    from morgancopy.createTrainingData import createTrainingData
    import numpy as np
    import tqdm
    import itertools
    import multiprocessing as mp
    import pickle
    from pathlib import Path
    import random

    substances = list(substanceDict.keys())

    # Save/load functions for data persistence
    def save_spectra_data(
        spectra, held_back_metabolite, combinations, filename
    ):
        """Save generated spectra data, held-back metabolite, and combinations to pickle file"""
        os.makedirs(cache_dir, exist_ok=True)
        filepath = f'{cache_dir}/{filename}.pkl'

        data_to_save = {
            'spectra': spectra,
            'held_back_metabolite': held_back_metabolite,
            'combinations': combinations,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(
            f"Saved {len(spectra)} spectra, held-back metabolite '{held_back_metabolite}', and {len(combinations)} combinations to {filepath}"
        )

    def load_spectra_data(filename):
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
                # Medium format - spectra + held_back_metabolite
                spectra = data['spectra']
                held_back_metabolite = data['held_back_metabolite']
                print(
                    f"Loaded {len(spectra)} spectra and held-back metabolite '{held_back_metabolite}' from {filepath} (medium format)"
                )
                return spectra, held_back_metabolite, None
            else:
                # New format - spectra + held_back_metabolite + combinations
                spectra = data['spectra']
                held_back_metabolite = data['held_back_metabolite']
                combinations = data['combinations']
                print(
                    f"Loaded {len(spectra)} spectra, held-back metabolite '{held_back_metabolite}', and {len(combinations)} combinations from {filepath}"
                )
                return spectra, held_back_metabolite, combinations
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

    def check_loaded_data(spectra, held_back_metabolite, combinations):
        if spectra is None:
            print('No cached data found. Generating new spectra...')

            # Generate all possible combinations of substances (4 to n substances)
            # This creates training data for different metabolite mixtures
            all_combinations = []
            for r in range(4, len(substances) + 1):
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
            held_back_metabolite = random.choice(list(substanceDict.keys()))
            print(
                f"Selected '{held_back_metabolite}' as held-back metabolite for testing"
            )

            # Extract spectrum IDs for each combination
            substanceSpectrumIds = [
                [combination[substance][-1] for substance in combination]
                for combination in combinations
            ]

            # Prepare multiprocessing arguments - one batch per substance combination
            mp_args = [
                (substances, count) for substances in substanceSpectrumIds
            ]

            # Use multiprocessing to parallelize data generation across CPU cores
            num_processes = max(1, mp.cpu_count() - 1)
            print(f'Using {num_processes} processes for data generation')

            batch_data = []
            if len(mp_args) > 1:
                with mp.Pool(processes=num_processes) as pool:
                    batch_data = list(
                        tqdm.tqdm(
                            pool.imap_unordered(create_batch_data, mp_args),
                            total=len(mp_args),
                        )
                    )
            else:
                batch_data = [create_batch_data(mp_args[0])]

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
                spectra, held_back_metabolite, combinations, cache_key
            )
        else:
            print('Using cached spectra data.')
            if held_back_metabolite is None:
                # If old cache format, select and save new held-back metabolite
                held_back_metabolite = random.choice(
                    list(substanceDict.keys())
                )
                print(
                    f"Cache missing held-back metabolite. Selected '{held_back_metabolite}' and updating cache..."
                )
                save_spectra_data(
                    spectra, held_back_metabolite, combinations, cache_key
                )
            print(f'Using {len(combinations)} combinations from cache')

        return spectra, held_back_metabolite, combinations

    # Generate cache key for current configuration
    cache_key = generate_cache_key(substanceDict, combo_number, count)

    # Try to load existing data first
    spectra, held_back_metabolite, combinations = load_spectra_data(cache_key)
    spectra, held_back_metabolite, combinations = check_loaded_data(
        spectra, held_back_metabolite, combinations
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
        cache_key,
        combinations,
        components_shape,
        createTrainingData,
        held_back_metabolite,
        intensities_shape,
        mp,
        np,
        positions_shape,
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
def _(mp, np, plt, reference_spectra, spectra, substanceDict):
    ## Preprocessing

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
            step = len(new_intensities) // downsample  # integer division for downsampling factor
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
        spectra, ranges, substanceDict, baseline_distortion=False, downsample=None, reverse=True
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
    downsample = int(2**13)  # Target resolution for ML model

    def process_single_spectrum(spectrum):
        """Worker function for parallel spectrum preprocessing"""
        return preprocess_spectra(
            spectra=spectrum,
            ranges=ranges,
            substanceDict=substanceDict,
            baseline_distortion=baseline_distortion,
            downsample=downsample,#
        )

    def process_single_reference(spectrum_key):
        """Worker function for parallel reference preprocessing"""
        pos_int = preprocess_peaks(
            positions=spectra[0]['positions'],
            intensities=reference_spectra[spectrum_key],
            downsample=2048,
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
        plt.plot(
            spectra[graphcounter2]['positions'],
            spectra[graphcounter2]['intensities'][0],
        )
        # Preprocessed spectrum (downsampled + baseline corrected)
        plt.plot(
            preprocessed_spectra[graphcounter2]['intensities'],
        )

    preprocessedfigure = plt.gca()
    return (preprocessedfigure,)


@app.cell
def _(
    held_back_metabolite,
    np,
    preprocessed_reference_spectra,
    preprocessed_spectra,
    substanceDict,
    torch,
):
    from sklearn.model_selection import train_test_split

    # Configure device for GPU acceleration if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    def get_training_data_mlp(
        spectra,
        reference_spectra,
        held_back_metabolite,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,  # Add explicit test ratio
        axes=0,
    ):
        """
        Improved data splitting to prevent data leakage and overfitting.
        Now includes test data both with and without the held-back metabolite.
        """
        data = []
        labels = []
        data_test = []
        labels_test = []

        held_back_key = substanceDict[held_back_metabolite][0]
        print(
            f'Using held-back metabolite: {held_back_metabolite} (key: {held_back_key})'
        )

        # Separate spectra based on held-back metabolite presence
        train_val_spectra = []  # Spectra without held-back metabolite for train/val
        test_with_holdback = []  # Spectra with held-back metabolite for testing
        test_without_holdback = []  # Spectra without held-back metabolite for testing

        for spectrum in spectra:
            if held_back_key in spectrum['ratios']:
                # This spectrum contains the held-back metabolite - use for testing
                test_with_holdback.append(spectrum)
            else:
                # This spectrum doesn't contain held-back metabolite
                train_val_spectra.append(spectrum)

        # Split the non-holdback spectra into train/val and additional test data
        # Reserve some of the non-holdback data for testing to evaluate generalization
        total_train_val = len(train_val_spectra)
        test_size = int(total_train_val * test_ratio)

        # Randomly sample some non-holdback spectra for testing
        import random
        random.seed(42)  # For reproducibility
        test_indices = random.sample(range(total_train_val), min(test_size, total_train_val))

        for i, spectrum in enumerate(train_val_spectra):
            if i in test_indices:
                test_without_holdback.append(spectrum)
            else:
                # Use for training/validation
                for substance in reference_spectra:
                    if substance != held_back_key:  # Skip held-back substance in training
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

                        data.append(temp_data)
                        labels.append(temp_label)

        # Create test data from spectra WITH held-back metabolite
        for spectrum in test_with_holdback:
            temp_data = np.concatenate(
                [
                    spectrum['intensities'],
                    reference_spectra[held_back_key],
                ]
            )
            temp_label = [1, spectrum['ratios'][held_back_key]]
            data_test.append(temp_data)
            labels_test.append(temp_label)

        # Create test data from spectra WITHOUT held-back metabolite
        # Test the model's ability to correctly predict absence
        for spectrum in test_without_holdback:
            temp_data = np.concatenate(
                [
                    spectrum['intensities'],
                    reference_spectra[held_back_key],
                ]
            )
            # The held-back metabolite should be absent (label = [0, 0])
            temp_label = [0, 0]
            data_test.append(temp_data)
            labels_test.append(temp_label)

        print(f'Training/validation spectra: {len(train_val_spectra) - test_size}')
        print(f'Test spectra with {held_back_metabolite}: {len(test_with_holdback)}')
        print(f'Test spectra without {held_back_metabolite}: {len(test_without_holdback)}')
        print(f'Total test samples: {len(data_test)}')

        # Split training data into train/validation
        data = np.array(data)
        data_train, data_val, labels_train, labels_val = train_test_split(
            data, labels, train_size=train_ratio/(train_ratio+val_ratio), shuffle=True, random_state=42
        )

        # Convert to tensors
        data_train = torch.tensor(data_train, dtype=torch.float32).to(device)
        labels_train = torch.tensor(labels_train, dtype=torch.float32).to(device)
        data_val = torch.tensor(data_val, dtype=torch.float32).to(device)
        labels_val = torch.tensor(labels_val, dtype=torch.float32).to(device)
        data_test = torch.tensor(data_test, dtype=torch.float32).to(device)
        labels_test = torch.tensor(labels_test, dtype=torch.float32).to(device)

        return {
            'data_train': data_train,
            'data_val': data_val,
            'data_test': data_test,
            'labels_train': labels_train,
            'labels_val': labels_val,
            'labels_test': labels_test,
        }

    training_data = get_training_data_mlp(
        spectra=preprocessed_spectra,
        reference_spectra=preprocessed_reference_spectra,
        held_back_metabolite=held_back_metabolite,
    )

    print([training_data[x].shape for x in training_data])
    print(len(training_data['data_train'][0]))

    sample_ratio = preprocessed_spectra[0]['ratios']
    data_length = len(training_data['data_train'][0])

    return data_length, device, sample_ratio, training_data


@app.cell(hide_code=True)
def _(data_length, device, mo, sample_ratio, training_data):
    mo.md(
        rf"""
    ## Training Data Preparation

    **Device Information:**

    - **Using Device:** {device}

    **Sample Information:**

    - **Sample Ratios (first spectrum):** {sample_ratio:}

    - **Feature Vector Length:** {data_length}

    **Tensor Shapes:**

    {chr(10).join([f"- **{key}:** {value.shape}" for key, value in training_data.items()])}
    """
    )
    return


@app.cell
def _(np, torch, tqdm, training_data):
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
                'total_parameters': sum(p.numel() for p in self.parameters())
            }

    class TransformerRegressor(nn.Module):
        def __init__(self, input_size, trial=None, **kwargs):
            """
            Transformer model for regression tasks.

            Args:
                input_size: Number of input features
                trial: Optuna trial object for hyperparameter suggestion
                **kwargs: Manual hyperparameter overrides
            """
            super(TransformerRegressor, self).__init__()

            # Get hyperparameters from trial or use provided values
            if trial is not None:
                self.d_model = int(trial.suggest_categorical('d_model', [64, 128, 256, 512]))
                self.nhead = int(trial.suggest_categorical('nhead', [4, 8, 16]))
                self.num_layers = int(trial.suggest_int('num_layers', 2, 8))
                self.dim_feedforward = int(trial.suggest_categorical('dim_feedforward', [256, 512, 1024, 2048]))
                self.dropout = trial.suggest_float('dropout', 0.1, 0.5)
                self.max_seq_len = int(trial.suggest_categorical('max_seq_len', [512, 1024, 2048]))
            else:
                # Default values or manual overrides
                self.d_model = kwargs.get('d_model', 256)
                self.nhead = kwargs.get('nhead', 8)
                self.num_layers = kwargs.get('num_layers', 4)
                self.dim_feedforward = kwargs.get('dim_feedforward', 512)
                self.dropout = kwargs.get('dropout', 0.1)
                self.max_seq_len = kwargs.get('max_seq_len', 1024)

            # Ensure nhead divides d_model evenly
            while self.d_model % self.nhead != 0:
                self.nhead = max(1, self.nhead - 1)

            # Calculate sequence length based on input size and d_model
            # Fix: Ensure we have at least one sequence element
            self.seq_len = max(1, min(input_size // self.d_model, self.max_seq_len))

            # Fix: If input is smaller than d_model, adjust d_model
            if input_size < self.d_model:
                self.d_model = input_size
                self.seq_len = 1
                # Re-adjust nhead to divide d_model evenly
                while self.d_model % self.nhead != 0:
                    self.nhead = max(1, self.nhead - 1)

            # Actual input size after reshaping - this should match the transformer input
            self.actual_input_size = self.seq_len * self.d_model

            # Fix: Input projection should map from actual input_size to our target size
            self.input_projection = nn.Linear(input_size, self.actual_input_size)

            # Positional encoding
            self.pos_encoding = PositionalEncoding(self.d_model, self.dropout, self.seq_len)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation='relu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=self.num_layers
            )

            # Output layers
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.output_projection = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model // 2, 1)
            )

            # Store architecture info
            self.architecture_info = {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'seq_len': self.seq_len,
                'max_seq_len': self.max_seq_len,
                'input_size': input_size,
                'actual_input_size': self.actual_input_size,
                'total_parameters': sum(p.numel() for p in self.parameters())
            }

        def forward(self, x):
            # Project input to match expected size
            x = self.input_projection(x)  # [batch_size, actual_input_size]

            # Reshape to sequence format
            batch_size = x.size(0)
            x = x.view(batch_size, self.seq_len, self.d_model)  # [batch_size, seq_len, d_model]

            # Add positional encoding
            x = self.pos_encoding(x)

            # Pass through transformer encoder
            x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]

            # Global average pooling over sequence dimension
            x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
            x = self.global_pool(x)  # [batch_size, d_model, 1]
            x = x.squeeze(-1)  # [batch_size, d_model]

            # Final output projection
            x = self.output_projection(x)  # [batch_size, 1]

            return x

        def get_architecture_info(self):
            """Return information about the model architecture"""
            return self.architecture_info


    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer input sequences"""

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)

        def forward(self, x):
            # x shape: [batch_size, seq_len, d_model]
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].transpose(0, 1)
            return self.dropout(x)

    def train_model(training_data, trial, model_type = 'mlp'):
        """
        Train a multi-task neural network for metabolite presence detection and concentration estimation.

        Architecture: Multi-layer perceptron with progressively smaller layers
        Tasks:
        1. Binary classification for metabolite presence
        2. Regression for concentration estimation

        Args:
            training_data: Dictionary with train/val/test splits
            trial: Optuna trial object for hyperparameter optimization

        Returns:
            Tuple of validation and test metrics
        """
        device = training_data['data_train'].device
        torch.cuda.empty_cache()

        # Remove loss_weight from hyperparameter search since we're using fixed formula
        max_epochs = 200  # Set a reasonable maximum
        batch_size = int(trial.suggest_float('batch_size', 10, 100, step=10))
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        # Remove loss_weight parameter - no longer needed
        input_length = len(training_data['data_train'][0])

        if model_type == 'transformer':
            model = TransformerRegressor(input_size=input_length, trial=trial).to(device)
        else:  # default to MLP
            model = MLPRegressor(input_size=input_length, trial=trial).to(device)

        # model.output_projection = nn.Sequential(
        #     nn.Linear(model.d_model, model.d_model // 2),
        #     nn.ReLU(),
        #     nn.Dropout(model.dropout),
        #     nn.Linear(model.d_model // 2, 2)  # 2 outputs: presence logit + concentration
        # ).to(device)

        # Extract data tensors (already on GPU)
        data_train = training_data['data_train']
        data_val = training_data['data_val']
        data_test = training_data['data_test']
        labels_train = training_data['labels_train']
        labels_val = training_data['labels_val']
        labels_test = training_data['labels_test']

        batch_start = torch.arange(0, len(data_train), batch_size)

        # Multi-task loss functions
        bce_loss = nn.BCEWithLogitsLoss()  # Binary cross-entropy for presence
        optimizer = optim.Adam(model.parameters(), lr=lr)

        def compute_loss(predictions, targets):
            """
            Compute weighted multi-task loss combining classification and regression.

            Returns:
                total_loss: Combined weighted loss using classification error + concentration MAE + RMSE
                classification_loss: BCE loss for presence prediction
                concentration_mae: MAE loss for concentration
                concentration_rmse: RMSE loss for concentration
            """
            presence_logits = predictions[:, 0]    # Raw logits for binary classification
            concentration_pred = predictions[:, 1]  # Concentration predictions

            presence_true = targets[:, 0]      # Ground truth presence (0 or 1)
            concentration_true = targets[:, 1]   # Ground truth concentration

            # Classification loss for presence detection
            classification_loss = bce_loss(presence_logits, presence_true)

            # Convert logits to probabilities for error calculation
            presence_pred = torch.sigmoid(presence_logits)
            presence_binary = (presence_pred > 0.5).float()
            classification_error = torch.mean((presence_binary != presence_true).float())

            # Regression losses for concentration (only when substance is present)
            present_mask = presence_true == 1
            if present_mask.sum() > 0:
                concentration_mae = torch.mean(
                    torch.abs(concentration_pred[present_mask] - concentration_true[present_mask])
                )
                concentration_rmse = torch.sqrt(
                    torch.mean(
                        (concentration_pred[present_mask] - concentration_true[present_mask]) ** 2
                    )
                )
            else:
                concentration_mae = torch.tensor(0.0, device=predictions.device)
                concentration_rmse = torch.tensor(0.0, device=predictions.device)

            # New combined loss formula: classification error * 0.5 + (0.5*MAE + 0.5*RMSE)
            total_loss = 0.5 * classification_error + 0.5 * (0.5 * concentration_mae + 0.5 * concentration_rmse)

            return total_loss, classification_loss, concentration_mae, concentration_rmse

        # Early stopping parameters - simplified and more reliable
        early_stop_patience = 15  # Number of epochs without improvement
        min_delta = 1e-6  # Minimum improvement to qualify as better
        validation_interval = 5  # Validate every 5 epochs instead of 10

        best_val_loss = np.inf
        epochs_without_improvement = 0
        best_weights = None

        for epoch in range(max_epochs):
            model.train()
            with tqdm.tqdm(
                batch_start, unit='batch', mininterval=0, disable=True
            ) as bar:
                bar.set_description(f'Epoch {epoch}')
                for start in bar:
                    # Mini-batch training
                    data_batch = data_train[start : start + batch_size]
                    labels_batch = labels_train[start : start + batch_size]
                    predictions = model(data_batch)
                    total_loss, class_loss, conc_mae, conc_rmse = compute_loss(
                        predictions, labels_batch
                    )

                    # Backpropagation
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    bar.set_postfix(
                        total_loss=float(total_loss),
                        class_loss=float(class_loss),
                        conc_mae=float(conc_mae),
                        conc_rmse=float(conc_rmse),
                    )

            # Validate more frequently for better early stopping
            if epoch % validation_interval == 0 or epoch == max_epochs - 1:
                model.eval()
                with torch.no_grad():
                    predictions = model(data_val)
                    val_loss, _, _, _ = compute_loss(predictions, labels_val)  # Fixed: expecting 4 values
                    val_loss = float(val_loss)

                    # Simple early stopping logic
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        best_weights = copy.deepcopy(model.state_dict())
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += validation_interval

                    # Early stopping check
                    if epochs_without_improvement >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch}: "
                              f"No improvement for {epochs_without_improvement} epochs")
                        break

        # Load best weights for final evaluation
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            # Validation metrics computation
            val_pred = model(data_val)
            val_presence_logits = val_pred[:, 0]
            val_concentration_pred = val_pred[:, 1]
            # Validation metrics computation
            val_pred = model(data_val)
            val_presence_logits = val_pred[:, 0]
            val_concentration_pred = val_pred[:, 1]
            val_presence_pred = torch.sigmoid(
                val_presence_logits
            )  # Convert to probabilities

            val_presence_true = labels_val[:, 0]
            val_concentration_true = labels_val[:, 1]

            # Classification metrics (presence detection)
            val_presence_binary = (val_presence_pred > 0.5).float()
            val_accuracy = torch.mean(
                (val_presence_binary == val_presence_true).float()
            )

            # Regression metrics (concentration estimation, only for present substances)
            present_mask = val_presence_true == 1
            if present_mask.sum() > 0:
                val_conc_mae = torch.mean(
                    torch.abs(
                        val_concentration_pred[present_mask]
                        - val_concentration_true[present_mask]
                    )
                )
                val_conc_rmse = torch.sqrt(
                    torch.mean(
                        (
                            val_concentration_pred[present_mask]
                            - val_concentration_true[present_mask]
                        )
                        ** 2
                    )
                )
                # R-squared coefficient of determination
                ss_res_val = torch.sum(
                    (
                        val_concentration_true[present_mask]
                        - val_concentration_pred[present_mask]
                    )
                    ** 2
                )
                ss_tot_val = torch.sum(
                    (
                        val_concentration_true[present_mask]
                        - torch.mean(val_concentration_true[present_mask])
                    )
                    ** 2
                )
                val_conc_r2 = 1 - (ss_res_val / ss_tot_val)
            else:
                val_conc_mae = torch.tensor(0.0)
                val_conc_rmse = torch.tensor(0.0)
                val_conc_r2 = torch.tensor(0.0)

            # Test metrics computation (same structure as validation)
            test_pred = model(data_test)
            test_presence_logits = test_pred[:, 0]
            test_concentration_pred = test_pred[:, 1]
            test_presence_pred = torch.sigmoid(test_presence_logits)

            test_presence_true = labels_test[:, 0]
            test_concentration_true = labels_test[:, 1]

            test_presence_binary = (test_presence_pred > 0.5).float()
            test_accuracy = torch.mean(
                (test_presence_binary == test_presence_true).float()
            )

            test_present_mask = test_presence_true == 1
            if test_present_mask.sum() > 0:
                test_conc_mae = torch.mean(
                    torch.abs(
                        test_concentration_pred[test_present_mask]
                        - test_concentration_true[test_present_mask]
                    )
                )
                test_conc_rmse = torch.sqrt(
                    torch.mean(
                        (
                            test_concentration_pred[test_present_mask]
                            - test_concentration_true[test_present_mask]
                        )
                        ** 2
                    )
                )
                ss_res_test = torch.sum(
                    (
                        test_concentration_true[test_present_mask]
                        - test_concentration_pred[test_present_mask]
                    )
                    ** 2
                )
                ss_tot_test = torch.sum(
                    (
                        test_concentration_true[test_present_mask]
                        - torch.mean(
                            test_concentration_true[test_present_mask]
                        )
                    )
                    ** 2
                )
                test_conc_r2 = 1 - (ss_res_test / ss_tot_test)
            else:
                test_conc_mae = torch.tensor(0.0)
                test_conc_rmse = torch.tensor(0.0)
                test_conc_r2 = torch.tensor(0.0)

        # Return validation metrics for optimization and test metrics for final evaluation
        return (
            float(val_accuracy),  # Primary metric for Optuna optimization
            float(val_conc_r2),
            float(val_conc_mae),
            float(val_conc_rmse),
            float(test_accuracy),  # Final test performance
            float(test_conc_r2),
            float(test_conc_mae),
            float(test_conc_rmse),
        )

    # Store device info for display
    device_info = training_data['data_train'].device
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


@app.cell(hide_code=True)
def _(mo, optuna, study):
    # Determine model type from best trial parameters
    model_type = 'Transformer' if 'd_model' in study.best_trial.params else 'MLP'

    # Create model-specific parameter display
    if model_type == 'Transformer':
        model_params_md = f"""
    **Transformer Architecture:**
    - **Model Dimension (d_model):** {study.best_trial.params['d_model']}
    - **Number of Attention Heads:** {study.best_trial.params['nhead']}
    - **Number of Encoder Layers:** {study.best_trial.params['num_layers']}
    - **Feedforward Dimension:** {study.best_trial.params['dim_feedforward']}
    - **Dropout Rate:** {study.best_trial.params['dropout']:.3f}
    - **Maximum Sequence Length:** {study.best_trial.params['max_seq_len']}

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
    - Test: Spectra containing held-back metabolite ({study.best_trial.user_attrs.get('held_back_metabolite', 'N/A')})

    **Total Trials Completed:** {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}
    """
    )
    return


@app.cell
def _(cache_dir, cache_key, tqdm, train_model, training_data, trials):
    import optuna
    from functools import partial

    def objective(training_data, trial, model_type='transformer'):
        """
        Optuna objective function for hyperparameter optimization.

        Optimizes the new combined loss: classification error * 0.5 + (0.5*MAE + 0.5*RMSE)

        Args:
            training_data: Training/validation/test splits
            trial: Optuna trial object for hyperparameter suggestions
            model_type: 'transformer' or 'mlp'

        Returns:
            Combined score for optimization (lower is better, so we negate it)
        """
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
        trial.set_user_attr('model_type', model_type)  # Store model type

        # Calculate classification error (1 - accuracy)
        val_classification_error = 1.0 - val_accuracy

        # Use the new optimization formula: classification error * 0.5 + (0.5*MAE + 0.5*RMSE)
        combined_score = 0.5 * val_classification_error + 0.5 * (0.5 * val_conc_mae + 0.5 * val_conc_rmse)

        # Return negative score since Optuna maximizes but we want to minimize the error
        return combined_score

    # Set model type here - change to 'mlp' to use MLP model
    MODEL_TYPE = 'mlp'  # or 'mlp'

    # Create or load existing Optuna study with persistent SQLite storage
    study = optuna.create_study(
        direction='minimize',  # Maximize the negative combined error (minimize error)
        study_name=cache_key,  # Use cache key for unique study identification
        storage=f'sqlite:///{cache_dir}/{cache_key}.db',  # Use cache key for database filename
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


if __name__ == "__main__":
    app.run()
