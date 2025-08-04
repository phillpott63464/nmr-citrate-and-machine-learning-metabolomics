import marimo

__generated_with = "0.14.13"
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

    count = 10
    trials = 10
    combo_number = 10
    notebook_name = 'randomisation_hold_back'
    cache_dir = f"./data_cache/{notebook_name}"
    return cache_dir, combo_number, count, trials


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
def _(cache_dir, combo_number, count):
    from morgan.createTrainingData import createTrainingData
    import morgan
    import numpy as np
    import tqdm
    import itertools
    import multiprocessing as mp
    import os
    import pickle
    from pathlib import Path
    import random

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

    # Extract spectrum IDs for reference spectra generation
    referenceSpectrumIds = [
        substanceDict[substance][-1] for substance in substanceDict
    ]

    substances = list(substanceDict.keys())

    # Generate all possible combinations of substances (1 to n substances)
    # This creates training data for different metabolite mixtures
    combinations = []
    for r in range(4, len(substances) + 1):
        for combo in itertools.combinations(substances, r):
            combo_dict = {
                substance: substanceDict[substance] for substance in combo
            }
            combinations.append(combo_dict)

    combinations = random.sample(combinations, combo_number)

    print(len(combinations))

    # Extract spectrum IDs for each combination
    substanceSpectrumIds = [
        [combination[substance][-1] for substance in combination]
        for combination in combinations
    ]

    # Save/load functions for data persistence
    def save_spectra_data(spectra, held_back_metabolite, filename):
        """Save generated spectra data and held-back metabolite to pickle file"""
        os.makedirs(cache_dir, exist_ok=True)
        filepath = f"{cache_dir}/{filename}.pkl"

        data_to_save = {
            'spectra': spectra,
            'held_back_metabolite': held_back_metabolite
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Saved {len(spectra)} spectra and held-back metabolite '{held_back_metabolite}' to {filepath}")

    def load_spectra_data(filename):
        """Load generated spectra data and held-back metabolite from pickle file"""
        filepath = f"{cache_dir}/{filename}.pkl"

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Handle both old and new cache formats
            if isinstance(data, list):
                # Old format - just spectra
                print(f"Loaded {len(data)} spectra from {filepath} (old format)")
                return data, None
            else:
                # New format - spectra + held_back_metabolite
                spectra = data['spectra']
                held_back_metabolite = data['held_back_metabolite']
                print(f"Loaded {len(spectra)} spectra and held-back metabolite '{held_back_metabolite}' from {filepath}")
                return spectra, held_back_metabolite
        return None, None

    def generate_cache_key(substanceDict, combinations, count):
        """Generate unique cache key based on parameters"""
        substance_key = "_".join(sorted(substanceDict.keys()))
        combo_key = f"combos_{len(combinations)}"
        count_key = f"count_{count}"
        return f"spectra_{substance_key}_{combo_key}_{count_key}"

    # Generate cache key for current configuration
    cache_key = generate_cache_key(substanceDict, combinations, count)

    # Try to load existing data first
    spectra, held_back_metabolite = load_spectra_data(cache_key)

    if spectra is None:
        print("No cached data found. Generating new spectra...")

        # Select random metabolite to hold back for testing
        held_back_metabolite = random.choice(list(substanceDict.keys()))
        print(f"Selected '{held_back_metabolite}' as held-back metabolite for testing")

        def create_batch_data(substances_and_count):
            """Generate training data batch for specific substance combination with random scaling"""
            substances, sample_count = substances_and_count
            return createTrainingData(
                substanceSpectrumIds=substances,
                sampleNumber=sample_count,
                rondomlyScaleSubstances=True,  # Randomize concentrations for training diversity
                referenceSubstanceSpectrumId='tsp',
            )

        # Prepare multiprocessing arguments - one batch per substance combination
        mp_args = [(substances, count) for substances in substanceSpectrumIds]

        # Use multiprocessing to parallelize data generation across CPU cores
        num_processes = max(1, mp.cpu_count() - 1)
        print(f'Using {num_processes} processes for data generation')

        batch_data = []
        if len(mp_args) > 1:
            with mp.Pool(processes=num_processes) as pool:
                batch_data = list(
                    tqdm.tqdm(pool.imap_unordered(create_batch_data, mp_args), total=len(mp_args))
                )
        else:
            batch_data = [create_batch_data(mp_args[0])]

        print(f'Generated {len(batch_data)} batches')

        # Reshape batch data into individual spectrum samples
        # Each spectrum contains intensities, positions, scales, and component information
        spectra = []
        labels = []

        for batch in batch_data:
            for i in range(count):
                # Extract individual sample scales (concentrations) from batch
                sample_scales = {
                    key: [values[i]] for key, values in batch['scales'].items()
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
        save_spectra_data(spectra, held_back_metabolite, cache_key)
    else:
        print("Using cached spectra data.")
        if held_back_metabolite is None:
            # If old cache format, select and save new held-back metabolite
            held_back_metabolite = random.choice(list(substanceDict.keys()))
            print(f"Cache missing held-back metabolite. Selected '{held_back_metabolite}' and updating cache...")
            save_spectra_data(spectra, held_back_metabolite, cache_key)

    # Display sample information for verification
    print(''.join(f"{x['scales']}\n'" for x in spectra[:5]))
    print(spectra[0]['intensities'].shape)
    print(spectra[0]['positions'].shape)
    print(spectra[0]['components'].shape)

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
        referenceSpectrumIds,
        sample_scales_preview,
        spectra,
        substanceDict,
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
def _(createTrainingData, plt, referenceSpectrumIds, spectra, substanceDict):
    # Generate pure component reference spectra (no random scaling)
    # These serve as templates for identifying substances in mixtures
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
    from scipy.signal import resample
    from scipy.interpolate import interp1d

    def preprocess_peaks(
        intensities,
        positions,
        ranges=[[-100, 100]],
        baseline_distortion=False,
        downsample=0,
    ):
        """
        Extract and preprocess spectral regions of interest.

        Args:
            intensities: Spectral intensity data
            positions: Chemical shift positions (ppm)
            ranges: List of [min, max] ppm ranges to extract
            baseline_distortion: Add realistic baseline drift
            downsample: Target number of points for downsampling
        """
        # Extract data points within specified ppm ranges
        indices_range = []
        for range_item in ranges:
            indices_range.append(
                np.where(
                    (positions >= range_item[0]) & (positions <= range_item[1])
                )[0]
            )

        new_intensities = []
        new_positions = []

        for range_indices in indices_range:
            temp_intensities = intensities[range_indices]
            temp_positions = positions[range_indices]

            # Add realistic baseline distortion to simulate experimental conditions
            if baseline_distortion:
                if len(temp_positions) > 1:
                    # Normalize positions for consistent baseline calculation
                    x_normalized = (
                        temp_positions - np.min(temp_positions)
                    ) / (np.max(temp_positions) - np.min(temp_positions))
                    # Sinusoidal baseline with 2% amplitude
                    baseline = 0.02 * np.sin(0.5 * np.pi * x_normalized)
                    temp_intensities = temp_intensities + baseline

            new_intensities = np.concatenate(
                [new_intensities, temp_intensities]
            )
            new_positions = np.concatenate([new_positions, temp_positions])

        # Downsample data to reduce computational complexity while preserving peak shapes
        if downsample > 0:
            # Sort by position to ensure proper binning
            sorted_indices = np.argsort(new_positions)
            new_positions = new_positions[sorted_indices]
            new_intensities = new_intensities[sorted_indices]

            # Create uniform grid for downsampling
            min_pos, max_pos = new_positions.min(), new_positions.max()
            bin_edges = np.linspace(min_pos, max_pos, downsample + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            downsampled_intensities = np.zeros(downsample)

            # Integrate intensities within each bin to preserve peak areas
            for i in range(downsample):
                mask = (new_positions >= bin_edges[i]) & (
                    new_positions < bin_edges[i + 1]
                )
                if np.any(mask):
                    bin_width = bin_edges[i + 1] - bin_edges[i]
                    if len(new_positions[mask]) > 1:
                        # Use trapezoidal integration normalized by bin width
                        area = np.trapz(
                            new_intensities[mask], new_positions[mask]
                        )
                        downsampled_intensities[i] = area / bin_width
                    else:
                        downsampled_intensities[i] = new_intensities[mask][0]
                else:
                    downsampled_intensities[i] = 0.0

            new_intensities = downsampled_intensities
            new_positions = bin_centers

        return new_positions, new_intensities

    def preprocess_ratio(scales, substanceDict):
        """Calculate concentration ratios relative to internal standard (tsp)"""
        ratios = {
            substance: scales[substance][0] / scales['tsp'][0]
            for substance in scales
        }

        return ratios

    def preprocess_spectra(
        spectra, ranges, substanceDict, baseline_distortion=False, downsample=0
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
    downsample = int(2048)  # Target resolution for ML model

    def process_single_spectrum(spectrum):
        """Worker function for parallel spectrum preprocessing"""
        return preprocess_spectra(
            spectra=spectrum,
            ranges=ranges,
            substanceDict=substanceDict,
            baseline_distortion=baseline_distortion,
            downsample=downsample,
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
                preprocessed_spectra[0]['positions'],
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
            preprocessed_spectra[graphcounter2]['positions'],
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
        spectra, reference_spectra, held_back_metabolite, train_ratio=0.7, val_ratio=0.15, axes=0
    ):
        """
        Prepare training data for multi-task learning model.

        Creates paired examples of (mixed spectrum + reference) -> (presence, concentration)
        Each training example consists of:
        - Mixed spectrum intensities and positions
        - Pure component reference spectrum
        - Labels: [presence (0/1), concentration ratio]

        Args:
            spectra: List of mixed spectrum dictionaries
            reference_spectra: Dictionary of pure component references
            held_back_metabolite: Specific metabolite to hold back for testing
            train_ratio: Training set fraction
            val_ratio: Validation set fraction
            axes: Unused parameter (legacy)

        Returns:
            Dictionary with train/val/test splits for features and labels
        """
        data = []
        labels = []
        data_test = []
        labels_test = []

        # Use the consistently cached held-back metabolite
        held_back_key = substanceDict[held_back_metabolite][0]
        print(f"Using held-back metabolite: {held_back_metabolite} (key: {held_back_key})")

        # Create training pairs: (mixed_spectrum + reference) -> (presence, concentration)
        for spectrum in spectra:
            for substance in reference_spectra:
                # Concatenate mixed spectrum with reference for comparison
                temp_data = np.concatenate(
                        [
                            spectrum[
                                'intensities'
                            ],  # Mixed spectrum intensities
                            spectrum['positions'],  # Chemical shift positions
                            reference_spectra[
                                substance
                            ],  # Pure reference spectrum
                        ]
                    )

                # Multi-task labels: presence detection + concentration estimation
                if substance in spectrum['ratios']:
                    temp_label = [1, spectrum['ratios'][substance]]  # Present with ratio
                else:
                    temp_label = [0, 0]  # Absent

                if substance == held_back_key: # Use the cached held-back metabolite
                    data_test.append(temp_data)
                    labels_test.append(temp_label)
                else: # Otherwise, it's training/validation data
                    data.append(temp_data)
                    labels.append(temp_label)

        data = np.array(data)

        data_train, data_val, labels_train, labels_val = train_test_split(
            data, labels, train_size=train_ratio, shuffle=True, random_state=42
        )

        # Convert to GPU tensors for efficient training
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

    def train_mlp_model(training_data, trial):
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

        # Hyperparameter suggestions from Optuna
        n_epochs = int(trial.suggest_float('n_epochs', 10, 100, step=10))
        batch_size = int(trial.suggest_float('batch_size', 10, 100, step=10))
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        div_size = trial.suggest_float('div_size', 2, 10, step=1)
        loss_weight = trial.suggest_float('loss_weight', 0.1, 10.0)

        # Progressive layer size reduction based on division factor
        a = len(training_data['data_train'][0])  # Input feature dimension
        b = int(a / div_size)
        c = int(b / div_size)
        d = int(c / div_size)
        e = int(d / div_size)

        # Multi-layer perceptron with ReLU activations
        model = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
            nn.Linear(b, c),
            nn.ReLU(),
            nn.Linear(c, d),
            nn.ReLU(),
            nn.Linear(d, e),
            nn.ReLU(),
            nn.Linear(e, 2),  # Output: [presence_logit, concentration]
        ).to(device)

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
        mse_loss = nn.MSELoss(
            reduction='none'
        )  # Mean squared error for concentration
        optimizer = optim.Adam(model.parameters(), lr=lr)

        def compute_loss(predictions, targets):
            """
            Compute weighted multi-task loss combining classification and regression.

            Returns:
                total_loss: Combined weighted loss
                classification_loss: BCE loss for presence prediction
                weighted_concentration_loss: MSE loss for concentration (weighted by presence)
            """
            presence_logits = predictions[
                :, 0
            ]    # Raw logits for binary classification
            concentration_pred = predictions[:, 1]  # Concentration predictions

            presence_true = targets[:, 0]      # Ground truth presence (0 or 1)
            concentration_true = targets[:, 1]   # Ground truth concentration

            # Classification loss for presence detection
            classification_loss = bce_loss(presence_logits, presence_true)

            # Regression loss for concentration (only when substance is present)
            concentration_loss = mse_loss(
                concentration_pred, concentration_true
            )
            # Weight by presence to avoid penalizing concentration errors when absent
            weighted_concentration_loss = torch.mean(
                concentration_loss * presence_true
            )

            # Combine losses with learnable weighting
            total_loss = (
                classification_loss + loss_weight * weighted_concentration_loss
            )

            return total_loss, classification_loss, weighted_concentration_loss

        # Training loop with early stopping based on validation loss
        best_loss = np.inf
        best_weights = None
        history = []

        for epoch in range(n_epochs):
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
                    total_loss, class_loss, conc_loss = compute_loss(
                        predictions, labels_batch
                    )

                    # Backpropagation
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    bar.set_postfix(
                        total_loss=float(total_loss),
                        class_loss=float(class_loss),
                        conc_loss=float(conc_loss),
                    )

            # Validation evaluation for early stopping
            model.eval()
            with torch.no_grad():
                predictions = model(data_val)
                val_loss, _, _ = compute_loss(predictions, labels_val)
                val_loss = float(val_loss)
                history.append(val_loss)

                # Save best model weights
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = copy.deepcopy(model.state_dict())

        # Load best weights for final evaluation
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
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
    return device_info, gpu_name, train_mlp_model


@app.cell(hide_code=True)
def _(device_info, gpu_name, mo):
    mo.md(
        rf"""
    ## Model Training Setup

    **Hardware Configuration:**

    - **Training Device:** {device_info} / {gpu_name}

    **Model Architecture:** Multi-Layer Perceptron (MLP)

    - Sequential neural network with ReLU activations

    - Variable width controlled by division size hyperparameter

    - Final output: Single regression value
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, optuna, study):
    mo.md(
        f"""
    ## Hyperparameter Optimization Results

    **Best Trial Performance (Validation Set):**

    - **Combined Score: {study.best_trial.value:.4f}** (0.5 * Accuracy + 0.5 * R², optimized metric)
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

    - **Number of Epochs:** {study.best_trial.params['n_epochs']:.0f}
    - **Batch Size:** {study.best_trial.params['batch_size']:.0f}
    - **Learning Rate:** {study.best_trial.params['lr']:.2e}
    - **Division Size:** {study.best_trial.params['div_size']:.0f} (controls network width - smaller values = wider layers)
    - **Loss Weight:** {study.best_trial.params['loss_weight']:.2f} (weighting for concentration vs presence loss)

    **Model Architecture:**

    Input size → {int(study.best_trial.params['div_size'])} divisions → ... → 2 outputs (presence + concentration)

    **Multi-Task Learning:**

    - **Task 1:** Binary classification for substance presence (BCEWithLogitsLoss)
    - **Task 2:** Regression for concentration prediction (MSE, weighted by presence)
    - **Combined Loss:** Classification + {study.best_trial.params['loss_weight']:.2f} × Concentration Loss

    **Data Split:**

    - Training: 70% 
    - Validation: 15% (used for hyperparameter optimization)
    - Test: 15% (held out for final evaluation)

    **Total Trials Completed:** {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}
    """
    )
    return


@app.cell
def _(cache_dir, cache_key, tqdm, train_mlp_model, training_data, trials):
    import optuna
    from functools import partial

    def objective(training_data, trial):
        """
        Optuna objective function for hyperparameter optimization.

        Optimizes a weighted combination of classification accuracy and regression R².
        This balances performance on both tasks in the multi-task learning setup.

        Args:
            training_data: Training/validation/test splits
            trial: Optuna trial object for hyperparameter suggestions

        Returns:
            Combined score for optimization (higher is better)
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
        ) = train_mlp_model(training_data, trial)

        # Store all metrics in trial for later analysis
        trial.set_user_attr('val_accuracy', val_accuracy)
        trial.set_user_attr('val_conc_r2', val_conc_r2)
        trial.set_user_attr('val_conc_mae', val_conc_mae)
        trial.set_user_attr('val_conc_rmse', val_conc_rmse)
        trial.set_user_attr('test_accuracy', test_accuracy)
        trial.set_user_attr('test_conc_r2', test_conc_r2)
        trial.set_user_attr('test_conc_mae', test_conc_mae)
        trial.set_user_attr('test_conc_rmse', test_conc_rmse)

        # Optimize weighted combination of both tasks (equal weighting)
        combined_score = 0.5 * val_accuracy + 0.5 * val_conc_r2
        return combined_score

    # Create or load existing Optuna study with persistent SQLite storage
    study = optuna.create_study(
        direction='maximize',  # Maximize the combined score
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
                partial(objective, training_data),
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
