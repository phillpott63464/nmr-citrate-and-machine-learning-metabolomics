import marimo # type: ignore

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    """Initial imports and hardware detection"""
    import marimo as mo # type: ignore
    import torch # type: ignore

    # Check hardware capabilities for GPU acceleration
    hip_version = torch.version.hip
    cuda_built = torch.backends.cuda.is_built()
    gpu_count = torch.cuda.device_count()
    return cuda_built, gpu_count, hip_version, mo, torch


@app.cell(hide_code=True)
def _(cuda_built, gpu_count, hip_version, mo):
    mo.md(
        f"""
    # NMR Spectral Analysis with Hilbert Transform

    ## Hardware Configuration

    This notebook performs NMR spectral analysis using machine learning models with optional Hilbert transform preprocessing.

    **GPU Setup Information:**

    - **HIP Version:** {hip_version}
    - **CUDA Built:** {cuda_built}
    - **GPU Device Count:** {gpu_count}

    The system will automatically use GPU acceleration if available, falling back to CPU processing otherwise.
    """
    )
    return


@app.cell
def _():
    """Configuration parameters for the entire analysis pipeline"""

    # Experiment parameters
    count = 100                    # Number of samples per metabolite combination
    trials = 100                  # Number of hyperparameter optimization trialss
    combo_number = 30             # Number of random metabolite combinations to generate
    notebook_name = 'randomisation_hold_back'  # Cache directory identifier

    # Model configuration
    MODEL_TYPE = 'transformer'            # Model architecture: 'mlp', 'transformer', or 'ensemble'
    downsample = None             # Target resolution for ML model (None = no downsampling)
    reverse = False                # Apply Hilbert transform (time domain analysis)
    ranged = True

    # Smart cache directory structure
    base_cache_dir = f'./data_cache/{notebook_name}'
    raw_data_dir = f'{base_cache_dir}/raw_data'  # Only depends on substances & generation params
    processed_data_dir = f'{base_cache_dir}/processed/{"time" if reverse else "freq"}/{"ranged" if ranged else "unranged"}{downsample}'  # Depends on preprocessing
    model_cache_dir = f'{processed_data_dir}/models/{MODEL_TYPE}'  # Depends on model type + processed data

    # Legacy cache_dir for backward compatibility
    cache_dir = f'{base_cache_dir}/{MODEL_TYPE}/{"time" if reverse else "freq"}/{downsample}'

    # NMR metabolite database mapping (substance name -> spectrum ID + chemical shift range)
    substanceDict = {
        'Citric acid': ['SP:3368', [2.2, 2.8]],
        'Succinic acid': ['SP:3211', [2.0, 2.5]],
        'Maleic acid': ['SP:3110', [6.0, 6.5]],
        'Lactic acid': ['SP:3675', [1.2, 1.5]],
        'L-Methionine': ['SP:3509', [2.0, 2.5]],
        'L-Proline': ['SP:3406', [2.0, 3.0]],
        'L-Phenylalanine': ['SP:3507', [6.5, 7.5]],
        'L-Serine': ['SP:3732', [3.5, 4.5]],
        'L-Threonine': ['SP:3437', [3.5, 4.5]],
        'L-Tryptophan': ['SP:3455', [7.0, 8.0]],
        'L-Tyrosine': ['SP:3464', [6.5, 7.5]],
        'L-Valine': ['SP:3490', [0.9, 1.5]],
        'Glycine': ['SP:3682', [3.5, 4.0]],
    }

    return (
        MODEL_TYPE,
        combo_number,
        count,
        downsample,
        model_cache_dir,
        processed_data_dir,
        ranged,
        raw_data_dir,
        reverse,
        substanceDict,
        trials,
    )


@app.cell(hide_code=True)
def _(mo, substanceDict):
    mo.md(
        f"""
    ## Experimental Configuration

    **Metabolite Panel:**

    This analysis uses {len(substanceDict)} metabolites commonly found in biological samples:

    {chr(10).join([f"- **{name}:** {ids[0]}" for name, ids in substanceDict.items()])}

    **Data Generation Strategy:**

    - **Hold-back validation:** Two metabolites are randomly selected and excluded from training
    - **Mixture complexity:** Combinations range from {len(substanceDict)//3} to {len(substanceDict)} metabolites
    - **Concentration variability:** Random scaling applied to simulate biological variation
    - **Reference standard:** TSP (trimethylsilyl propanoic acid) used for normalization

    This approach tests the model's ability to detect and quantify metabolites it has never seen during training.
    """
    )
    return


@app.cell
def _():
    """Import data generation dependencies"""
    from morgan.createTrainingData import createTrainingData
    import morgan
    import numpy as np # type: ignore
    import tqdm # type: ignore
    import itertools
    import random
    import pickle
    from pathlib import Path
    import os

    return createTrainingData, itertools, np, os, pickle, random, tqdm


@app.cell
def _(os, pickle, random, raw_data_dir):
    """Data persistence utilities for caching generated spectra with smart directory structure"""

    def save_spectra_data(spectra, held_back_metabolites, combinations, filename):
        """
        Save generated spectra data to raw data cache (model-independent)

        Args:
            spectra: List of spectrum dictionaries with intensities, positions, scales
            held_back_metabolites: List of metabolites excluded from training
            combinations: List of metabolite combinations used
            filename: Cache file identifier
        """
        os.makedirs(raw_data_dir, exist_ok=True)
        filepath = f'{raw_data_dir}/{filename}.pkl'

        data_to_save = {
            'spectra': spectra,
            'held_back_metabolites': held_back_metabolites,
            'combinations': combinations,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(
            f"Saved {len(spectra)} spectra, held-back metabolites '{[x for x in held_back_metabolites]}', "
            f"and {len(combinations)} combinations to {filepath}"
        )

    def load_spectra_data(filename, substanceDict):
        """
        Load cached spectra data from raw data cache

        Returns:
            tuple: (spectra, held_back_metabolites, combinations) or (None, None, None) if not found
        """
        filepath = f'{raw_data_dir}/{filename}.pkl'

        if not os.path.exists(filepath):
            return None, None, None

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Handle legacy cache formats
        if isinstance(data, list):
            # Old format - just spectra
            print(f'Loaded {len(data)} spectra from {filepath} (legacy format)')
            return data, None, None

        # Extract data with fallback for older formats
        spectra = data['spectra']

        # Handle held_back_metabolites field variations
        held_back_metabolites = data.get('held_back_metabolites', 
                                       data.get('held_back_metabolite', None))

        # Ensure held_back_metabolites is a list
        if isinstance(held_back_metabolites, str):
            held_back_metabolites = [
                held_back_metabolites,
                random.choice(list(substanceDict.keys())),
            ]

        combinations = data.get('combinations', None)

        format_type = "full" if combinations is not None else "partial"
        print(f"Loaded {len(spectra)} spectra, held-back metabolites '{held_back_metabolites}', "
              f"and {len(combinations) if combinations else 0} combinations from {filepath} ({format_type} format)")

        return spectra, held_back_metabolites, combinations

    import hashlib

    def generate_raw_cache_key(substanceDict, combo_number, count):
        """Generate cache key for raw data (independent of model/preprocessing)"""
        substance_key = '_'.join(sorted(substanceDict.keys()))
        combo_key = f'combos_{combo_number}'
        count_key = f'count_{count}'

        # Combine all parts into a single string
        raw_key = f'raw_spectra_{substance_key}_{combo_key}_{count_key}'

        # Generate a hash of the combined string
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def generate_processed_cache_key(raw_cache_key, downsample, reverse):
        """Generate cache key for processed datasets"""
        processing_key = f'{"time" if reverse else "freq"}_{downsample}'

        # Combine the raw cache key with the processing key
        processed_key = f'processed_{raw_cache_key}_{processing_key}'

        # Generate a hash of the combined string
        return hashlib.sha256(processed_key.encode()).hexdigest()


    return (
        generate_processed_cache_key,
        generate_raw_cache_key,
        load_spectra_data,
        save_spectra_data,
    )


@app.cell
def _(createTrainingData):
    """NMR spectrum generation utilities"""

    def create_batch_data(substances_and_count):
        """
        Generate training data batch for specific substance combination

        Args:
            substances_and_count: Tuple of (substance_spectrum_ids, sample_count)

        Returns:
            dict: Batch containing intensities, positions, scales, and components
        """
        substances, sample_count = substances_and_count
        return createTrainingData(
            substanceSpectrumIds=substances,
            sampleNumber=sample_count,
            rondomlyScaleSubstances=True,  # Enable concentration randomization
            referenceSubstanceSpectrumId='tsp',  # Internal standard
        )

    return (create_batch_data,)


@app.cell
def _(
    combo_number,
    count,
    create_batch_data,
    generate_raw_cache_key,
    itertools,
    load_spectra_data,
    random,
    save_spectra_data,
    substanceDict,
    tqdm,
):
    """Main data generation pipeline with smart caching"""

    def check_loaded_data(spectra, held_back_metabolites, combinations):
        """
        Generate new training data if not cached, otherwise use existing data

        Returns:
            tuple: (spectra, held_back_metabolites, combinations)
        """
        if spectra is None:
            print('No cached raw data found. Generating new spectra...')

            # Generate all possible metabolite combinations (complexity: 4+ substances)
            substances = list(substanceDict.keys())
            all_combinations = []

            for r in range(len(substances) // 3, len(substances) + 1):
                for combo in itertools.combinations(substances, r):
                    combo_dict = {
                        substance: substanceDict[substance]
                        for substance in combo
                    }
                    all_combinations.append(combo_dict)

            # Randomly sample combinations to manage computational load
            if combo_number is not None:
                combinations = random.sample(all_combinations, combo_number)
            else:
                combinations = all_combinations
            print(f'Generated {len(combinations)} random combinations')

            # Select two metabolites for hold-back validation
            held_back_metabolites = random.sample(list(substanceDict.keys()), 2)
            print(f"Selected '{held_back_metabolites}' as held-back metabolites for testing")

            # Extract spectrum IDs for NMR simulation
            substanceSpectrumIds = [
                [combination[substance][0] for substance in combination]
                for combination in combinations
            ]

            # Prepare batch processing arguments
            mp_args = [
                (substances, count) for substances in substanceSpectrumIds
            ]

            # Sequential processing with progress tracking
            print(f'Generating {len(mp_args)} batches sequentially...')
            batch_data = []
            for arg in tqdm.tqdm(mp_args, desc="Generating batches"):
                batch_data.append(create_batch_data(arg))

            print(f'Generated {len(batch_data)} batches')

            # Reshape batch data into individual spectrum samples
            spectra = []
            for batch in batch_data:
                for i in range(count):
                    # Extract individual sample from batch
                    sample_scales = {
                        key: [values[i]]
                        for key, values in batch['scales'].items()
                    }

                    spectrum = {
                        'scales': sample_scales,                    # Metabolite concentrations
                        'intensities': batch['intensities'][i:i+1], # NMR signal intensities
                        'positions': batch['positions'],            # Chemical shift positions (ppm)
                        'components': batch['components'],          # Individual metabolite spectra
                    }
                    spectra.append(spectrum)

            # Cache generated data for future use
            save_spectra_data(spectra, held_back_metabolites, combinations, raw_cache_key)
        else:
            print('Using cached raw spectra data.')
            if held_back_metabolites is None:
                # Legacy cache: select new held-back metabolites
                held_back_metabolites = random.sample(list(substanceDict.keys()), 2)
                print(f"Cache missing held-back metabolites. Selected '{held_back_metabolites}' and updating cache...")
                save_spectra_data(spectra, held_back_metabolites, combinations, raw_cache_key)
            print(f'Using {len(combinations)} combinations from cache')

        return spectra, held_back_metabolites, combinations

    # Generate cache key for raw data only (model-independent)
    raw_cache_key = generate_raw_cache_key(substanceDict, combo_number, count)
    spectra, held_back_metabolites, combinations = load_spectra_data(raw_cache_key, substanceDict)
    spectra, held_back_metabolites, combinations = check_loaded_data(
        spectra, held_back_metabolites, combinations
    )

    # Extract spectrum IDs for downstream processing
    substanceSpectrumIds = [
        [combination[substance][0] for substance in combination]
        for combination in combinations
    ]

    print(f'Total combinations: {len(combinations)}')
    print('Sample scales preview:')
    print(''.join(f"{x['scales']}\n" for x in spectra[:5]))
    print(f"Intensities shape: {spectra[0]['intensities'].shape}")
    print(f"Positions shape: {spectra[0]['positions'].shape}")
    print(f"Components shape: {spectra[0]['components'].shape}")

    return combinations, held_back_metabolites, raw_cache_key, spectra


@app.cell(hide_code=True)
def _(combinations, count, held_back_metabolites, mo, spectra):
    mo.md(
        f"""
    ## Data Generation Results

    **Successfully generated {count} samples for each of {len(combinations)} metabolite combinations**

    **Hold-back Validation Setup:**

    - **Test metabolite:** {held_back_metabolites[0]} (completely excluded from training)
    - **Validation metabolite:** {held_back_metabolites[1]} (used for hyperparameter tuning)

    **Data Structure:**

    - **Total spectra:** {len(spectra)}
    - **Intensities shape:** {spectra[0]['intensities'].shape} (NMR signal data)
    - **Positions shape:** {spectra[0]['positions'].shape} (chemical shift scale in ppm)
    - **Components shape:** {spectra[0]['components'].shape} (individual metabolite contributions)

    **Sample Concentration Data (first 5 spectra):**
    ```
    {chr(10).join([f"Sample {i+1}: {spectrum['scales']}" for i, spectrum in enumerate(spectra[:5])])}
    ```

    Each spectrum contains:

    - **Scales:** Relative concentrations of each metabolite (normalized to TSP internal standard)
    - **Intensities:** Combined NMR spectrum from all metabolites
    - **Positions:** Chemical shift positions in parts per million (ppm)
    - **Components:** Individual metabolite spectra for reference
    """
    )
    return


@app.cell
def _(spectra):
    """Generate sample spectrum visualizations"""
    import matplotlib.pyplot as plt # type: ignore

    print(f"Total spectra available: {len(spectra)}")
    graph_count = 3  # 3x3 grid of sample spectra

    # Create visualization grid showing diverse sample spectra
    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for graphcounter in range(1, graph_count**2 + 1):
        plt.subplot(graph_count, graph_count, graphcounter)
        plt.plot(
            spectra[graphcounter]['positions'],
            spectra[graphcounter]['intensities'][0],
        )
        plt.title(f'Sample {graphcounter}')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')

    plt.tight_layout()
    spectrafigures = plt.gca()
    return graph_count, plt, spectrafigures


@app.cell(hide_code=True)
def _(mo, spectra, spectrafigures):
    mo.md(
        f"""
    ## Spectral Data Visualization

    **Generated {len(spectra)} NMR spectra with varying metabolite compositions and concentrations**

    The plots below show sample spectra from the generated dataset. Each spectrum represents a unique combination of metabolites at different concentrations, simulating the complexity found in real biological samples.

    **Key Features:**

    - **Peak diversity:** Different metabolites contribute characteristic peaks at specific chemical shifts
    - **Intensity variation:** Random concentration scaling creates realistic amplitude differences
    - **Baseline effects:** Simulated experimental artifacts for model robustness

    {mo.as_html(spectrafigures)}

    These spectra serve as training data for machine learning models to learn metabolite identification and quantification patterns.
    """
    )
    return


@app.cell
def _(createTrainingData, substanceDict):
    def _():
        """Generate pure component reference spectra for metabolite identification"""

        # Extract individual metabolite spectrum IDs
        referenceSpectrumIds = [
            substanceDict[substance][0] for substance in substanceDict
        ]

        # Generate pure component spectra (no concentration randomization)
        reference_spectra_raw = createTrainingData(
            substanceSpectrumIds=referenceSpectrumIds,
            sampleNumber=1,
            rondomlyScaleSubstances=False,  # Preserve original intensities
        )

        # Map substance names to their reference spectra for easy lookup
        reference_spectra = {
            substanceDict[substance][0]: [reference_spectra_raw['components'][index], reference_spectra_raw['scales'][substanceDict[substance][0]]]
            for index, substance in enumerate(substanceDict)
        }

        print("Generated reference spectra for metabolite identification:")
        for substance, spectrum_id in reference_spectra.items():
            print(f"  {substance}: {len(spectrum_id[0])} data points")

        return reference_spectra

    reference_spectra = _()

    print(reference_spectra['SP:3368'][1])
    return (reference_spectra,)


@app.cell
def _(plt, reference_spectra, spectra, substanceDict):
    def _():
        """Visualize reference spectra for each metabolite"""

        plt.figure(figsize=(12, 8))

        # Plot reference spectrum for each metabolite
        for substance in substanceDict:
            spectrum_id = substanceDict[substance][0]
            plt.plot(
                spectra[0]['positions'],
                reference_spectra[spectrum_id][0],
                label=substance,
                alpha=0.7
            )

        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title('Reference Spectra for Individual Metabolites')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gca()


    referencefigure = _()
    return (referencefigure,)


@app.cell(hide_code=True)
def _(mo, referencefigure, substanceDict):
    mo.md(
        f"""
    ## Reference Spectra Generation

    **Pure component spectra extracted for {len(substanceDict)} metabolites**

    These reference spectra serve as templates for the machine learning model to identify individual metabolites within complex mixtures.

    **Reference Spectrum Characteristics:**

    - **No concentration scaling:** Original intensities preserved for consistent identification
    - **Unique fingerprints:** Each metabolite has characteristic peak patterns
    - **Chemical shift assignments:** Peaks appear at specific ppm values based on molecular structure

    {mo.as_html(referencefigure)}

    **Application in Machine Learning:**

    The model receives both the mixture spectrum and a reference spectrum as input, learning to:

    1. **Detect presence:** Binary classification of whether the reference metabolite exists in the mixture
    2. **Quantify concentration:** Regression to predict relative concentration if present

    This approach enables metabolite-specific analysis where the model can be queried for any metabolite of interest.
    """
    )
    return


@app.cell
def _():
    """Import preprocessing dependencies"""
    import multiprocessing as mp
    from scipy.signal import resample, hilbert # type: ignore
    from scipy.interpolate import interp1d # type: ignore
    from scipy.fft import ifft, irfft # type: ignore

    # Preprocessing configuration
    baseline_distortion = True  # Add realistic experimental artifacts

    return baseline_distortion, mp


@app.cell
def _(np):
    """Core spectrum preprocessing functions"""

    def log(msg):
        with open('log.txt', 'a') as f:
            f.writelines(f'{msg}\n')

    def preprocess_peaks(
        intensities,
        positions,
        scales=None,
        substanceDict = None,
        ranged=False,
        baseline_distortion=False,
        downsample=None,
        reverse=False,
    ):
        """
        Extract and preprocess spectral regions with optional Hilbert transform

        Args:
            intensities: Spectral intensity data
            positions: Chemical shift positions (ppm)
            ranges: List of [min, max] ppm ranges to extract
            baseline_distortion: Add realistic baseline drift
            downsample: Target number of points for downsampling
            reverse: Apply Hilbert transform for time-domain analysis

        Returns:
            tuple: (new_positions, new_intensities)
        """
        new_positions = positions  # Default: keep original positions
        new_intensities = intensities  # Default: keep original intensities

        # Select only certain chemical shift ranges
        if ranged:
            ranges = [[-0.1, 0.1]]

            for scale in scales:
                # log(scale)
                for substance in substanceDict:
                    # log(substanceDict[substance][0])
                    if scale == substanceDict[substance][0]:
                        ranges.append(substanceDict[substance][1])

            indicies = set() # Array but with no duplicates
            for x in ranges:
                lower_bound, upper_bound = x
                for i, position in enumerate(new_positions):
                    if lower_bound <= position <= upper_bound:
                        indicies.add(i) # Add instead of append to handle overlapping ranges

            indicies = sorted(indicies) # Sort indicies for consistent ordering

            # Calculate next power of 2
            length = len(indicies)
            if length == 0:
                length = 1

            # Find the next power of 2 using bit operations
            next_power = 1
            while next_power < length:
                next_power <<= 1  # Equivalent to next_power *= 2

            # Calculate how much padding we need
            pad_needed = next_power - length

            if pad_needed > 0:
                left_pad = pad_needed // 2
                right_pad = pad_needed - left_pad

                # Try to pad symmetrically, but respect boundaries
                for _ in range(left_pad):
                    if indicies[0] > 0:
                        indicies.insert(0, indicies[0] - 1)
                    else:
                        # Can't pad left, add to right padding
                        right_pad += 1

                for _ in range(right_pad):
                    if indicies[-1] < len(new_positions) - 1:
                        indicies.append(indicies[-1] + 1)
                    else:
                        # Can't pad right, try padding left again
                        if indicies[0] > 0:
                            indicies.insert(0, indicies[0] - 1)
                        else:
                            # If we can't extend, we'll have to accept non-power-of-2
                            break

            # Final verification - if still not power of 2, force it
            final_length = len(indicies)
            if final_length & (final_length - 1) != 0:
                # Calculate the next power of 2 again
                target_power = 1
                while target_power < final_length:
                    target_power <<= 1

                # If we're closer to the lower power of 2, truncate; otherwise pad
                lower_power = target_power >> 1
                if abs(final_length - lower_power) < abs(final_length - target_power):
                    # Truncate to lower power of 2
                    indicies = indicies[:lower_power]
                else:
                    # Pad to higher power of 2 by duplicating edge values
                    while len(indicies) < target_power:
                        # Alternate between duplicating first and last elements
                        if len(indicies) % 2 == 0:
                            indicies.append(indicies[-1])  # Duplicate last
                        else:
                            indicies.insert(0, indicies[0])  # Duplicate first

            # log(f'Indicies: {len(indicies)}')

            temp_positions = [new_positions[i] for i in indicies]
            temp_intensities = [new_intensities[i] for i in indicies]

            new_positions = temp_positions
            new_intensities = temp_intensities

        # log(f'New_positions: {len(new_positions)}')

        # Convert to FID if needed
        if reverse:
            # Apply Hilbert transform for time-domain representation
            from scipy.signal import hilbert # type: ignore
            from scipy.fft import ifft # type: ignore

            fid = ifft(hilbert(new_intensities))
            fid[0] = 0
            threshold = 1e-16
            fid[np.abs(fid) < threshold] = 0
            fid = fid[fid != 0]
            new_intensities = fid.astype(np.complex64)
            new_positions = [0, 0]


        if downsample is not None and len(new_intensities) > downsample:
            step = len(new_intensities) // downsample

            # Frequency domain filtering to prevent aliasing
            new_len = downsample
            new_nyquist = new_len // 2 + 1
            filtered = np.zeros_like(new_intensities)
            filtered[:new_nyquist] = new_intensities[:new_nyquist]

            # Downsample new_intensities
            new_intensities = new_intensities[::step]

            # Check if new_positions exists and is not [0, 0]
            if 'new_positions' in locals() and not np.array_equal(new_positions, [0, 0]):
                new_positions = new_positions[::step]


        return new_positions, new_intensities

    def preprocess_ratio(scales, substanceDict):
        """
        Calculate concentration ratios relative to internal standard (TSP)

        Args:
            scales: Dictionary of substance concentrations
            substanceDict: Mapping of substance names to spectrum IDs

        Returns:
            dict: Concentration ratios normalized to TSP
        """
        ratios = {
            substance: scales[substance][0] / scales['tsp'][0]
            for substance in scales
            if 'tsp' in scales and scales['tsp'][0] > 0
        }
        return ratios

    return preprocess_peaks, preprocess_ratio


@app.cell
def _(
    baseline_distortion,
    downsample,
    mp,
    preprocess_peaks,
    preprocess_ratio,
    ranged,
    reverse,
    substanceDict,
):
    """Parallel preprocessing pipeline for spectra and references"""

    def preprocess_spectra(
        spectra,
        substanceDict,
        reverse,
        ranged=ranged,
        baseline_distortion=False,
        downsample=None,
    ):
        """
        Complete preprocessing pipeline for a single spectrum

        Returns:
            dict: Preprocessed spectrum with intensities, positions, scales, components, ratios
        """
        new_positions, new_intensities = preprocess_peaks(
            intensities=spectra['intensities'][0],
            positions=spectra['positions'],
            scales=spectra['scales'],
            ranged=ranged,
            substanceDict=substanceDict,
            baseline_distortion=baseline_distortion,
            downsample=downsample,
            reverse=reverse,
        )

        ratios = preprocess_ratio(spectra['scales'], substanceDict)

        return {
            'intensities': new_intensities,
            'positions': new_positions,
            'scales': spectra['scales'],
            'components': spectra['components'],
            'ratios': ratios,
        }

    def process_single_spectrum(spectrum):
        """Worker function for parallel spectrum preprocessing"""
        return preprocess_spectra(
            spectra=spectrum,
            substanceDict=substanceDict,
            baseline_distortion=baseline_distortion,
            ranged=ranged,
            downsample=downsample,
            reverse=reverse,
        )

    def process_single_reference(spectrum_key_and_data):
        """Worker function for parallel reference preprocessing"""
        spectrum_key, reference_data, positions, scales = spectrum_key_and_data
        positions, intensities = preprocess_peaks(
            positions=positions,
            intensities=reference_data,
            scales=spectrum_key,
            ranged=ranged,
            substanceDict=substanceDict,
            downsample=downsample,
            reverse=reverse,
        )
        return (spectrum_key, positions, intensities)

    def process_spectra_parallel(spectra):
        """Parallel preprocessing of all training spectra"""
        num_processes = max(1, mp.cpu_count() - 1)
        print(f'Using {num_processes} processes for spectra preprocessing')

        with mp.Pool(processes=num_processes) as pool:
            preprocessed_spectra = pool.map(process_single_spectrum, spectra)

        return preprocessed_spectra

    def process_references_parallel(reference_spectra, sample_positions):
        """Parallel preprocessing of reference spectra"""
        num_processes = max(1, mp.cpu_count() - 1)
        print(f'Using {num_processes} processes for reference preprocessing')

        # Prepare arguments for parallel processing
        args = [
            (key, intensities, sample_positions, scales)
            for key, (intensities, scales) in reference_spectra.items()
        ]

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_single_reference, args)

        preprocessed_reference_spectra = {
            key: [positions, intensities] for key, positions, intensities in results
        }

        return preprocessed_reference_spectra
    return process_references_parallel, process_spectra_parallel


@app.cell
def _(
    process_references_parallel,
    process_spectra_parallel,
    reference_spectra,
    spectra,
):
    """Execute preprocessing pipelines"""

    # Process all training spectra
    print("Preprocessing training spectra...")
    preprocessed_spectra = process_spectra_parallel(spectra)

    # Process reference spectra
    print("Preprocessing reference spectra...")
    preprocessed_reference_spectra = process_references_parallel(
        reference_spectra, 
        spectra[0]['positions']  # Use sample positions for reference
    )

    # Display preprocessing results
    print(f"Preprocessed data dimensions:")
    print(f"  Positions: {len(preprocessed_spectra[0]['positions'])}")
    print(f"  Intensities: {len(preprocessed_spectra[0]['intensities'])}")
    print(f"  Data type: {type(preprocessed_spectra[0]['intensities'][0])}")
    return preprocessed_reference_spectra, preprocessed_spectra


@app.cell
def _(
    plt,
    preprocessed_reference_spectra,
    reference_spectra,
    reverse,
    spectra,
    substanceDict,
):
    """Generate before/after preprocessing comparison plots"""

    plt.figure(figsize=(15, 6))

    # Original spectra (left panel)
    plt.subplot(1, 2, 1)
    for substance in substanceDict:
        spectrum_id = substanceDict[substance][0]
        plt.plot(
            spectra[0]['positions'],
            reference_spectra[spectrum_id][0],
            alpha=0.7,
            label=substance
        )
    plt.title('Original Reference Spectra')
    plt.xlabel('Chemical Shift (ppm)')
    plt.ylabel('Intensity')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Preprocessed spectra (right panel)
    plt.subplot(1, 2, 2)
    for substance in substanceDict:
        spectrum_id = substanceDict[substance][0]
        if reverse:
            # Time domain: plot magnitude of complex data
            complex_data = preprocessed_reference_spectra[spectrum_id]
            plt.plot(complex_data, alpha=0.7, label=substance)
            plt.title('Preprocessed (Hilbert Transform - Time Domain)')
            plt.xlabel('Time Points')
            plt.ylabel('Magnitude')
        else:
            # Frequency domain: normal plotting
            plt.plot(
                preprocessed_reference_spectra[spectrum_id][0],
                preprocessed_reference_spectra[spectrum_id][1],
                alpha=0.7,
                label=substance
            )
            plt.title('Preprocessed (Frequency Domain)')
            plt.xlabel('Chemical Shift (ppm)')
            plt.ylabel('Intensity')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    preprocessedreferencefigure = plt.gca()

    print(preprocessed_reference_spectra['SP:3368'][0])
    return (preprocessedreferencefigure,)


@app.cell
def _(graph_count, plt, preprocessed_spectra, reverse):
    def _():
        """Compare original vs preprocessed training spectra"""

        plt.figure(figsize=(graph_count * 4, graph_count * 4))

        for graphcounter2 in range(1, graph_count**2 + 1):
            plt.subplot(graph_count, graph_count, graphcounter2)

            if reverse:
                # Time domain: plot magnitude of complex data
                complex_data = preprocessed_spectra[graphcounter2]['intensities']
                plt.plot(complex_data)
                plt.title(f'Sample {graphcounter2} (Time Domain)')
                plt.xlabel('Time Points')
                plt.ylabel('Magnitude')
            else:
                # Frequency domain: normal plotting
                plt.plot(preprocessed_spectra[graphcounter2]['positions'], preprocessed_spectra[graphcounter2]['intensities'])
                plt.title(f'Sample {graphcounter2} (Frequency Domain)')
                plt.xlabel('Data Points')
                plt.ylabel('Intensity')
            plt.tight_layout()
        return plt.gca()


    preprocessedfigure = _()
    return (preprocessedfigure,)


@app.cell(hide_code=True)
def _(
    baseline_distortion,
    mo,
    preprocessed_spectra,
    preprocessedfigure,
    preprocessedreferencefigure,
    ranged,
    reverse,
):
    mo.md(
        f"""
    ## Spectral Preprocessing Pipeline

    **Preprocessing Configuration:**

    - **Spectral ranged:** {"Ranged" if ranged else "Disabled"}
    - **Baseline distortion:** {'Enabled' if baseline_distortion else 'Disabled'}
    - **Hilbert transform:** {'Applied (time domain)' if reverse else 'Not applied (frequency domain)'}
    - **Data type:** {'Complex64 (time domain)' if reverse else 'Float32 (frequency domain)'}

    **Processed Data Dimensions:**

    - **Positions:** {len(preprocessed_spectra[0]['positions'])} points
    - **Intensities:** {len(preprocessed_spectra[0]['intensities'])} points

    ### Reference Spectra Comparison

    {mo.as_html(preprocessedreferencefigure)}

    ### Training Spectra Samples

    {mo.as_html(preprocessedfigure)}

    **Preprocessing Benefits:**

    {'**Time Domain Analysis (Hilbert Transform):**' if reverse else '**Frequency Domain Analysis:**'}
    {'''
    - **Phase information:** Preserves both magnitude and phase components
    - **Time-resolved features:** Enables analysis of FID decay patterns  
    - **Reduced spectral complexity:** Focuses on fundamental signal characteristics
    - **Improved SNR:** Time domain filtering can enhance signal quality''' if reverse else '''
    - **Chemical shift resolution:** Maintains precise ppm scale for peak identification
    - **Spectral features:** Preserves traditional NMR peak patterns
    - **Direct interpretation:** Results directly relate to chemical structure
    - **Standard analysis:** Compatible with conventional NMR processing'''}

    The preprocessed data serves as input for machine learning models to learn metabolite detection and quantification patterns.
    """
    )
    return


@app.cell
def _():
    """Import machine learning dependencies"""
    from torch.utils.data import Dataset, DataLoader # type: ignore
    import h5py # type: ignore

    return DataLoader, Dataset, h5py


@app.cell
def _(Dataset, h5py, torch):
    """Streamable dataset class for memory-efficient data loading"""

    class StreamableNMRDataset(Dataset):
        """
        Memory-efficient dataset that loads NMR data from HDF5 files on demand

        This approach prevents loading entire datasets into memory, enabling
        training on larger datasets that exceed available RAM.
        """

        def __init__(self, file_path, dataset_name):
            self.file_path = file_path
            self.dataset_name = dataset_name

            # Verify file structure and get dataset length
            with h5py.File(self.file_path, 'r') as f:
                self.length = f[f'{dataset_name}_data'].shape[0]

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # Load individual samples on demand to minimize memory usage
            with h5py.File(self.file_path, 'r') as f:
                data = torch.tensor(
                    f[f'{self.dataset_name}_data'][idx], 
                    dtype=torch.complex64
                )
                labels = torch.tensor(
                    f[f'{self.dataset_name}_labels'][idx], 
                    dtype=torch.float32
                )
            return data, labels

    return (StreamableNMRDataset,)


@app.cell
def _(StreamableNMRDataset, h5py, os, processed_data_dir):
    """Data persistence utilities for streamable datasets with smart caching"""

    def save_datasets_to_files(train_data, train_labels, val_data, val_labels, 
                             test_data, test_labels, processed_cache_key):
        """
        Save datasets to HDF5 files in processed data directory

        Args:
            *_data, *_labels: PyTorch tensors for each dataset split
            processed_cache_key: Cache key for processed datasets

        Returns:
            str: Path to saved HDF5 file
        """
        os.makedirs(processed_data_dir, exist_ok=True)
        file_path = f'{processed_data_dir}/{processed_cache_key}_datasets.h5'

        print(f"Saving processed datasets to {file_path}...")
        print(f"Dataset shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        print(f"Dataset dtypes - Train: {train_data.dtype}, Val: {val_data.dtype}, Test: {test_data.dtype}")

        with h5py.File(file_path, 'w') as f:
            # Save each dataset split with compression
            f.create_dataset('train_data', data=train_data.numpy(), 
                           compression='gzip', compression_opts=9)
            f.create_dataset('train_labels', data=train_labels.numpy(), 
                           compression='gzip', compression_opts=9)
            f.create_dataset('val_data', data=val_data.numpy(), 
                           compression='gzip', compression_opts=9)
            f.create_dataset('val_labels', data=val_labels.numpy(), 
                           compression='gzip', compression_opts=9)
            f.create_dataset('test_data', data=test_data.numpy(), 
                           compression='gzip', compression_opts=9)
            f.create_dataset('test_labels', data=test_labels.numpy(), 
                           compression='gzip', compression_opts=9)

            # Store metadata for validation
            # Handle case where tensors might not have expected dimensions due to empty data
            if len(train_data.shape) >= 2 and train_data.shape[0] > 0:
                # Normal case: 2D tensor with samples x features
                f.attrs['data_length'] = train_data.shape[1]
            elif len(train_data.shape) == 1 and train_data.shape[0] > 0:
                # 1D tensor case - use the single dimension as data length
                f.attrs['data_length'] = train_data.shape[0]
            else:
                # Empty tensor case - set data_length to 0
                f.attrs['data_length'] = 0
                print("Warning: Empty training data detected, setting data_length to 0")
            
            f.attrs['train_size'] = train_data.shape[0] if len(train_data.shape) > 0 else 0
            f.attrs['val_size'] = val_data.shape[0] if len(val_data.shape) > 0 else 0
            f.attrs['test_size'] = test_data.shape[0] if len(test_data.shape) > 0 else 0

        file_size_mb = os.path.getsize(file_path) / (1024**2)
        print(f"Processed datasets saved successfully. File size: {file_size_mb:.2f} MB")
        return file_path

    def load_datasets_from_files(processed_cache_key):
        """
        Create streamable datasets from HDF5 files in processed data directory

        Returns:
            tuple: (datasets_dict, data_length) or None if file doesn't exist
        """
        file_path = f'{processed_data_dir}/{processed_cache_key}_datasets.h5'

        if not os.path.exists(file_path):
            return None

        print(f"Loading processed datasets from {file_path}...")

        # Create streamable datasets for each split
        train_dataset = StreamableNMRDataset(file_path, 'train')
        val_dataset = StreamableNMRDataset(file_path, 'val')
        test_dataset = StreamableNMRDataset(file_path, 'test')

        # Load metadata
        with h5py.File(file_path, 'r') as f:
            data_length = f.attrs['data_length']
            train_size = f.attrs['train_size']
            val_size = f.attrs['val_size'] 
            test_size = f.attrs['test_size']

        print(f"Loaded processed datasets - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        print(f"Feature vector length: {data_length}")

        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
        }, data_length

    return load_datasets_from_files, save_datasets_to_files


@app.cell
def _(
    downsample,
    generate_processed_cache_key,
    held_back_metabolites,
    load_datasets_from_files,
    np,
    preprocessed_reference_spectra,
    preprocessed_spectra,
    random,
    raw_cache_key,
    reference_spectra,
    reverse,
    save_datasets_to_files,
    spectra,
    substanceDict,
    torch,
):
    """Main training data preparation with smart caching based on preprocessing"""

    def get_training_data_mlp(
        spectra,
        reference_spectra,
        held_back_metabolites,
        processed_cache_key,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    ):
        """
        Create training datasets with hold-back validation for metabolite detection

        Strategy:
        - Training: Spectra without held-back metabolites (for all other metabolites)
        - Validation: Mixed spectra with one held-back metabolite (hyperparameter tuning)
        - Test: Spectra with the other held-back metabolite (final evaluation)

        Args:
            spectra: Preprocessed training spectra
            reference_spectra: Pure component reference spectra
            held_back_metabolites: [test_metabolite, validation_metabolite]
            processed_cache_key: Cache key that includes preprocessing parameters

        Returns:
            tuple: (datasets_dict, feature_vector_length)
        """
        # Check for existing cached datasets based on processed data
        existing_datasets = load_datasets_from_files(processed_cache_key)

        if existing_datasets is not None:
            return existing_datasets

        print("Creating new processed datasets with hold-back validation...")

        # Get spectrum IDs for held-back metabolites
        held_back_key_test = substanceDict[held_back_metabolites[0]][0]
        held_back_key_validation = substanceDict[held_back_metabolites[1]][0]
        print(f'Hold-back metabolites: {held_back_metabolites}')
        print(f'  Test key: {held_back_key_test}')
        print(f'  Validation key: {held_back_key_validation}')

        # Separate spectra based on held-back metabolite presence
        train_spectra = []
        val_with_holdback = []
        test_with_holdback = []

        # Find maximum length across all spectra for padding
        max_intensities_length = max(len(spectrum['intensities']) for spectrum in spectra)
        max_positions_length = max(len(spectrum['positions']) for spectrum in spectra if spectrum['positions'] != [0, 0])

        # Pad all spectra to the same length
        for spectrum in spectra:
            # Pad intensities
            if len(spectrum['intensities']) < max_intensities_length:
                padding_needed = max_intensities_length - len(spectrum['intensities'])
                if isinstance(spectrum['intensities'], np.ndarray):
                    spectrum['intensities'] = np.concatenate([
                        spectrum['intensities'], 
                        np.full(padding_needed, -np.inf)
                    ])
                else:
                    spectrum['intensities'] = spectrum['intensities'] + [-float('inf')] * padding_needed

            # Pad positions (only if not [0, 0] placeholder)
            if spectrum['positions'] != [0, 0] and len(spectrum['positions']) < max_positions_length:
                padding_needed = max_positions_length - len(spectrum['positions'])
                if isinstance(spectrum['positions'], np.ndarray):
                    spectrum['positions'] = np.concatenate([
                        spectrum['positions'], 
                        np.full(padding_needed, -np.inf)
                    ])
                else:
                    spectrum['positions'] = spectrum['positions'] + [-float('inf')] * padding_needed

        # Find maximum length across all reference spectra for padding
        max_ref_intensities_length = 0
        max_ref_positions_length = 0

        for key, value in reference_spectra.items():
            positions, intensities = value
            max_ref_intensities_length = max(max_ref_intensities_length, len(intensities))
            if positions != [0, 0]:
                max_ref_positions_length = max(max_ref_positions_length, len(positions))

        # Pad all reference spectra to the same length
        for key, value in reference_spectra.items():
            positions, intensities = value

            # Pad intensities
            if len(intensities) < max_ref_intensities_length:
                padding_needed = max_ref_intensities_length - len(intensities)
                if isinstance(intensities, np.ndarray):
                    intensities = np.concatenate([intensities, np.full(padding_needed, -np.inf)])
                else:
                    intensities = intensities + [-float('inf')] * padding_needed

            # Pad positions (only if not [0, 0] placeholder)
            if positions != [0, 0] and len(positions) < max_ref_positions_length:
                padding_needed = max_ref_positions_length - len(positions)
                if isinstance(positions, np.ndarray):
                    positions = np.concatenate([positions, np.full(padding_needed, -np.inf)])
                else:
                    positions = positions + [-float('inf')] * padding_needed

            # Update the reference spectra with padded data
            reference_spectra[key] = [positions, intensities]

        for spectrum in spectra:
            if held_back_key_test in spectrum['ratios']:
                test_with_holdback.append(spectrum)
            elif held_back_key_validation in spectrum['ratios']:
                val_with_holdback.append(spectrum)
            else:
                train_spectra.append(spectrum)

        # Further split train_spectra for additional validation/test data
        train_size = len(train_spectra)
        test_size = int(train_size * test_ratio)
        val_size = int(train_size * val_ratio)

        # Randomize and split indices
        all_indices = list(range(train_size))
        random.seed(42)  # Reproducible splits
        random.shuffle(all_indices)

        test_indices = set(all_indices[:test_size])
        val_indices = set(all_indices[test_size:test_size + val_size])
        train_indices = set(all_indices[test_size + val_size:])

        # Initialize data containers
        data_train, labels_train = [], []
        data_val, labels_val = [], []
        data_test, labels_test = [], []

        # Process training data (exclude held-back metabolites)
        for i, spectrum in enumerate(train_spectra):
            if i in train_indices:
                for substance in reference_spectra:
                    if substance not in [held_back_key_test, held_back_key_validation]:
                        # Concatenate spectrum + reference for metabolite-specific analysis
                        temp_data = np.concatenate([
                            spectrum['intensities'],
                            reference_spectra[substance][0],
                        ])

                        # Create label: [presence, concentration]
                        if substance in spectrum['ratios']:
                            temp_label = [1, spectrum['ratios'][substance]]
                        else:
                            temp_label = [0, 0]

                        data_train.append(temp_data)
                        labels_train.append(temp_label)

        # Create validation data with held-back metabolite
        def create_dataset_for_metabolite(spectra_list, target_key, data_list, labels_list):
            # Positive samples (with target metabolite)
            for spectrum in spectra_list:
                temp_data = np.concatenate([
                    spectrum['intensities'],
                    reference_spectra[target_key][0],
                ])
                temp_label = [1, spectrum['ratios'][target_key]]
                data_list.append(temp_data)
                labels_list.append(temp_label)

        # Additional validation/test data from train_spectra splits
        train_without_holdback = [train_spectra[i] for i in train_indices]
        val_without_holdback = [train_spectra[i] for i in val_indices] 
        test_without_holdback = [train_spectra[i] for i in test_indices]

        # Add negative samples (without held-back metabolites)
        for spectra_subset, data_list, labels_list, target_key in [
            (val_without_holdback, data_val, labels_val, held_back_key_validation),
        ]:
            for spectrum in spectra_subset:
                temp_data = np.concatenate([
                    spectrum['intensities'],
                    reference_spectra[target_key][0],
                ])
                temp_label = [0, 0]  # Not present
                data_list.append(temp_data)
                labels_list.append(temp_label)

        # Add positive samples with held-back metabolites
        create_dataset_for_metabolite(val_with_holdback, held_back_key_validation, data_val, labels_val)
        create_dataset_for_metabolite(test_with_holdback, held_back_key_test, data_test, labels_test)

        # Display dataset statistics
        print(f'Dataset sizes:')
        print(f'  Training: {len(data_train)} samples')
        print(f'  Validation: {len(data_val)} samples ({len(val_with_holdback)} with {held_back_metabolites[1]})')
        print(f'  Test: {len(data_test)} samples ({len(test_with_holdback)} with {held_back_metabolites[0]})')

        # Convert to tensors
        datasets = {
            'train': (torch.tensor(data_train, dtype=torch.complex64), 
                     torch.tensor(labels_train, dtype=torch.float32)),
            'val': (torch.tensor(data_val, dtype=torch.complex64), 
                   torch.tensor(labels_val, dtype=torch.float32)),
            'test': (torch.tensor(data_test, dtype=torch.complex64), 
                    torch.tensor(labels_test, dtype=torch.float32))
        }

        # Save to HDF5 for streamable access
        file_path = save_datasets_to_files(
            *datasets['train'], *datasets['val'], *datasets['test'],
            processed_cache_key
        )

        # Clear memory and create streamable datasets
        del datasets
        torch.cuda.empty_cache()

        return load_datasets_from_files(processed_cache_key)

    # Generate cache key that includes preprocessing parameters
    processed_cache_key = generate_processed_cache_key(raw_cache_key, downsample, reverse)

    # Execute training data preparation with preprocessing-aware caching
    training_data, data_length = get_training_data_mlp(
        spectra=preprocessed_spectra,
        reference_spectra=preprocessed_reference_spectra,
        held_back_metabolites=held_back_metabolites,
        processed_cache_key=processed_cache_key,
    )

    # Delete data from earlier in the pipelines
    del spectra, reference_spectra, preprocessed_spectra, preprocessed_reference_spectra

    return data_length, processed_cache_key, training_data


@app.cell(hide_code=True)
def _(data_length, held_back_metabolites, mo, training_data):
    mo.md(
        f"""
    ## Training Data Preparation Complete

    **Hold-back Validation Strategy:**

    This approach tests the model's ability to detect and quantify metabolites it has never seen during training, simulating real-world scenarios where new metabolites may be encountered.

    - **Training metabolites:** All except {held_back_metabolites[0]} and {held_back_metabolites[1]}
    - **Validation metabolite:** {held_back_metabolites[1]} (for hyperparameter tuning)
    - **Test metabolite:** {held_back_metabolites[0]} (for final performance evaluation)

    **Dataset Characteristics:**

    - **Feature vector length:** {data_length} (concatenated spectrum + reference)
    - **Data type:** Complex64 (supports both frequency and time domain)
    - **Storage format:** HDF5 with compression for memory efficiency

    **Dataset Sizes:**
    {chr(10).join([f"- **{split.title()} dataset:** {len(dataset)}" for split, dataset in training_data.items()])}

    **Multi-task Learning Setup:**

    Each sample contains:

    1. **Input:** [Mixed spectrum | Reference spectrum] → Feature vector of length {data_length}
    2. **Output:** [Presence (0/1), Concentration ratio] → 2D target vector

    This enables the model to simultaneously learn:

    - **Binary classification:** Is the reference metabolite present in the mixture?
    - **Concentration regression:** If present, what is its relative concentration?
    """
    )
    return


@app.cell
def _():
    """Import model architecture dependencies"""
    import copy
    import torch.optim as optim # type: ignore
    import torch.nn as nn # type: ignore
    import math

    return copy, math, nn, optim


@app.cell
def _(torch):
    """Utility function for removing padding from input tensors"""

    def remove_padding(tensor, pad_value=-float('inf')):
        """
        Remove padding values from tensor

        Args:
            tensor: Input tensor that may contain padding
            pad_value: The padding value to remove (default: -inf)

        Returns:
            tensor: Tensor with padding removed
        """
        if tensor.dtype.is_complex:
            # For complex tensors, check both real and imaginary parts
            mask = ~(torch.isinf(tensor.real) | torch.isinf(tensor.imag))
        else:
            # For real tensors, check for inf values
            mask = ~torch.isinf(tensor)

        # Find the last True value in the mask to determine actual length
        if mask.any():
            # Get the last valid index across all dimensions
            if tensor.dim() > 1:
                # For batched data, find max valid length across batch
                max_valid_idx = 0
                for batch_idx in range(tensor.size(0)):
                    batch_mask = mask[batch_idx]
                    if batch_mask.any():
                        last_valid = torch.where(batch_mask)[0][-1].item()
                        max_valid_idx = max(max_valid_idx, last_valid + 1)
                return tensor[:, :max_valid_idx]
            else:
                # For 1D tensors
                last_valid = torch.where(mask)[0][-1].item()
                return tensor[:last_valid + 1]

        # If no valid data found, return empty tensor with correct shape
        if tensor.dim() > 1:
            return tensor[:, :0]
        else:
            return tensor[:0]

    return (remove_padding,)


@app.cell
def _(nn, remove_padding, torch):
    """Multi-Layer Perceptron for metabolite detection and quantification"""

    class MLPRegressor(nn.Module):
        def __init__(self, input_size=2048, trial=None):
            super().__init__()
            self.input_size = input_size
            self.window_size = 256

            stride_ratio = trial.suggest_float('stride_ratio', 0.25, 0.75)
            self.stride = int(self.window_size * stride_ratio)
            self.stride = max(1, self.stride)

            # Local feature extractor for each window
            # *4 because we handle real+imag for both spectrum and reference
            self.local_feature_extractor = nn.Sequential(
                nn.Linear(self.window_size * 4, 512),  # *4 for real+imag of spectrum+reference
                nn.ReLU(),
                nn.Linear(512, 128)
            )

            # Calculate number of windows correctly
            num_windows = (input_size // 2 - self.window_size) // self.stride + 1

            if num_windows <= 0:
                num_windows = 1

            self.global_aggregator = nn.Sequential(
                nn.Linear(num_windows * 128, 256),
                nn.ReLU(),
                nn.Linear(256, 2)  # [presence, concentration]
            )

        def forward(self, x):
            batch_size = x.size(0)

            # **NEW: Remove padding before processing**
            x = remove_padding(x)

            # Update input_size based on actual data length after padding removal
            actual_input_size = x.size(1)

            # **FIX: Handle complex input by separating real and imaginary parts**
            if x.dtype.is_complex:
                x_real = x.real.float()
                x_imag = x.imag.float()
                # Concatenate real and imaginary parts
                x = torch.cat([x_real, x_imag], dim=-1)
            else:
                x = x.float()
                # If input is real, create zero imaginary part
                x = torch.cat([x, torch.zeros_like(x)], dim=-1)

            # Now x has shape [batch, actual_input_size * 2] (real + imag)
            # Split spectrum and reference (each has real + imag components)
            quarter_size = actual_input_size // 2
            spectrum_real = x[:, :quarter_size]
            spectrum_imag = x[:, quarter_size:quarter_size*2]
            reference_real = x[:, quarter_size*2:quarter_size*3] 
            reference_imag = x[:, quarter_size*3:]

            window_features = []

            spectrum_length = spectrum_real.size(1)

            # Adjust window size if it's larger than actual data
            effective_window_size = min(self.window_size, spectrum_length)

            for i in range(0, spectrum_length - effective_window_size + 1, self.stride):
                # Extract windows for all components
                spec_real_window = spectrum_real[:, i:i+effective_window_size]
                spec_imag_window = spectrum_imag[:, i:i+effective_window_size]
                ref_real_window = reference_real[:, i:i+effective_window_size]
                ref_imag_window = reference_imag[:, i:i+effective_window_size]

                # Pad window to expected size if needed
                if effective_window_size < self.window_size:
                    pad_size = self.window_size - effective_window_size
                    spec_real_window = torch.cat([spec_real_window, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)
                    spec_imag_window = torch.cat([spec_imag_window, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)
                    ref_real_window = torch.cat([ref_real_window, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)
                    ref_imag_window = torch.cat([ref_imag_window, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)

                # Concatenate all components
                window_input = torch.cat([
                    spec_real_window, spec_imag_window,
                    ref_real_window, ref_imag_window
                ], dim=-1)

                features = self.local_feature_extractor(window_input)
                window_features.append(features)

            if len(window_features) == 0:
                # Fallback for edge cases
                window_input = torch.cat([
                    spectrum_real[:, :effective_window_size],
                    spectrum_imag[:, :effective_window_size],
                    reference_real[:, :effective_window_size],
                    reference_imag[:, :effective_window_size]
                ], dim=-1)

                # Pad if needed
                if effective_window_size < self.window_size:
                    pad_size = self.window_size - effective_window_size
                    padding = torch.zeros(batch_size, pad_size * 4, device=x.device)
                    window_input = torch.cat([window_input, padding], dim=1)

                features = self.local_feature_extractor(window_input)
                window_features.append(features)

            global_features = torch.cat(window_features, dim=-1)
            return self.global_aggregator(global_features)
    return (MLPRegressor,)


@app.cell
def _(math, nn, remove_padding, torch):
    """Advanced Sliding Window Transformer architecture for NMR spectral analysis"""

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for transformer sequences"""

        def __init__(self, d_model, dropout=0.0, max_len=5000):
            super(PositionalEncoding, self).__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)

        def forward(self, x):
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].transpose(0, 1)
            return x

    class TransformerRegressor(nn.Module):
        """
        Sliding Window Transformer specifically designed for complex NMR FID data
        """

        def __init__(self, input_size, trial=None, **kwargs):
            super(TransformerRegressor, self).__init__()

            self.input_size = input_size
            self.window_size = kwargs.get('window_size', 256)

            # Hyperparameter configuration
            if trial is not None:
                self.d_model = int(trial.suggest_categorical('d_model', [128, 256, 512]))
                self.nhead = int(trial.suggest_categorical('nhead', [8, 16]))
                self.num_layers = int(trial.suggest_int('num_layers', 3, 6))
                self.dim_feedforward = int(trial.suggest_categorical('dim_feedforward', [512, 1024, 2048]))

                # Sliding window parameters
                stride_ratio = trial.suggest_float('stride_ratio', 0.25, 0.75)
                self.stride = int(self.window_size * stride_ratio)
                self.stride = max(1, self.stride)
            else:
                self.d_model = kwargs.get('d_model', 256)
                self.nhead = kwargs.get('nhead', 8)
                self.num_layers = kwargs.get('num_layers', 4)
                self.dim_feedforward = kwargs.get('dim_feedforward', 1024)
                self.stride = kwargs.get('stride', 128)


            # Ensure attention head compatibility
            while self.d_model % self.nhead != 0:
                self.nhead = max(1, self.nhead - 1)

            # Local window transformer for processing individual windows
            # *4 because we handle real+imag for both spectrum and reference
            self.window_projection = nn.Linear(self.window_size * 4, self.d_model)

            # Local transformer for each window
            local_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward // 2,  # Smaller for local processing
                dropout=0.0,
                activation='gelu',
                batch_first=True,
            )
            self.local_transformer = nn.TransformerEncoder(
                local_encoder_layer, num_layers=max(1, self.num_layers // 2)
            )

            # Calculate number of windows
            num_windows = (input_size // 2 - self.window_size) // self.stride + 1
            if num_windows <= 0:
                num_windows = 1

            # Global transformer for inter-window relationships
            global_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
            )
            self.global_transformer = nn.TransformerEncoder(
                global_encoder_layer, num_layers=max(1, self.num_layers - self.num_layers // 2)
            )

            # Positional encoding for global context
            self.pos_encoding = PositionalEncoding(self.d_model, 0.0, num_windows)

            # Task-specific heads
            self.presence_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.LayerNorm(self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, 1)
            )

            self.concentration_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.LayerNorm(self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, 1)
            )

        def forward(self, x):
            batch_size = x.size(0)

            # **NEW: Remove padding before processing**
            x = remove_padding(x)

            # Update input_size based on actual data length after padding removal
            actual_input_size = x.size(1)

            # Handle complex input by separating real and imaginary parts
            if x.dtype.is_complex:
                x_real = x.real.float()
                x_imag = x.imag.float()
                # Concatenate real and imaginary parts
                x = torch.cat([x_real, x_imag], dim=-1)
            else:
                x = x.float()
                # If input is real, create zero imaginary part
                x = torch.cat([x, torch.zeros_like(x)], dim=-1)

            # Now x has shape [batch, actual_input_size * 2] (real + imag)
            # Split spectrum and reference (each has real + imag components)
            quarter_size = actual_input_size // 2
            spectrum_real = x[:, :quarter_size]
            spectrum_imag = x[:, quarter_size:quarter_size*2]
            reference_real = x[:, quarter_size*2:quarter_size*3] 
            reference_imag = x[:, quarter_size*3:]

            window_features = []
            spectrum_length = spectrum_real.size(1)

            # Adjust window size if it's larger than actual data
            effective_window_size = min(self.window_size, spectrum_length)

            # Process each sliding window
            for i in range(0, spectrum_length - effective_window_size + 1, self.stride):
                # Extract windows for all components
                spec_real_window = spectrum_real[:, i:i+effective_window_size]
                spec_imag_window = spectrum_imag[:, i:i+effective_window_size]
                ref_real_window = reference_real[:, i:i+effective_window_size]
                ref_imag_window = reference_imag[:, i:i+effective_window_size]

                # Pad window to expected size if needed
                if effective_window_size < self.window_size:
                    pad_size = self.window_size - effective_window_size
                    spec_real_window = torch.cat([spec_real_window, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)
                    spec_imag_window = torch.cat([spec_imag_window, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)
                    ref_real_window = torch.cat([ref_real_window, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)
                    ref_imag_window = torch.cat([ref_imag_window, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)

                # Concatenate all components for this window
                window_input = torch.cat([
                    spec_real_window, spec_imag_window,
                    ref_real_window, ref_imag_window
                ], dim=-1)

                # Project window to transformer dimension
                window_embed = self.window_projection(window_input).unsqueeze(1)  # [batch, 1, d_model]

                # Local transformer processing
                local_features = self.local_transformer(window_embed)  # [batch, 1, d_model]
                window_features.append(local_features.squeeze(1))  # [batch, d_model]

            # Handle edge case where no windows were created
            if len(window_features) == 0:
                # Fallback: use first window_size points
                window_input = torch.cat([
                    spectrum_real[:, :effective_window_size],
                    spectrum_imag[:, :effective_window_size],
                    reference_real[:, :effective_window_size],
                    reference_imag[:, :effective_window_size]
                ], dim=-1)

                # Pad if needed
                if effective_window_size < self.window_size:
                    pad_size = self.window_size - effective_window_size
                    padding = torch.zeros(batch_size, pad_size * 4, device=x.device)
                    window_input = torch.cat([window_input, padding], dim=1)

                window_embed = self.window_projection(window_input).unsqueeze(1)
                local_features = self.local_transformer(window_embed)
                window_features.append(local_features.squeeze(1))

            # Stack all window features into sequence
            window_sequence = torch.stack(window_features, dim=1)  # [batch, num_windows, d_model]

            # Add positional encoding for global context
            window_sequence = self.pos_encoding(window_sequence)

            # Global transformer for inter-window relationships
            global_features = self.global_transformer(window_sequence)  # [batch, num_windows, d_model]

            # Global average pooling across windows
            pooled_features = torch.mean(global_features, dim=1)  # [batch, d_model]

            # Task-specific predictions
            presence_logits = self.presence_head(pooled_features)
            concentration_pred = self.concentration_head(pooled_features)

            return torch.cat([presence_logits, concentration_pred], dim=1)

    return (TransformerRegressor,)


@app.cell
def _(MLPRegressor, TransformerRegressor, nn, remove_padding, torch):
    """Hybrid ensemble combining MLP and Transformer architectures"""

    class HybridEnsembleRegressor(nn.Module):
        """
        Ensemble model combining MLP and Transformer strengths

        - MLP: Efficient processing of global spectral features
        - Transformer: Advanced sequence modeling and attention mechanisms
        - Learnable weights: Adaptive combination based on task requirements
        """

        def __init__(self, input_size, trial=None, **kwargs):
            super(HybridEnsembleRegressor, self).__init__()

            # Initialize component models
            self.mlp = MLPRegressor(input_size, trial, **kwargs)
            self.transformer = TransformerRegressor(input_size, trial, **kwargs)

            # Task-specific ensemble weights
            if trial is not None:
                self.classification_weight = trial.suggest_float('class_ensemble_weight', 0.1, 0.9)
                self.concentration_weight = trial.suggest_float('conc_ensemble_weight', 0.1, 0.9)
            else:
                self.classification_weight = kwargs.get('class_ensemble_weight', 0.3)  # Favor transformer
                self.concentration_weight = kwargs.get('conc_ensemble_weight', 0.7)   # Favor MLP

        def forward(self, x):
            # **NEW: Remove padding before processing**
            x = remove_padding(x)

            # Get predictions from both models (they will handle their own padding removal)
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

    return (HybridEnsembleRegressor,)


@app.cell
def _(nn, torch):
    """Advanced loss function with curriculum learning and adaptive weighting"""

    def compute_loss(predictions, targets, epoch=0, max_epochs=200):
        """
        Multi-task loss with balanced weighting
        """
        # Ensure all tensors are float32 for loss computation
        if predictions.dtype.is_complex:
            predictions = predictions.real
        if targets.dtype.is_complex:
            targets = targets.real

        predictions = predictions.float()
        targets = targets.float()

        presence_logits = predictions[:, 0]
        concentration_pred = predictions[:, 1] 
        presence_true = targets[:, 0]
        concentration_true = targets[:, 1]

        # Binary classification loss
        classification_loss = nn.BCEWithLogitsLoss()(presence_logits, presence_true)

        # More aggressive curriculum learning - start with 0.5 weight instead of 0
        curriculum_weight = min(1.0, 0.5 + epoch / (max_epochs * 0.5))

        present_mask = presence_true == 1

        if present_mask.sum() > 0:
            # Simpler concentration loss without confidence weighting
            concentration_loss = nn.SmoothL1Loss()(
                concentration_pred[present_mask], 
                concentration_true[present_mask]
            )

            # Calculate monitoring metrics
            concentration_diff = concentration_pred[present_mask] - concentration_true[present_mask]
            concentration_mae = torch.mean(torch.abs(concentration_diff))
            concentration_rmse = torch.sqrt(torch.mean(concentration_diff ** 2))
        else:
            concentration_loss = torch.tensor(0.0, device=predictions.device)
            concentration_mae = torch.tensor(0.0, device=predictions.device)
            concentration_rmse = torch.tensor(0.0, device=predictions.device)

        # Balanced weighting: equal importance to both tasks
        total_loss = 0.5 * classification_loss + 0.5 * curriculum_weight * concentration_loss

        return total_loss, classification_loss, concentration_mae, concentration_rmse

    return (compute_loss,)


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
def _(
    DataLoader,
    HybridEnsembleRegressor,
    MLPRegressor,
    TransformerRegressor,
    compute_loss,
    copy,
    np,
    optim,
    torch,
    tqdm,
):
    def train_model(training_data, trial, model_type='mlp'):
        """
        Train a multi-task neural network using streamable DataLoaders for memory efficiency.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        max_epochs = 200
        batch_size = int(trial.suggest_float('batch_size', 10, 100, step=10))
        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        # Get input size from first sample of streamable dataset
        sample_data, _ = training_data['train_dataset'][0]
        input_length = len(sample_data)

        if model_type == 'transformer':
            model = TransformerRegressor(
                input_size=input_length, trial=trial
            ).to(device)
        elif model_type == 'mlp':
            model = MLPRegressor(input_size=input_length, trial=trial).to(device)
        elif model_type == 'ensemble':
            model = HybridEnsembleRegressor(input_size=input_length, trial=trial).to(device)

        # Create DataLoaders with streamable datasets
        # Increase num_workers for better I/O performance with file-based datasets
        # num_workers = min(4, max(1, torch.get_num_threads() // 2))
        num_workers = 0

        train_loader = DataLoader(
            training_data['train_dataset'], 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            training_data['val_dataset'], 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
        test_loader = DataLoader(
            training_data['test_dataset'], 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )

        # Test model with a small batch to catch memory issues early
        try:
            sample_batch = next(iter(train_loader))
            dummy_data, dummy_labels = sample_batch
            dummy_data = dummy_data.to(device, non_blocking=True)
            dummy_labels = dummy_labels.to(device, non_blocking=True)

            # Ensure labels are float32 (critical for loss functions)
            if dummy_labels.dtype.is_complex:
                dummy_labels = dummy_labels.real
            dummy_labels = dummy_labels.float()

            dummy_output = model(dummy_data)

            # Use your proper loss function instead of raw MSELoss
            dummy_loss, _, _, _ = compute_loss(dummy_output, dummy_labels, 0, 200)
            dummy_loss.backward()
            model.zero_grad()

            del dummy_data, dummy_labels, dummy_output, dummy_loss
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            del model
            torch.cuda.empty_cache()
            raise e

        # try:
        #     model = torch.compile(model)
        # except Exception:
        #     print('Compilation failed — using uncompiled model.')

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Early stopping parameters
        early_stop_patience = 30
        min_delta = 1e-4
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
                    total_loss, class_loss, conc_mae, conc_rmse = compute_loss(
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
                        val_loss, _, _, _ = compute_loss(
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


@app.cell
def _(
    MODEL_TYPE,
    model_cache_dir,
    os,
    processed_cache_key,
    torch,
    tqdm,
    train_model,
    training_data,
    trials,
):
    import optuna # type: ignore
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

    # Create study name that includes model type and processed data key
    study_name = f'study_{MODEL_TYPE}_{processed_cache_key}'

    os.makedirs(model_cache_dir, exist_ok=True)

    # Create or load existing Optuna study with model-specific storage
    study = optuna.create_study(
        direction='minimize',  # Minimize combined error
        study_name=study_name,
        storage=f'sqlite:///{model_cache_dir}/optuna_{MODEL_TYPE}.db',  # Model-specific database
        load_if_exists=True,  # Resume previous optimization if study exists
    )

    # Count completed trials for progress tracking
    completed_trials = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.state is not optuna.trial.TrialState.FAIL
        ]
    )

    # Run hyperparameter optimization if more trials are needed
    if trials - completed_trials > 0:
        with tqdm.tqdm(
            total=trials - completed_trials, desc='Optimizing'
        ) as pbar:

            def callback(study, trial):
                """Progress callback for Optuna optimization"""
                if trial.state is not optuna.trial.TrialState.FAIL:
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
def _(MODEL_TYPE, held_back_metabolites, mo, optuna, study):
    # Use the actual MODEL_TYPE instead of guessing from parameters
    model_type_display = MODEL_TYPE.upper()

    # Create model-specific parameter display based on actual MODEL_TYPE
    if MODEL_TYPE == 'transformer':
        model_params_md = f"""
    **Transformer Architecture:**
    - **Model Dimension (d_model):** {study.best_trial.params.get('d_model', 'N/A')}
    - **Number of Attention Heads:** {study.best_trial.params.get('nhead', 'N/A')}
    - **Number of Encoder Layers:** {study.best_trial.params.get('num_layers', 'N/A')}
    - **Feedforward Dimension:** {study.best_trial.params.get('dim_feedforward', 'N/A')}
    - **Stride Ratio:** {study.best_trial.params.get('stride_ratio', 'N/A'):.3f}

    **Model Architecture:**

    Input Projection → Positional Encoding → Transformer Encoder → Global Average Pooling → Output Projection
    """
    elif MODEL_TYPE == 'mlp':
        # Handle sliding window MLP parameters
        model_params_md = f"""
    **Sliding Window MLP Architecture:**
    - **Stride Ratio:** {study.best_trial.params.get('stride_ratio', 'N/A'):.3f}
    - **Window Size:** 256 (fixed)
    - **Actual Stride:** {int(256 * study.best_trial.params.get('stride_ratio', 0.5))}

    **Model Architecture:**

    Input → Sliding Windows → Local Feature Extraction (per window) → Global Aggregation → Output

    **Window Processing:**
    - Each window processes 256 points
    - Windows overlap with stride of {int(256 * study.best_trial.params.get('stride_ratio', 0.5))} points
    - Local features (128D) extracted from each window
    - Global aggregation combines all window features
    """
    elif MODEL_TYPE == 'ensemble':
        model_params_md = f"""
    **Hybrid Ensemble Architecture:**
    - **Classification Weight:** {study.best_trial.params.get('class_ensemble_weight', 'N/A'):.3f}
    - **Concentration Weight:** {study.best_trial.params.get('conc_ensemble_weight', 'N/A'):.3f}

    **Component Models:**
    - **MLP:** Stride Ratio: {study.best_trial.params.get('stride_ratio', 'N/A'):.3f}
    - **Transformer:** d_model: {study.best_trial.params.get('d_model', 'N/A')}, Layers: {study.best_trial.params.get('num_layers', 'N/A')}

    **Model Architecture:**

    Input → [MLP Branch + Transformer Branch] → Weighted Ensemble → Output
    """
    else:
        model_params_md = f"""
    **{model_type_display} Architecture:**
    - Unknown model configuration

    **Available Parameters:**
    {chr(10).join([f"- **{key}:** {value}" for key, value in study.best_trial.params.items()])}
    """

    mo.md(
        f"""
    ## Hyperparameter Optimization Results

    **Model Type:** {model_type_display}

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
