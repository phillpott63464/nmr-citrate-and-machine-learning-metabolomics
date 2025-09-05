import marimo

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
    count = 100                   # Number of samples per metabolite combination
    trials = 100                  # Number of hyperparameter optimization trialss
    combo_number = 30             # Number of random metabolite combinations to generate
    notebook_name = 'randomisation_hold_back'  # Cache directory identifier

    # Model configuration
    MODEL_TYPE = 'mlp'            # Model architecture: 'mlp', 'transformer', or 'ensemble'
    downsample = 2**9            # Target resolution for ML model (None = no downsampling)
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
        'Citric acid': ['SP:3368', [[2.4, 2.8]]],
        # 'Succinic acid': ['SP:3211',],
        # 'Maleic acid': ['SP:3110', [[5.8, 6.3]]],
        # 'Lactic acid': ['SP:3675', [[1.2, 1.5], [3.9, 4.2]]],
        # 'L-Methionine': ['SP:3509', [[2.0, 2.4], [2.8, 3.0]]],
        # 'L-Proline': ['SP:3406', [[1.8, 4.3]]],
        # 'L-Phenylalanine': ['SP:3507', [[3, 8]]],
        'L-Serine': ['SP:3732',],
        'L-Threonine': ['SP:3437',],
        'L-Tryptophan': ['SP:3455',],
        'L-Tyrosine': ['SP:3464',],
        'L-Valine': ['SP:3490',],
        'Glycine': ['SP:3682',],
    }

    import pandas as pd
    multiplets = pd.read_csv('morgan/Casmdb_Data/multiplets.csv')

    def _():
        for key, item in substanceDict.items():
            centers = multiplets[multiplets['spectrum_id'] == item[0]]
            centers = centers['center']
            centers = centers.to_numpy()
            centers = set(centers)

            arrays = [[x - 0.1, x + 0.1] for x in centers]

            if len(item) < 2:
                item.append(arrays)
            else:
                item[1] += arrays

    _()

    print(substanceDict)

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
def _(hashlib):
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

    return generate_processed_cache_key, generate_raw_cache_key


@app.cell
def _():
    """Import data generation dependencies"""
    from morgan.createTrainingData import createTrainingData
    import morgan
    import numpy as np # type: ignore
    from tqdm import tqdm # type: ignore
    import itertools
    import random
    import pickle
    from pathlib import Path
    import os
    import h5py # type: ignore
    import hashlib

    return (
        createTrainingData,
        h5py,
        hashlib,
        itertools,
        np,
        os,
        pickle,
        random,
        tqdm,
    )


@app.cell
def _(createTrainingData, h5py, itertools, np, os, random, raw_data_dir, tqdm):
    """Streaming data generation with HDF5 Generates using Morgan's code overlaying a modified version of NMRsim (optimised with jax)"""

    def create_streaming_dataset(substanceDict, combo_number, count, raw_cache_key):
        """
        Stream generated spectra directly to HDF5 without loading into memory

        Returns:
            str: Path to the HDF5 file containing streamed data
        """
        os.makedirs(raw_data_dir, exist_ok=True)
        filepath = f'{raw_data_dir}/{raw_cache_key}.h5'

        # Check if file already exists
        if os.path.exists(filepath):
            print(f"Found existing dataset at {filepath}")
            return filepath

        print(f"Creating new streaming dataset at {filepath}")

        # Generate combinations
        substances = list(substanceDict.keys())
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
        else:
            combinations = all_combinations

        # Select held-back metabolites
        held_back_metabolites = random.sample(list(substanceDict.keys()), 2)

        # Calculate total number of spectra we'll generate
        total_spectra = len(combinations) * count

        # Create HDF5 file with streaming capability
        with h5py.File(filepath, 'w') as f:
            # Store metadata
            f.attrs['combo_number'] = combo_number if combo_number is not None else len(combinations)
            f.attrs['count'] = count
            f.attrs['held_back_metabolites'] = [m.encode('utf-8') for m in held_back_metabolites]
            f.attrs['total_combinations'] = len(combinations)
            f.attrs['total_spectra'] = total_spectra

            # Store combinations as string data
            combo_strings = [str(combo) for combo in combinations]
            f.create_dataset('combinations', data=[s.encode('utf-8') for s in combo_strings])

            # Get dimensions from a sample batch
            print("Getting sample dimensions...")
            sample_combination = combinations[0]
            sample_spectrum_ids = [sample_combination[substance][0] for substance in sample_combination]
            sample_batch = createTrainingData(
                substanceSpectrumIds=sample_spectrum_ids,
                sampleNumber=1,
                rondomlyScaleSubstances=True,
                referenceSubstanceSpectrumId='tsp'
            )

            # Extract dimensions
            intensity_shape = sample_batch['intensities'].shape[1]  # Nrobustnessumber of points
            position_shape = sample_batch['positions'].shape[0]

            print(f"Data dimensions - Intensities: {intensity_shape}, Positions: {position_shape}")

            # Create datasets with known dimensions
            # Use chunking for efficient streaming writes
            chunk_size = min(100, total_spectra)  # Reasonable chunk size

            intensities_ds = f.create_dataset(
                'intensities', 
                shape=(total_spectra, intensity_shape),
                dtype=np.float32,
                chunks=(chunk_size, intensity_shape),
                compression='gzip',
                compression_opts=1  # Light compression for speed
            )

            # Positions are the same for all spectra, so store once
            positions_ds = f.create_dataset(
                'positions', 
                shape=(position_shape,),
                dtype=np.float32
            )
            positions_ds[:] = sample_batch['positions']

            # Store scales as variable-length strings (since they're dictionaries)
            scales_ds = f.create_dataset(
                'scales',
                shape=(total_spectra,),
                dtype=h5py.string_dtype(),
                chunks=(chunk_size,)
            )

            # Store which combination each spectrum belongs to
            combo_indices_ds = f.create_dataset(
                'combination_indices',
                shape=(total_spectra,),
                dtype=np.int32,
                chunks=(chunk_size,)
            )

            # Stream data generation
            spectrum_idx = 0

            print(f"Streaming {len(combinations)} combinations with {count} samples each...")

            for combo_idx, combination in enumerate(tqdm(combinations, desc="Generating combinations")):
                # Extract spectrum IDs for this combination
                spectrum_ids = [combination[substance][0] for substance in combination]

                # Generate batch for this combination
                batch_data = createTrainingData(
                    substanceSpectrumIds=spectrum_ids,
                    sampleNumber=count,
                    rondomlyScaleSubstances=True,
                    referenceSubstanceSpectrumId='tsp'
                )

                # Stream individual spectra from this batch
                for sample_idx in range(count):
                    # Store intensity data
                    intensities_ds[spectrum_idx] = batch_data['intensities'][sample_idx]

                    # Store scales as JSON string
                    sample_scales = {
                        key: [values[sample_idx]]
                        for key, values in batch_data['scales'].items()
                    }
                    scales_ds[spectrum_idx] = str(sample_scales)

                    # Store combination index
                    combo_indices_ds[spectrum_idx] = combo_idx

                    spectrum_idx += 1

                # Optional: flush to disk periodically for very large datasets
                if combo_idx % 10 == 0:
                    f.flush()

        print(f"Successfully streamed {total_spectra} spectra to {filepath}")
        return filepath

    return (create_streaming_dataset,)


@app.cell
def _(h5py, np, os):
    """Load data from streamed HDF5 files"""

    def load_streaming_dataset(filepath):
        """
        Load metadata and create iterators for streamed HDF5 data

        Returns:
            dict: Dataset information and lazy loading functions
        """
        if not os.path.exists(filepath):
            return None

        with h5py.File(filepath, 'r') as f:
            # Load metadata
            metadata = {
                'combo_number': f.attrs['combo_number'],
                'count': f.attrs['count'],
                'held_back_metabolites': [
                    m.decode('utf-8') if isinstance(m, bytes) else m
                    for m in f.attrs['held_back_metabolites']
                ],
                'total_combinations': f.attrs['total_combinations'],
                'total_spectra': f.attrs['total_spectra'],
                'intensity_shape': f['intensities'].shape[1],
                'position_shape': f['positions'].shape[0],
                'filepath': filepath
            }

            # Load combinations list
            combinations_raw = f['combinations'][:]
            metadata['combinations'] = [
                eval(s.decode('utf-8') if isinstance(s, bytes) else s)
                for s in combinations_raw
            ]
        print(f"Loaded dataset metadata - {metadata['total_spectra']} spectra from {metadata['total_combinations']} combinations")
        print(f"Held-back metabolites: {metadata['held_back_metabolites']}")

        return metadata

    def get_spectrum_batch(filepath, start_idx, batch_size):
        """Get a batch of spectra from the HDF5 file"""
        with h5py.File(filepath, 'r') as f:
            end_idx = min(start_idx + batch_size, f['intensities'].shape[0])

            # Load batch data
            intensities = f['intensities'][start_idx:end_idx]
            scales_raw = f['scales'][start_idx:end_idx]
            combo_indices = f['combination_indices'][start_idx:end_idx]

            # Parse scales from strings
            scales = []
            for scale_str in scales_raw:
                scales.append(eval(scale_str.decode('utf-8') if isinstance(scale_str, bytes) else scale_str))

            # Get components for the combinations used in this batch
            unique_combo_indices = np.unique(combo_indices)

            return {
                'intensities': intensities,
                'scales': scales,
                'combo_indices': combo_indices,
                'start_idx': start_idx,
                'end_idx': end_idx
            }

    return get_spectrum_batch, load_streaming_dataset


@app.cell
def _(
    combo_number,
    count,
    create_streaming_dataset,
    downsample,
    generate_processed_cache_key,
    generate_raw_cache_key,
    load_streaming_dataset,
    reverse,
    substanceDict,
):
    """Updated main data generation pipeline using streaming"""

    # Generate cache key for raw data
    raw_cache_key = generate_raw_cache_key(substanceDict, combo_number, count)
    # Generate cache key that includes preprocessing parameters
    processed_cache_key = generate_processed_cache_key(raw_cache_key, downsample, reverse)

    dataset_filepath = create_streaming_dataset(substanceDict, combo_number, count, raw_cache_key) # Creates the dataset
    dataset_metadata = load_streaming_dataset(dataset_filepath) # Loads the dataset metadata

    # Extract information for compatibility with existing code
    held_back_metabolites = dataset_metadata['held_back_metabolites']
    combinations = dataset_metadata['combinations']

    print(f'Dataset ready with {dataset_metadata["total_spectra"]} spectra')
    print(f'Held-back metabolites: {held_back_metabolites}')
    print(f'Dataset file: {dataset_filepath}')

    return (
        combinations,
        dataset_filepath,
        held_back_metabolites,
        processed_cache_key,
    )


@app.cell
def _(
    baseline_distortion,
    dataset_filepath,
    downsample,
    get_spectrum_batch,
    h5py,
    os,
    partial,
    pickle,
    preprocess_spectra,
    processed_cache_key,
    processed_data_dir,
    ranged,
    reverse,
    substanceDict,
):
    """Create a streaming dataset class for PyTorch compatibility"""

    class StreamingNMRDataset:
        """
        Streaming dataset that loads NMR data from HDF5 on-demand
        """
        def __init__(self, filepath, preprocessed_enabled=False):
            self.filepath = filepath
            self.preprocessed_enabled = preprocessed_enabled
            self.cache_dir = f'{processed_data_dir}/{processed_cache_key}'
            os.makedirs(self.cache_dir, exist_ok=True)
            with h5py.File(filepath, 'r') as f:
                self.length = f['intensities'].shape[0]
                self.positions = f['positions'][:]

        def _cache_path(self, idx, preprocess_scale):
            # You may want to include preprocessing parameters in the key for robustness
            return os.path.join(self.cache_dir, f"{preprocess_scale}-spectrum_{idx}.pkl")

        def __len__(self):
            return self.length

        def __getitem__(self, idx, preprocess_scale=None):
            if isinstance(idx, slice):
                # Handle slicing
                indices = range(*idx.indices(self.length))
            else:
                indices = [idx]

            out_batch = []
            for i in indices:
                cache_path = self._cache_path(i, preprocess_scale)
                if self.preprocessed_enabled:
                    preprocess_func = preprocess_func = partial(
                        preprocess_spectra,
                        substanceDict=substanceDict,
                        baseline_distortion=baseline_distortion,
                        ranged=ranged,
                        downsample=downsample,
                        reverse=reverse,
                        scale=preprocess_scale
                    )

                    if os.path.exists(cache_path):
                        with open(cache_path, "rb") as f:
                            spectrum = pickle.load(f)
                    else:
                        batch = self.get_batch(i, 1)
                        spectrum = {
                            'intensities': batch['intensities'][0],
                            'positions': self.positions,
                            'scales': batch['scales'][0],
                        }
                        spectrum = preprocess_func(spectrum)
                        with open(cache_path, "wb") as f:
                            pickle.dump(spectrum, f)
                else:
                    batch = self.get_batch(i, 1)
                    spectrum = {
                        'intensities': batch['intensities'][0],
                        'positions': self.positions,
                        'scales': batch['scales'][0],
                    }
                out_batch.append(spectrum)

            if len(out_batch) == 1:
                return out_batch[0]

            return out_batch

        def get_batch(self, start_idx, batch_size):
            """Get multiple spectra efficiently"""
            return get_spectrum_batch(self.filepath, start_idx, batch_size)

    # Create streaming dataset
    spectra = StreamingNMRDataset(dataset_filepath)

    print(f"Created streaming dataset with {len(spectra)} spectra")

    print(spectra[:5])

    return StreamingNMRDataset, spectra


@app.cell(hide_code=True)
def _(combinations, count, held_back_metabolites, mo, spectra):
    mo.md(
        rf"""
    ## Data Generation Results

    **Successfully generated {count} samples for each of {len(combinations)} metabolite combinations**

    **Hold-back Validation Setup:**

    - **Test metabolite:** {held_back_metabolites[0]} (completely excluded from training)
    - **Validation metabolite:** {held_back_metabolites[1]} (used for hyperparameter tuning)

    **Data Structure:**

    - **Total spectra:** {len(spectra)}
    - **Intensities shape:** {spectra[0]['intensities'].shape} (NMR signal data)
    - **Positions shape:** {spectra[0]['positions'].shape} (chemical shift scale in ppm)

    **Sample Concentration Data (first 5 spectra):**

    {chr(10).join([f"Sample {i+1}: {spectrum['scales']}" for i, spectrum in enumerate(spectra[:5])])}

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
            # spectra[graphcounter]['positions'],
            spectra[graphcounter]['intensities'],
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

        data = spectra[0]['positions'][20000:250000]
        data2 = spectra[0]['intensities'][20000:250000]

        # Plot reference spectrum for each metabolite
        for substance in substanceDict:
            spectrum_id = substanceDict[substance][0]
            data2 = reference_spectra[spectrum_id][0][20000:250000]
            plt.plot(
                data,
                data2,
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
    from functools import partial

    # Preprocessing configuration
    baseline_distortion = True  # Add realistic experimental artifacts

    return baseline_distortion, partial


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
            scales: Dictionary of substance concentrations (for training data) or substance ID (for reference data)
            substanceDict: Mapping of substance names to [spectrum_id, range]
            ranged: Select only certain chemical shift ranges
            baseline_distortion: Add realistic baseline drift
            downsample: Target number of points for downsampling
            reverse: Apply Hilbert transform for time-domain analysis

        Returns:
            tuple: (positions, intensities)
        """

        # Select only certain chemical shift ranges
        if ranged:
            ranges = [[-0.1, 0.1]]
            # ranges = []
            # Handle different types of scales parameter
            if isinstance(scales, dict):
                # Training data: scales is a dictionary of substance concentrations
                for scale in scales:
                    for substance in substanceDict:
                        if scale == substanceDict[substance][0]:
                            for x in substanceDict[substance][1]:
                                ranges.append(x)    
            elif isinstance(scales, str):
                # Reference data: scales is a single substance ID
                for substance in substanceDict:
                    if scales == substanceDict[substance][0]:
                        for x in substanceDict[substance][1]:
                            ranges.append(x)
                        break
            # If scales is None or other type, just use the default range

            indicies = set() # Array but with no duplicates
            for x in ranges:
                lower_bound, upper_bound = x
                for i, position in enumerate(positions):
                    if lower_bound <= position <= upper_bound:
                        indicies.add(i) # Add instead of append to handle overlapping ranges

            indicies = sorted(indicies) # Sort indicies for consistent ordering

            length = len(indicies)
            if length == 0:
                length = 1

            if downsample is not None:
                pad_needed = downsample - length
            else:
                pad_needed = 0

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
                    if indicies[-1] < len(positions) - 1:
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

            positions = [positions[i] for i in indicies]
            intensities = [intensities[i] for i in indicies]


        # Convert to FID if needed
        if reverse:
            # Apply Hilbert transform for time-domain representation
            from scipy.signal import hilbert # type: ignore
            from scipy.fft import ifft # type: ignore

            fid = ifft(hilbert(intensities))
            fid[0] = 0
            threshold = 1e-16
            fid[np.abs(fid) < threshold] = 0
            fid = fid[fid != 0]
            intensities = fid.astype(np.complex64)
            positions = [0, 0]


        if downsample is not None and len(intensities) > downsample:
            step = len(intensities) // downsample

            # Frequency domain filtering to prevent aliasing
            new_len = downsample
            new_nyquist = new_len // 2 + 1
            filtered = np.zeros_like(intensities)
            filtered[:new_nyquist] = intensities[:new_nyquist]

            # Downsample intensities
            intensities = intensities[::step]

            # Check if positions exists and is not [0, 0]
            if 'positions' in locals() and not np.array_equal(positions, [0, 0]):
                positions = positions[::step]

        if reverse:
            positions = np.asarray(positions, dtype=np.complex64)
            intensities = np.asarray(intensities, dtype=np.complex64)
        else:
            positions = np.asarray(positions, dtype=np.float32)
            intensities = np.asarray(intensities, dtype=np.float32)

        return positions, intensities

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
def _(preprocess_peaks, preprocess_ratio, ranged):
    """Parallel preprocessing pipeline for spectra and references"""

    def preprocess_spectra(
        spectra,
        substanceDict,
        reverse,
        ranged=ranged,
        baseline_distortion=False,
        downsample=None,
        scale=None,
    ):
        """
        Complete preprocessing pipeline for a single spectrum

        Returns:
            dict: Preprocessed spectrum with intensities, positions, scales, components, ratios
        """

        if scale is not None:
            scales = scale
        else:
            scales = spectra['scales']

        new_positions, new_intensities = preprocess_peaks(
            intensities=spectra['intensities'],
            positions=spectra['positions'],
            scales=scales,
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
            'ratios': ratios,
        }

    """Preprocess function for streaming dataset"""
    return (preprocess_spectra,)


@app.cell
def _(
    StreamingNMRDataset,
    baseline_distortion,
    dataset_filepath,
    downsample,
    preprocess_peaks,
    ranged,
    reference_spectra,
    reverse,
    spectra,
    substanceDict,
):
    """Execute preprocessing pipelines"""

    # Process all training spectra
    # print("Preprocessing training spectra...")
    # preprocessed_spectra = [
    #     preprocess_spectra(
    #         spectra=spectrum,
    #         substanceDict=substanceDict,
    #         baseline_distortion=baseline_distortion,
    #         ranged=ranged,
    #         downsample=downsample,
    #         reverse=reverse,
    #     )
    #     for spectrum in spectra
    # ]

    preprocessed_spectra = StreamingNMRDataset(dataset_filepath, preprocessed_enabled=True)

    # Process reference spectra
    print("Preprocessing reference spectra...")
    preprocessed_reference_spectra = {
        spectrum: preprocess_peaks(
            intensities=reference_spectra[spectrum][0],
            positions=spectra[0]['positions'],
            scales=f'{spectrum}',
            substanceDict = substanceDict,
            ranged=ranged,
            baseline_distortion=baseline_distortion,
            downsample=downsample,
            reverse=reverse,
        )
        for spectrum in reference_spectra
    }

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
    # Number of substances
    num_substances = len(substanceDict)

    # Create a figure with subplots for each substance
    plt.figure(figsize=(15, 6 * num_substances))

    for i, substance in enumerate(substanceDict):
        spectrum_id = substanceDict[substance][0]

        # Original spectra (left panel)
        plt.subplot(num_substances, 2, 2 * i + 1)
        plt.plot(
            spectra[0]['positions'],
            reference_spectra[spectrum_id][0],
            alpha=0.7,
            label=substance
        )
        plt.title(f'Original Reference Spectrum: {substance}')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Preprocessed spectra (right panel)
        plt.subplot(num_substances, 2, 2 * i + 2)
        if reverse:
            # Time domain: plot magnitude of complex data
            complex_data = preprocessed_reference_spectra[spectrum_id][1]
            print(complex_data)
            plt.plot(complex_data, alpha=0.7, label=substance)
            plt.title(f'Preprocessed (Hilbert Transform - Time Domain): {substance}')
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
            plt.title(f'Preprocessed (Frequency Domain): {substance}')
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
            complex_data_intensities = preprocessed_spectra.__getitem__(graphcounter2, 'SP:3368')['intensities']
            complex_data_positions = preprocessed_spectra.__getitem__(graphcounter2, 'SP:3368')['positions']

            if reverse:
                # Time domain: plot magnitude of complex data
                plt.plot(complex_data_intensities)
                plt.title(f'Sample {graphcounter2} (Time Domain)')
                plt.xlabel('Time Points')
                plt.ylabel('Magnitude')
            else:
                # Frequency domain: normal plotting
                plt.plot(complex_data_positions, complex_data_intensities)
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

    return DataLoader, Dataset


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
            self.dataset_data_name = f'{dataset_name}_data'
            self.dataset_labels_name = f'{dataset_name}_labels'

            # Verify file structure and get dataset length
            with h5py.File(self.file_path, 'r') as f:
                self.length = f[f'{self.dataset_data_name}'].shape[0]

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # Load individual samples on demand to minimize memory usage
            with h5py.File(self.file_path, 'r') as f:
                data = torch.tensor(
                    f[f'{self.dataset_data_name}'][idx], 
                    dtype=torch.complex64
                )
                labels = torch.tensor(
                    f[f'{self.dataset_labels_name}'][idx], 
                    dtype=torch.float32
                )
            return data, labels

    return (StreamableNMRDataset,)


@app.cell
def _(StreamableNMRDataset, h5py, os, processed_data_dir):
    """Data persistence utilities for streamable datasets with smart caching"""

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
        try:
            with h5py.File(file_path, 'r') as f:
                data_length = f.attrs['data_length']
                train_size = f.attrs['train_size']
                val_size = f.attrs['val_size'] 
                test_size = f.attrs['test_size']
        except:
            print('Dataset corrupted, removing')
            os.remove(file_path)
            return None

        print(f"Loaded processed datasets - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        print(f"Feature vector length: {data_length}")

        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
        }, data_length

    return (load_datasets_from_files,)


@app.cell
def _(
    error,
    preprocessed_reference_spectra,
    preprocessed_spectra,
    reference_spectra,
    tqdm,
):
    """Length tester. Ensures the length of every input will always be identical. Also just so happens to cache all preprocessing"""

    import multiprocessing

    lengths = set()
    for spectrum in preprocessed_reference_spectra:
        lengths.add(len(preprocessed_reference_spectra[spectrum][0]))

    print(sorted(lengths))

    print((len(reference_spectra)+1)*len(preprocessed_spectra))

    def check_spectrum_length(args):
        i, reference_spectra, preprocessed_spectra = args
        local_lengths = set()
        for substance in reference_spectra:
            local_lengths.add(len(preprocessed_spectra.__getitem__(i, substance)['intensities']))
        local_lengths.add(len(preprocessed_spectra.__getitem__(i, None)['intensities']))
        return local_lengths

    def _():
        tasks = [(i, reference_spectra, preprocessed_spectra)
             for i in range(len(preprocessed_spectra))]

        with multiprocessing.Pool() as pool:
            results = []
            for res in tqdm(pool.imap(check_spectrum_length, tasks), total=len(tasks)):
                results.append(res)

        # Flatten and check for uniqueness
        all_lengths = set()
        for result in results:
            all_lengths.update(result)
            if len(all_lengths) > 1:
                raise error('Not all spectra are the same size')

    _()

    print(sorted(lengths))


    return


@app.cell
def _(
    h5py,
    held_back_metabolites,
    load_datasets_from_files,
    np,
    os,
    preprocessed_reference_spectra,
    preprocessed_spectra,
    processed_cache_key,
    processed_data_dir,
    random,
    substanceDict,
    tqdm,
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
        Streams data directly to HDF5 to avoid memory issues
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

        for i in tqdm(range(len(spectra)), desc="Processing Spectra"):
            spectrum = spectra[i]
            if held_back_key_test in spectrum['ratios']:
                test_with_holdback.append(i)
            elif held_back_key_validation in spectrum['ratios']:
                val_with_holdback.append(i)
            else:
                train_spectra.append(i)

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

        # Calculate dataset sizes first (count without loading data)
        train_count = 0
        for i, spec_idx in enumerate(train_spectra):
            if i in train_indices:
                for substance in reference_spectra:
                    if substance not in [held_back_key_test, held_back_key_validation]:
                        train_count += 1

        val_count = len(val_indices) + len(val_with_holdback)
        test_count = len(test_with_holdback)

        print(f'Dataset sizes:')
        print(f'  Training: {train_count} samples')
        print(f'  Validation: {val_count} samples ({len(val_with_holdback)} with {held_back_metabolites[1]})')
        print(f'  Test: {test_count} samples ({len(test_with_holdback)} with {held_back_metabolites[0]})')

        # Get sample data shape and ensure consistent data types
        sample_spectrum = spectra[0]
        sample_reference_key = list(reference_spectra.keys())[0]
        sample_reference = reference_spectra[sample_reference_key]

        # Ensure both are numpy arrays with same dtype
        spectrum_intensities = sample_spectrum['intensities']
        reference_intensities = sample_reference[1]

        sample_data = np.concatenate([spectrum_intensities, reference_intensities])
        data_length = len(sample_data)

        # Create HDF5 file for streaming
        os.makedirs(processed_data_dir, exist_ok=True)
        file_path = f'{processed_data_dir}/{processed_cache_key}_datasets.h5'

        print(f"Streaming processed datasets to {file_path}...")

        with h5py.File(file_path, 'w') as f:
            spectrumtype = sample_spectrum['intensities'].dtype
            # Create datasets with known sizes and consistent dtype
            train_data_ds = f.create_dataset(
                'train_data', 
                shape=(train_count, data_length),
                dtype=spectrumtype,
                compression='gzip', 
                compression_opts=9
            )

            val_data_ds = f.create_dataset(
                'val_data', 
                shape=(val_count, data_length),
                dtype=spectrumtype,
                compression='gzip', 
                compression_opts=9
            )

            test_data_ds = f.create_dataset(
                'test_data', 
                shape=(test_count, data_length),
                dtype=spectrumtype,
                compression='gzip', 
                compression_opts=9
            )

            train_labels_ds = f.create_dataset(
                'train_labels', 
                shape=(train_count, 2),
                dtype=np.float32,
                compression='gzip', 
                compression_opts=9
            )

            val_labels_ds = f.create_dataset(
                'val_labels', 
                shape=(val_count, 2),
                dtype=np.float32,
                compression='gzip', 
                compression_opts=9
            )

            test_labels_ds = f.create_dataset(
                'test_labels', 
                shape=(test_count, 2),
                dtype=np.float32,
                compression='gzip', 
                compression_opts=9
            )

            # Stream training data directly to HDF5
            train_idx = 0
            for i, spec_idx in tqdm(enumerate(train_spectra), total=len(train_spectra)):
                if i in train_indices:
                    for substance in reference_spectra:
                        spectrum = spectra.__getitem__(spec_idx, substance)  # Load one spectrum at a time
                        if substance not in [held_back_key_test, held_back_key_validation]:
                            # Ensure consistent data types
                            spectrum_intensities = spectrum['intensities']
                            reference_intensities = reference_spectra[substance][1]

                            # Create data sample
                            temp_data = np.concatenate([spectrum_intensities, reference_intensities])

                            # Create label
                            if substance in spectrum['ratios']:
                                temp_label = np.array([1.0, spectrum['ratios'][substance]], dtype=np.float32)
                            else:
                                temp_label = np.array([0.0, 0.0], dtype=np.float32)

                            # Write directly to HDF5
                            train_data_ds[train_idx] = temp_data
                            train_labels_ds[train_idx] = temp_label
                            train_idx += 1

            # Stream validation data (negative samples)
            val_idx = 0
            for i in tqdm(val_indices, total=len(val_indices)):
                spectrum = spectra.__getitem__(train_spectra[i], held_back_key_validation[1])
                spectrum_intensities = spectrum['intensities']
                reference_intensities = reference_spectra[held_back_key_validation][1]

                temp_data = np.concatenate([spectrum_intensities, reference_intensities])
                temp_label = np.array([0.0, 0.0], dtype=np.float32)  # Not present

                val_data_ds[val_idx] = temp_data
                val_labels_ds[val_idx] = temp_label
                val_idx += 1

            # Stream validation data (positive samples with held-back metabolite)
            for spec_idx in tqdm(val_with_holdback, total=len(val_with_holdback)):
                spectrum = spectra.__getitem__(spec_idx, held_back_key_validation[1])
                spectrum_intensities = spectrum['intensities']
                reference_intensities = reference_spectra[held_back_key_validation][1]

                temp_data = np.concatenate([spectrum_intensities, reference_intensities])
                temp_label = np.array([1.0, spectrum['ratios'][held_back_key_validation]], dtype=np.float32)

                val_data_ds[val_idx] = temp_data
                val_labels_ds[val_idx] = temp_label
                val_idx += 1

            # Stream test data (positive samples with held-back metabolite)
            test_idx = 0
            for spec_idx in tqdm(test_with_holdback, total=len(test_with_holdback)):
                spectrum = spectra.__getitem__(spec_idx, held_back_key_test[1])
                spectrum_intensities = spectrum['intensities']
                reference_intensities = reference_spectra[held_back_key_test][1]

                temp_data = np.concatenate([spectrum_intensities, reference_intensities])
                temp_label = np.array([1.0, spectrum['ratios'][held_back_key_test]], dtype=np.float32)

                test_data_ds[test_idx] = temp_data
                test_labels_ds[test_idx] = temp_label
                test_idx += 1

            # Store metadata for validation
            f.attrs['data_length'] = data_length
            f.attrs['train_size'] = train_count
            f.attrs['val_size'] = val_count
            f.attrs['test_size'] = test_count

        file_size_mb = os.path.getsize(file_path) / (1024**2)
        print(f"Processed datasets saved successfully. File size: {file_size_mb:.2f} MB")

        # Load and return streamable datasets
        return load_datasets_from_files(processed_cache_key)


    # Execute training data preparation with preprocessing-aware caching
    training_data, data_length = get_training_data_mlp(
        spectra=preprocessed_spectra,
        reference_spectra=preprocessed_reference_spectra,
        held_back_metabolites=held_back_metabolites,
        processed_cache_key=processed_cache_key,
    )

    return data_length, training_data


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
def _(nn, torch):
    """Multi-Layer Perceptron for metabolite detection and quantification"""

    class MLPRegressor(nn.Module):
        """
        MLP that properly handles complex NMR FID data
        """

        def __init__(self, input_size, trial=None, div_size=None):
            super(MLPRegressor, self).__init__()

            # Determine layer size reduction factor
            if trial is not None:
                self.div_size = trial.suggest_float('div_size', 2, 10, step=1)
            elif div_size is not None:
                self.div_size = div_size
            else:
                self.div_size = 4

            # For complex input, we need to handle real and imaginary parts
            # This doubles the effective input size
            effective_input_size = input_size * 2  # real + imaginary components

            # Calculate progressive layer sizes
            a = effective_input_size
            b = int(a / self.div_size)
            c = int(b / self.div_size)
            d = int(c / self.div_size)
            e = int(d / self.div_size)

            self.layer_sizes = [a, b, c, d, e, 2]

            # Define network architecture
            self.model = nn.Sequential(
                nn.Linear(a, b),
                nn.ReLU(),
                nn.Linear(b, c),
                nn.ReLU(),
                nn.Linear(c, d),
                nn.ReLU(),
                nn.Linear(d, e),
                nn.ReLU(),
                nn.Linear(e, 2),  # Output: [presence_logit, concentration]
            )

        def forward(self, x):
            # Properly handle complex input by concatenating real and imaginary parts
            if x.dtype.is_complex:
                # Concatenate real and imaginary parts to preserve all information
                x_real = x.real
                x_imag = x.imag
                x = torch.cat([x_real, x_imag], dim=-1)

            return self.model(x)

    return (MLPRegressor,)


@app.cell
def _(math, nn, torch):
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
        Transformer specifically designed for complex NMR FID data
        """

        def __init__(self, input_size, trial=None, **kwargs):
            super(TransformerRegressor, self).__init__()

            # Hyperparameter configuration
            if trial is not None:
                self.d_model = int(trial.suggest_categorical('d_model', [128, 256, 512]))
                self.nhead = int(trial.suggest_categorical('nhead', [8, 16]))
                self.num_layers = int(trial.suggest_int('num_layers', 3, 6))
                self.dim_feedforward = int(trial.suggest_categorical('dim_feedforward', [512, 1024, 2048]))
                self.target_seq_len = int(trial.suggest_categorical('target_seq_len', [64, 128, 256]))
            else:
                self.d_model = kwargs.get('d_model', 256)
                self.nhead = kwargs.get('nhead', 8)
                self.num_layers = kwargs.get('num_layers', 4)
                self.dim_feedforward = kwargs.get('dim_feedforward', 1024)
                self.target_seq_len = kwargs.get('target_seq_len', 128)

            # Ensure attention head compatibility
            while self.d_model % self.nhead != 0:
                self.nhead = max(1, self.nhead - 1)

            # Separate projections for real and imaginary parts
            # This preserves phase relationships in the complex data
            self.spectrum_real_projection = nn.Linear(input_size // 2, self.d_model * self.target_seq_len // 2)
            self.spectrum_imag_projection = nn.Linear(input_size // 2, self.d_model * self.target_seq_len // 2)
            self.reference_real_projection = nn.Linear(input_size // 2, self.d_model * self.target_seq_len // 2)
            self.reference_imag_projection = nn.Linear(input_size // 2, self.d_model * self.target_seq_len // 2)

            # Complex-aware positional encoding
            self.spectrum_pos_encoding = PositionalEncoding(self.d_model, 0.0, self.target_seq_len)

            # Phase-aware attention mechanism
            self.magnitude_attention = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.nhead,
                dropout=0.0,
                batch_first=True
            )

            self.phase_attention = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.nhead,
                dropout=0.0,
                batch_first=True
            )

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_layers
            )

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

            # Handle complex input properly
            if x.dtype.is_complex:
                # Split into real and imaginary components
                x_real = x.real
                x_imag = x.imag
            else:
                # If real input, create zero imaginary part
                x_real = x
                x_imag = torch.zeros_like(x)

            # Split concatenated input into spectrum and reference
            mid_point = x_real.size(1) // 2
            spectrum_real = x_real[:, :mid_point]
            spectrum_imag = x_imag[:, :mid_point]
            reference_real = x_real[:, mid_point:]
            reference_imag = x_imag[:, mid_point:]

            # Project real and imaginary parts separately
            spectrum_real_proj = self.spectrum_real_projection(spectrum_real)
            spectrum_imag_proj = self.spectrum_imag_projection(spectrum_imag)
            reference_real_proj = self.reference_real_projection(reference_real)
            reference_imag_proj = self.reference_imag_projection(reference_imag)

            # Combine real and imaginary projections
            spectrum_combined = torch.cat([spectrum_real_proj, spectrum_imag_proj], dim=-1)
            reference_combined = torch.cat([reference_real_proj, reference_imag_proj], dim=-1)

            # Reshape to sequence format
            spectrum_patches = spectrum_combined.view(batch_size, self.target_seq_len, self.d_model)
            reference_patches = reference_combined.view(batch_size, self.target_seq_len, self.d_model)

            # Add positional encoding
            spectrum_patches = self.spectrum_pos_encoding(spectrum_patches)
            reference_patches = self.spectrum_pos_encoding(reference_patches)

            # Magnitude and phase-aware attention
            magnitude_attended, _ = self.magnitude_attention(
                spectrum_patches, reference_patches, reference_patches
            )

            # Calculate phase information
            spectrum_phase = torch.atan2(spectrum_imag_proj.unsqueeze(1).expand(-1, self.target_seq_len, -1), 
                                       spectrum_real_proj.unsqueeze(1).expand(-1, self.target_seq_len, -1))
            reference_phase = torch.atan2(reference_imag_proj.unsqueeze(1).expand(-1, self.target_seq_len, -1),
                                        reference_real_proj.unsqueeze(1).expand(-1, self.target_seq_len, -1))

            # Phase-aware features (simplified for now)
            phase_features = torch.cat([spectrum_phase, reference_phase], dim=-1)
            if phase_features.size(-1) < self.d_model:
                # Pad phase features to match d_model
                padding = self.d_model - phase_features.size(-1)
                phase_features = torch.cat([phase_features, torch.zeros(batch_size, self.target_seq_len, padding, device=phase_features.device)], dim=-1)
            else:
                phase_features = phase_features[:, :, :self.d_model]

            phase_attended, _ = self.phase_attention(
                phase_features, phase_features, phase_features
            )

            # Combine magnitude and phase information
            combined_features = magnitude_attended + phase_attended

            # Self-attention processing
            encoded = self.transformer_encoder(combined_features)

            # Global average pooling
            pooled_features = torch.mean(encoded, dim=1)

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
        best_weights = copy.deepcopy(model.state_dict())  # Initialize with current weights

        for epoch in range(max_epochs):
            model.train()

            # Training loop with DataLoader
            with tqdm(train_loader, unit='batch', mininterval=0, disable=True) as bar:
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

        # Load best weights - now guaranteed to not be None
        if best_weights is not None:
            model.load_state_dict(best_weights)
        else:
            print("Warning: No improvement found during training, using final weights")

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
    np,
    os,
    partial,
    processed_cache_key,
    torch,
    tqdm,
    train_model,
    training_data,
    trials,
):
    import optuna # type: ignore

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

            if np.isnan(combined_score):
                print("NaN detected in combined_score!")
                return 4.0  # Penalty

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
        with tqdm(
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

    **Model Architecture:**

    Input Projection → Positional Encoding → Transformer Encoder → Global Average Pooling → Output Projection
    """
    elif MODEL_TYPE == 'mlp':
        # Handle sliding window MLP parameters
        model_params_md = f"""
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
