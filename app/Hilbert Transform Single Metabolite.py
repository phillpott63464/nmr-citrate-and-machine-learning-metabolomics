import marimo

__generated_with = '0.14.16'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import os

    return mo, os


@app.cell
def _(os):
    # global variables

    count = 1000
    trials = 100
    graph_count = 3
    combo_number = None
    notebook_name = 'hilbert_transform_single_metabolite'
    cache_dir = f'./data_cache/{notebook_name}'

    if os.path.exists(cache_dir) == False:
        os.mkdir(cache_dir)

    # Define metabolites and their spectrum IDs for NMR simulation
    substanceDict = {
        'Citric acid': ['SP:3368'],
        'Succinic acid': ['SP:3211'],
        'Maleic acid': ['SP:3110'],
    }
    return cache_dir, count, graph_count, notebook_name, substanceDict, trials


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate sample data""")
    return


@app.cell
def _(count, substanceDict):
    # Data generation with multiprocessing

    from morgan.createTrainingData import createTrainingData
    import numpy as np
    import multiprocessing as mp
    from functools import partial

    substanceSpectrumIds = [
        substanceDict[substance][-1] for substance in substanceDict
    ]

    def generate_batch(batch_size, substanceSpectrumIds):
        """Generate a batch of training data"""
        return createTrainingData(
            substanceSpectrumIds=substanceSpectrumIds,
            sampleNumber=batch_size,
            rondomlyScaleSubstances=True,
            # referenceSubstanceSpectrumId='dss',
            # points=2**11
            # multipletOffsetCap=1,
            # noiseRange=(2, -6)
        )

    def process_batch_data(args):
        """Process a single batch and return formatted spectra"""
        batch_data, start_idx, batch_size = args
        spectra_batch = []

        for i in range(batch_size):
            # Extract scales for this sample
            sample_scales = {
                key: [values[i]]
                for key, values in batch_data['scales'].items()
            }

            # Create spectrum dict in current format
            spectrum = {
                'scales': sample_scales,
                'intensities': batch_data['intensities'][
                    i : i + 1
                ],  # Keep 2D shape
                'positions': batch_data['positions'],
                'components': batch_data[
                    'components'
                ],  # This is shared across all samples
            }
            spectra_batch.append(spectrum)

        return spectra_batch

    # Determine optimal batch size and number of processes
    num_processes = max(1, mp.cpu_count() - 1)
    batch_size = max(1, count // num_processes)

    print(f'Using {num_processes} processes with batch size {batch_size}')

    # Generate batches of data in parallel
    batch_args = []
    remaining_samples = count

    for i in range(num_processes):
        current_batch_size = min(batch_size, remaining_samples)
        if current_batch_size <= 0:
            break

        batch_data = generate_batch(current_batch_size, substanceSpectrumIds)
        batch_args.append((batch_data, i * batch_size, current_batch_size))
        remaining_samples -= current_batch_size

    # Process batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        batch_results = pool.map(process_batch_data, batch_args)

    # Flatten results into single list
    spectra = []
    for batch in batch_results:
        spectra.extend(batch)

    # Display sample information (same as before)
    print(
        ''.join(f"{x['scales']}\n" for x in spectra[:5])
    )   # Show first 5 only
    print(spectra[0]['intensities'].shape)   # Y axis
    print(spectra[0]['positions'].shape)   # X axis
    print(spectra[0]['components'].shape)   # Peaks of all separate components

    # Get sample information for markdown display
    sample_scales_preview = '\n'.join(
        [
            f"Sample {i+1}: {spectrum['scales']}"
            for i, spectrum in enumerate(spectra[:5])
        ]
    )
    intensities_shape = spectra[0]['intensities'].shape
    positions_shape = spectra[0]['positions'].shape
    components_shape = spectra[0]['components'].shape

    return createTrainingData, mp, np, partial, spectra


@app.cell(hide_code=True)
def _(hilbertfigures, mo):
    mo.md(
        rf"""
    ## Perform Hilbert Transform

    {mo.as_html(hilbertfigures)}
    """
    )
    return


@app.cell
def _(graph_count, spectra):
    from scipy.signal import hilbert
    import matplotlib.pyplot as plt

    hilberts = []
    for index, spectrum2 in enumerate(
        spectra
    ):  # Corrected 'ennumerate' to 'enumerate'
        hilberts.append(
            (
                hilbert(
                    spectrum2['intensities'][
                        0
                    ]  # Assuming 'intensities' is a list or array
                )
            )
        )

        if (
            index > graph_count**2
        ):  # Changed 'i' to 'index' to use the correct loop variable
            break

    print(hilberts[0])

    # CreaPerformte visualization grid showing sample spectra
    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for graphcounter2 in range(1, graph_count**2 + 1):
        plt.subplot(graph_count, graph_count, graphcounter2)
        plt.plot(
            hilberts[graphcounter2],
        )

    hilbertfigures = plt.gca()

    return hilbert, hilbertfigures, hilberts, plt


@app.cell(hide_code=True)
def _(inversefigures, mo):
    mo.md(
        rf"""
    ## Perform Inverse Complex Fourier Transform

    {mo.as_html(inversefigures)}
    """
    )
    return


@app.cell
def _(graph_count, hilberts, np, plt):
    from scipy.fft import ifft

    from math import log2

    inverses, inverses_real = [], []
    for hilbertarray in hilberts:
        inv = ifft(hilbertarray)
        inv[0] = 0
        # Zero out very small values (both real and imaginary)0
        # inv_real = np.real(inv).astype(np.float32)
        threshold = 1e-16
        # threshold = np.percentile(inv.astype(np.float32), 90)
        inv_real = inv
        inv_real[np.abs(inv_real) < threshold] = 0
        inv_real = inv_real[inv_real != 0]
        print(log2(len(inv_real)))

        inverses.append(inv)
        inverses_real.append(inv_real)

    # Create visualization grid showing sample spectra
    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for idx in range(min(len(inverses), graph_count**2)):
        plt.subplot(graph_count, graph_count, idx + 1)
        plt.plot(inverses[idx])
        plt.plot(inverses_real[idx])

    inversefigures = plt.gca()
    return ifft, inversefigures, inverses, inverses_real


@app.cell
def _(mo, uninversedfigures):
    mo.md(rf"""{mo.as_html(uninversedfigures)}""")
    return


@app.cell
def _(inverses, inverses_real, np, plt):
    from scipy.fft import fft

    uninversed = []
    uninversed_real = []
    for inverse in inverses:
        uninversed.append(fft(inverse))
    for inverse_real in inverses_real:
        uninversed_real.append(fft(inverse_real))

    # Create visualization grid showing sample spectra
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(
        uninversed[0].astype(np.float32),
    )
    plt.title('Uninversed')
    plt.subplot(1, 2, 2)
    plt.plot(
        uninversed_real[0].astype(np.float32),
    )
    plt.title('Uninversed real')
    plt.legend()

    uninversedfigures = plt.gca()
    return uninversed, uninversedfigures


@app.cell
def _(hilberts, inverses, mo, uninversed):
    mo.md(
        rf"""
    ### Typing


    {hilberts[0].dtype}

    {inverses[0].dtype}

    {uninversed[0].dtype}
    """
    )
    return


@app.cell
def _(mo, referencefigure):
    mo.md(
        rf"""
    ## Generate Reference Spectra

    {mo.as_html(referencefigure)}
    """
    )
    return


@app.cell
def _(createTrainingData, hilbert, ifft, plt, spectra, substanceDict):
    # Generate pure component reference spectra (no random scaling)
    # These serve as templates for identifying substances in mixtures
    referenceSpectrumIds = [
        substanceDict[substance][-1] for substance in substanceDict
    ]

    reference_spectra_raw = createTrainingData(
        substanceSpectrumIds=referenceSpectrumIds,
        sampleNumber=1,
        rondomlyScaleSubstances=False,  # Keep original intensities for references
        # points=2**11
    )

    # Map substance names to their reference spectra
    reference_spectra = {
        substanceDict[substance][0]: reference_spectra_raw['components'][index]
        for index, substance in enumerate(substanceDict)
    }

    print(reference_spectra)

    # Visualize reference spectra for each substance
    for substance in substanceDict:
        plt.subplot(1, 2, 1)
        plt.plot(
            spectra[0]['positions'],
            reference_spectra[substanceDict[substance][0]],
        )
        plt.subplot(1, 2, 2)
        plt.plot(ifft(hilbert(reference_spectra[substanceDict[substance][0]])))

    referencefigure = plt.gca()

    return reference_spectra, referencefigure


@app.cell
def _(mo, preprocessedreferencefigure):
    mo.md(
        rf"""
    ## Preprocess Spectra

    {mo.as_html(preprocessedreferencefigure)}
    """
    )
    return


@app.cell
def _(hilbert, ifft, mp, np, plt, reference_spectra, spectra, substanceDict):
    from scipy.signal import resample
    from scipy.fft import irfft
    from scipy.interpolate import interp1d

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

        # new_intensities = (new_intensities - new_intensities.mean()) / new_intensities.std()

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
        downsample=0,
        reverse=False,
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
    # downsample = None

    def process_single_spectrum(spectrum):
        """Worker function for parallel spectrum preprocessing"""
        return preprocess_spectra(
            spectra=spectrum,
            ranges=ranges,
            substanceDict=substanceDict,
            baseline_distortion=baseline_distortion,
            downsample=downsample,
            reverse=True,
        )

    def process_single_reference(spectrum_key):
        """Worker function for parallel reference preprocessing"""
        pos_int = preprocess_peaks(
            positions=spectra[0]['positions'],
            intensities=reference_spectra[spectrum_key],
            downsample=downsample,
            reverse=True,
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
            plt.subplot(2, 2, 1)
            plt.plot(
                spectra[0]['positions'],
                reference_spectra[substanceDict[substance][0]],
            )
            plt.subplot(2, 2, 2)
            plt.plot(
                # preprocessed_spectra[0]['positions'],
                preprocessed_reference_spectra[substanceDict[substance][0]],
            )
            plt.subplot(2, 2, 3)
            plt.plot(preprocessed_spectra[0]['intensities'])

        return plt.gca()

    preprocessedreferencefigure = generate_figure()

    print(len(preprocessed_spectra[0]['positions']))
    print(len(preprocessed_spectra[0]['intensities']))
    print(len(preprocessed_spectra[1]['intensities']))
    print(len(preprocessed_spectra[2]['intensities']))

    positions_count = len(preprocessed_spectra[0]['positions'])
    intensities_count = len(preprocessed_spectra[0]['intensities'])
    return (
        preprocessed_reference_spectra,
        preprocessed_spectra,
        preprocessedreferencefigure,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Convert to Tensors""")
    return


@app.cell
def _(np, preprocessed_reference_spectra, preprocessed_spectra):
    from sklearn.model_selection import train_test_split
    import torch

    # Add device detection and setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    def get_training_data_mlp(
        spectra, reference_spectra, train_ratio=0.7, val_ratio=0.15, axes=0
    ):
        """
        Input:
            data = [
                data instances, (float32)
                axes, (x/y)
                points along each axis (float32)
            ]

            labels = array of labels (float32)

            train_ratio = ratio of data to be trained. Defaults 0.7, 70% data used for training
            val_ratio = ratio of data for validation. Defaults 0.15, 15% data used for validation
            # test_ratio = 1 - train_ratio - val_ratio (automatically calculated)

            axes = number of axes to remove. Defaults to 0 (no transformation). Use negative numbers

        Output:
            data_train = [
                data instances, (float32)
                points along both axes in single vector, (float32)
            ]
            data_val = identical
            data_test = identical
            labels_train = array of labels (float32)
            labels_val = identical
            labels_test = identical
        """
        data = []
        labels = []

        for spectrum in spectra:
            for substance in reference_spectra:
                data.append(
                    np.concatenate(
                        [spectrum['intensities'], reference_spectra[substance]]
                    )
                )
                labels.append(spectrum['ratios'][substance])

        print(spectra[0])

        print(spectra[0]['ratios'])

        data = np.array(data)

        # First split: separate out test data
        test_ratio = 1 - train_ratio - val_ratio
        data_temp, data_test, labels_temp, labels_test = train_test_split(
            data, labels, test_size=test_ratio, shuffle=True, random_state=42
        )

        # Second split: divide remaining data into train and validation
        # Calculate validation ratio relative to the remaining data
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        data_train, data_val, labels_train, labels_val = train_test_split(
            data_temp,
            labels_temp,
            test_size=val_ratio_adjusted,
            shuffle=True,
            random_state=42,
        )

        # Move tensors to GPU
        data_train = torch.tensor(data_train, dtype=torch.float32).to(device)
        labels_train = torch.tensor(labels_train, dtype=torch.float32).to(
            device
        )
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
    )

    print([training_data[x].shape for x in training_data])
    print(len(training_data['data_train'][0]))

    # Get sample ratio for display
    sample_ratio = preprocessed_spectra[0]['ratios']
    data_length = len(training_data['data_train'][0])

    return torch, training_data


@app.cell
def _():
    ## Define Training Loop
    return


@app.cell
def _(np, torch, training_data):
    import copy
    import torch.optim as optim
    import torch.nn as nn
    import tqdm
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
                nn.Linear(e, 1),  # Output: concentration
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
                self.d_model = int(
                    trial.suggest_categorical('d_model', [64, 128, 256, 512])
                )
                self.nhead = int(
                    trial.suggest_categorical('nhead', [4, 8, 16])
                )
                self.num_layers = int(trial.suggest_int('num_layers', 2, 8))
                self.dim_feedforward = int(
                    trial.suggest_categorical(
                        'dim_feedforward', [256, 512, 1024, 2048]
                    )
                )
                self.dropout = trial.suggest_float('dropout', 0.1, 0.5)
                self.max_seq_len = int(
                    trial.suggest_categorical('max_seq_len', [512, 1024, 2048])
                )
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
            self.seq_len = min(input_size // self.d_model, self.max_seq_len)
            if self.seq_len == 0:
                self.seq_len = 1
                self.d_model = input_size

            # Actual input size after reshaping
            self.actual_input_size = self.seq_len * self.d_model

            # Input projection layer to handle size mismatch
            self.input_projection = nn.Linear(
                input_size, self.actual_input_size
            )

            # Positional encoding
            self.pos_encoding = PositionalEncoding(
                self.d_model, self.dropout, self.seq_len
            )

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation='relu',
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_layers
            )

            # Output layers
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.output_projection = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model // 2, 1),
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
                'total_parameters': sum(p.numel() for p in self.parameters()),
            }

        def forward(self, x):
            # Project input to match expected size
            x = self.input_projection(x)  # [batch_size, actual_input_size]

            # Reshape to sequence format
            batch_size = x.size(0)
            x = x.view(
                batch_size, self.seq_len, self.d_model
            )  # [batch_size, seq_len, d_model]

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
            return self.dropout(x)

    def train_model(training_data, trial, model_type='mlp'):
        # Get device from training data tensors
        device = training_data['data_train'].device
        torch.cuda.empty_cache()

        # Remove n_epochs from hyperparameter search since we'll use early stopping
        # n_epochs = int(trial.suggest_float('n_epochs', 10, 100, step=10))
        max_epochs = 200  # Set a reasonable maximum
        batch_size = int(trial.suggest_float('batch_size', 10, 100, step=10))
        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        # Multi-layer perceptron with ReLU activations
        input_size = len(training_data['data_train'][0])
        if model_type == 'transformer':
            model = TransformerRegressor(
                input_size=input_size, trial=trial
            ).to(device)
        else:  # default to MLP
            model = MLPRegressor(input_size=input_size, trial=trial).to(device)

        # Data is already on GPU from get_training_data_mlp
        data_train = training_data['data_train']
        data_val = training_data['data_val']
        data_test = training_data['data_test']
        labels_train = training_data['labels_train']
        labels_val = training_data['labels_val']
        labels_test = training_data['labels_test']

        batch_start = torch.arange(0, len(data_train), batch_size)

        # Combined loss function: 0.5 * MAE + 0.5 * RMSE
        def combined_loss_fn(predictions, targets):
            mae = torch.mean(torch.abs(predictions - targets))
            mse = torch.mean((predictions - targets) ** 2)
            rmse = torch.sqrt(mse)
            return 0.5 * mae + 0.5 * rmse

        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss = np.inf
        best_weights = None
        history = []

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
                    # take a batch
                    data_batch = data_train[start : start + batch_size]
                    labels_batch = labels_train[start : start + batch_size]
                    # forward pass
                    labels_pred = model(data_batch)
                    loss = combined_loss_fn(
                        labels_pred.squeeze(), labels_batch
                    )
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(loss=float(loss))

            # Validate more frequently for better early stopping
            if epoch % validation_interval == 0 or epoch == max_epochs - 1:
                model.eval()
                with torch.no_grad():
                    labels_pred = model(data_val)
                    val_loss = combined_loss_fn(
                        labels_pred.squeeze(), labels_val
                    )
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
                        print(
                            f'Early stopping at epoch {epoch}: '
                            f'No improvement for {epochs_without_improvement} epochs'
                        )
                        break

        # Load best weights and evaluate on test set for final metrics
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            # Validation metrics (for optimization)
            val_pred = model(data_val).squeeze()
            val_mae = torch.mean(torch.abs(val_pred - labels_val))
            val_rmse = torch.sqrt(torch.mean((val_pred - labels_val) ** 2))
            ss_res_val = torch.sum((labels_val - val_pred) ** 2)
            ss_tot_val = torch.sum((labels_val - torch.mean(labels_val)) ** 2)
            val_r2_score = 1 - (ss_res_val / ss_tot_val)

            # Test metrics (for final evaluation)
            test_pred = model(data_test).squeeze()
            test_mae = torch.mean(torch.abs(test_pred - labels_test))
            test_rmse = torch.sqrt(torch.mean((test_pred - labels_test) ** 2))
            ss_res_test = torch.sum((labels_test - test_pred) ** 2)
            ss_tot_test = torch.sum(
                (labels_test - torch.mean(labels_test)) ** 2
            )
            test_r2_score = 1 - (ss_res_test / ss_tot_test)

        # Return validation metrics for optimization and test metrics for final evaluation
        return (
            val_mae,
            val_rmse,
            val_r2_score,
            test_mae,
            test_rmse,
            test_r2_score,
        )

    # Get device info from training data
    device_info = training_data['data_train'].device
    gpu_name = ''
    if torch.cuda.is_available():
        gpu_name = f' ({torch.cuda.get_device_name(0)})'
    return tqdm, train_model


@app.cell(hide_code=True)
def _(mo, training_data):
    mo.md(
        rf"""
    ## Hyperparameter Optimisation

    {training_data['data_train'].shape}
    {training_data['data_train'].dtype}
    """
    )
    return


@app.cell
def _(
    cache_dir,
    notebook_name,
    partial,
    tqdm,
    train_model,
    training_data,
    trials,
):
    import optuna

    def objective(training_data, trial):
        """
        Optuna objective function for hyperparameter optimization.
        """
        (
            val_mae,
            val_rmse,
            val_r2_score,
            test_mae,
            test_rmse,
            test_r2_score,
        ) = train_model(training_data, trial, model_type='mlp')

        # Store all metrics in trial for later analysis
        trial.set_user_attr('val_mae', float(val_mae))
        trial.set_user_attr('val_rmse', float(val_rmse))
        trial.set_user_attr('val_r2_score', float(val_r2_score))
        trial.set_user_attr('test_mae', float(test_mae))
        trial.set_user_attr('test_rmse', float(test_rmse))
        trial.set_user_attr('test_r2_score', float(test_r2_score))

        # Optimize for combined MAE + RMSE (lower is better)
        combined_score = 0.5 * val_mae + 0.5 * val_rmse
        return combined_score

    # Create or load existing Optuna study with persistent SQLite storage
    study = optuna.create_study(
        direction='minimize',  # Minimize the combined MAE + RMSE score
        study_name=notebook_name,  # Use cache key for unique study identification
        storage=f'sqlite:///{cache_dir}/{notebook_name}.db',  # Use cache key for database filename
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

    # Store optimization results for display
    best_trial = study.best_trial
    best_params = best_trial.params

    # Extract all metrics from best trial
    val_mae = best_trial.user_attrs.get('val_mae', 0)
    val_rmse = best_trial.user_attrs.get('val_rmse', 0)
    val_r2_score = best_trial.user_attrs.get('val_r2_score', 0)
    test_mae = best_trial.user_attrs.get('test_mae', 0)
    test_rmse = best_trial.user_attrs.get('test_rmse', 0)
    test_r2_score = best_trial.user_attrs.get('test_r2_score', 0)

    # Calculate statistics for all completed trials
    trial_values = [t.value for t in study.trials if t.value is not None]
    optimization_stats = {
        'total_trials': len(study.trials),
        'completed_trials': len(
            [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        ),
        'pruned_trials': len(
            [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ]
        ),
        'best_value': min(trial_values) if trial_values else 0,
        'worst_value': max(trial_values) if trial_values else 0,
        'mean_value': sum(trial_values) / len(trial_values)
        if trial_values
        else 0,
    }

    return (
        best_params,
        best_trial,
        optimization_stats,
        test_mae,
        test_r2_score,
        test_rmse,
        val_mae,
        val_r2_score,
        val_rmse,
    )


@app.cell
def _(
    best_params,
    best_trial,
    mo,
    optimization_stats,
    test_mae,
    test_r2_score,
    test_rmse,
    val_mae,
    val_r2_score,
    val_rmse,
):
    # Format best parameters for display
    params_str = '\n'.join(
        [f'- **{key}**: {value}' for key, value in best_params.items()]
    )

    mo.md(
        rf"""
    ## Hyperparameter Optimisation

    ### Optimization Results

    **Study Configuration:**

    - Direction: Minimize combined MAE + RMSE score (0.5 * MAE + 0.5 * RMSE)
    - Total trials: {optimization_stats['total_trials']}
    - Completed trials: {optimization_stats['completed_trials']}
    - Pruned trials: {optimization_stats['pruned_trials']}

    ### Best Trial Performance

    **Validation Set (Optimization Target):**

    - **Combined Score (0.5 * MAE + 0.5 * RMSE):** {best_trial.value:.6f}
    - **R² Score:** {val_r2_score:.6f}
    - **MAE:** {val_mae:.6f}
    - **RMSE:** {val_rmse:.6f}

    **Test Set (Final Evaluation):**

    - **R² Score:** {test_r2_score:.6f}
    - **MAE:** {test_mae:.6f}
    - **RMSE:** {test_rmse:.6f}

    **Best Parameters:**
    {params_str}

    ### Performance Statistics (Combined MAE + RMSE)

    - **Best value:** {optimization_stats['best_value']:.6f}
    - **Worst value:** {optimization_stats['worst_value']:.6f}
    - **Mean value:** {optimization_stats['mean_value']:.6f}
    """
    )
    return


if __name__ == '__main__':
    app.run()
