import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    # Store device info for markdown display
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
def _():
    # Data generation with multiprocessing

    from morgan.createTrainingData import createTrainingData
    import numpy as np
    import itertools
    import multiprocessing as mp
    import os

    substanceDict = {
        'Citric acid': ['SP:3368'],
        'Succinic acid': ['SP:3211'],
        'Maleic acid': ['SP:3110'],
        'Lactic acid': ['SP:3675'],
    }

    referenceSpectrumIds = [
        substanceDict[substance][-1] for substance in substanceDict
    ]

    substances = list(substanceDict.keys())

    combinations = []
    for r in range(1, len(substances) + 1):  # r goes from 1 to the number of substances
        for combo in itertools.combinations(substances, r):
            combo_dict = {substance: substanceDict[substance] for substance in combo}
            combinations.append(combo_dict)

    substanceSpectrumIds = [
        [combination[substance][-1] for substance in combination]
        for combination in combinations
    ]

    count = 1000

    def create_batch_data(substances_and_count):
        """Worker function for multiprocessing"""
        substances, sample_count = substances_and_count
        return createTrainingData(
            substanceSpectrumIds=substances,
            sampleNumber=sample_count,
            rondomlyScaleSubstances=True,
        )

    # Prepare arguments for multiprocessing
    mp_args = [(substances, count) for substances in substanceSpectrumIds]

    # Determine number of processes (use CPU count - 1 to leave one core free)
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} processes for data generation")

    # Generate data using multiprocessing
    batch_data = []
    if len(mp_args) > 1:  # Only use multiprocessing if we have multiple batches
        with mp.Pool(processes=num_processes) as pool:
            batch_data = pool.map(create_batch_data, mp_args)
    else:
        # Single batch - no need for multiprocessing
        batch_data = [create_batch_data(mp_args[0])]

    print(f"Generated {len(batch_data)} batches")

    # Extract data into your current format
    spectra = []
    labels = []

    for batch in batch_data:
        for i in range(count):
            # Extract scales for this sample
            sample_scales = {
                key: [values[i]] for key, values in batch['scales'].items()
            }

            # Create spectrum dict in your current format
            spectrum = {
                'scales': sample_scales,
                'intensities': batch['intensities'][
                    i : i + 1
                ],  # Keep 2D shape
                'positions': batch['positions'],
                'components': batch[
                    'components'
                ],  # This is shared across all samples
            }
            spectra.append(spectrum)

    print(
        ''.join(f"{x['scales']}\n'" for x in spectra[:5])
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

    return (
        components_shape,
        count,
        createTrainingData,
        intensities_shape,
        labels,
        mp,
        np,
        positions_shape,
        referenceSpectrumIds,
        sample_scales_preview,
        spectra,
        substanceDict,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(
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

    **Generated {count} samples successfully**

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
def _(labels, spectra):
    import matplotlib.pyplot as plt

    print(len(spectra))
    print(len(labels))

    graph_count = 3

    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for graphcounter in range(1, graph_count**2 + 1):
        plt.subplot(graph_count, graph_count, graphcounter)
        # plt.title(f'{round(labels[graphcounter], 3)}M')
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
    reference_spectra_raw = createTrainingData(
        substanceSpectrumIds=referenceSpectrumIds,
        sampleNumber=1,  # This creates 1000 samples in one call
        rondomlyScaleSubstances=False,
        # referenceSubstanceSpectrumId='dss',
    )

    reference_spectra = {
        substanceDict[substance][0]: reference_spectra_raw['components'][index]
        for index, substance in enumerate(substanceDict)
    }

    print(reference_spectra)

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
        indices_range = []
        for range_item in ranges:  # Changed from 'range' to 'range_item'
            indices_range.append(
                np.where(
                    (positions >= range_item[0]) & (positions <= range_item[1])
                )[0]
            )

        new_intensities = []
        new_positions = []

        for (
            range_indices
        ) in indices_range:  # Changed from 'range' to 'range_indices'
            # Fix: properly extract intensities - flatten the 2D array for this range
            temp_intensities = intensities[
                range_indices
            ]  # Take first row and specified indices
            temp_positions = positions[
                range_indices
            ]  # Direct indexing for positions

            # Add baseline distortion if enabled
            if baseline_distortion:
                # Normalize positions to [0, 1] for consistent baseline calculation
                if len(temp_positions) > 1:  # Avoid division by zero
                    x_normalized = (
                        temp_positions - np.min(temp_positions)
                    ) / (np.max(temp_positions) - np.min(temp_positions))
                    baseline = 0.02 * np.sin(0.5 * np.pi * x_normalized)
                    temp_intensities = temp_intensities + baseline

            new_intensities = np.concatenate(
                [new_intensities, temp_intensities]
            )
            new_positions = np.concatenate([new_positions, temp_positions])

        if downsample > 0:
            # For NMR data, positions are typically in descending order
            # Sort the data by position to ensure proper binning
            sorted_indices = np.argsort(new_positions)
            new_positions = new_positions[sorted_indices]
            new_intensities = new_intensities[sorted_indices]

            min_pos, max_pos = new_positions.min(), new_positions.max()
            bin_edges = np.linspace(min_pos, max_pos, downsample + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            downsampled_intensities = np.zeros(downsample)

            for i in range(downsample):
                mask = (new_positions >= bin_edges[i]) & (
                    new_positions < bin_edges[i + 1]
                )
                if np.any(mask):
                    # Use interpolation to better preserve peak shapes
                    # This approach maintains area while giving better peak representation
                    bin_width = bin_edges[i + 1] - bin_edges[i]
                    if len(new_positions[mask]) > 1:
                        # Use trapezoidal integration but normalize by actual data spacing
                        area = np.trapz(
                            new_intensities[mask], new_positions[mask]
                        )
                        downsampled_intensities[i] = area / bin_width
                    else:
                        # Single point - just use the intensity value
                        downsampled_intensities[i] = new_intensities[mask][0]
                else:
                    downsampled_intensities[i] = 0.0

            new_intensities = downsampled_intensities
            new_positions = bin_centers

        return new_positions, new_intensities

    def preprocess_ratio(scales, substanceDict):
        ratios = {
            substance: scales[substance][0]
            # substanceDict[substance][0]: scales[substanceDict[substance][0]][0]
            / scales['tsp'][0]
            for substance in scales
        }

        return ratios

    def preprocess_spectra(
        spectra, ranges, substanceDict, baseline_distortion=False, downsample=0
    ):
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

    ranges = [[-100, 100]]
    baseline_distortion = True
    downsample = int(2048)

    # Helper function for multiprocessing spectrum processing
    def process_single_spectrum(spectrum):
        return preprocess_spectra(
            spectra=spectrum,
            ranges=ranges,
            substanceDict=substanceDict,
            baseline_distortion=baseline_distortion,
            downsample=downsample,
        )

    # Helper function for multiprocessing reference processing
    def process_single_reference(spectrum_key):
        pos_int = preprocess_peaks(
            positions=spectra[0]['positions'],
            intensities=reference_spectra[spectrum_key],
            downsample=2048,
        )
        return (spectrum_key, pos_int[1])  # Return key and processed intensities

    def process_spectra(spectra):
        # Determine number of processes (use CPU count - 1 to leave one core free)
        num_processes = max(1, mp.cpu_count() - 1)
        print(f"Using {num_processes} processes for spectra preprocessing")

        # Process spectra in parallel
        with mp.Pool(processes=num_processes) as pool:
            preprocessed_spectra = pool.map(process_single_spectrum, spectra)

        return preprocessed_spectra

    def process_references(reference_spectra):
        # Determine number of processes (use CPU count - 1 to leave one core free)
        num_processes = max(1, mp.cpu_count() - 1)
        print(f"Using {num_processes} processes for reference preprocessing")

        # Process references in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_single_reference, reference_spectra.keys())

        # Convert results back to dictionary
        preprocessed_reference_spectra = {key: intensities for key, intensities in results}

        return preprocessed_reference_spectra

    preprocessed_spectra = process_spectra(spectra)

    preprocessed_reference_spectra = process_references(reference_spectra)

    def generate_figure():

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
def _(graph_count, labels, plt, preprocessed_spectra, spectra):
    print(len(spectra))
    print(len(labels))

    plt.figure(figsize=(graph_count * 4, graph_count * 4))

    for graphcounter2 in range(1, graph_count**2 + 1):
        plt.subplot(graph_count, graph_count, graphcounter2)
        # plt.title(f'{round(labels[graphcounter], 3)}M')
        plt.plot(
            spectra[graphcounter2]['positions'],
            spectra[graphcounter2]['intensities'][0],
        )
        plt.plot(
            preprocessed_spectra[graphcounter2]['positions'],
            preprocessed_spectra[graphcounter2]['intensities'],
        )

    preprocessedfigure = plt.gca()
    return (preprocessedfigure,)


@app.cell
def _(np, preprocessed_reference_spectra, preprocessed_spectra, torch):
    from sklearn.model_selection import train_test_split

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
                        [
                            spectrum['intensities'],
                            spectrum['positions'],
                            reference_spectra[substance],
                        ]
                    )
                )

                if substance in spectrum['ratios']:
                    labels.append([
                        1,
                        spectrum['ratios'][substance]
                    ])
                else:
                    labels.append([0, 0])

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
def _(np, torch, training_data):
    import tqdm
    import copy
    import torch.optim as optim
    import torch.nn as nn

    def train_mlp_model(training_data, trial):
        # Get device from training data tensors
        device = training_data['data_train'].device

        n_epochs = int(trial.suggest_float('n_epochs', 10, 100, step=10))
        batch_size = int(trial.suggest_float('batch_size', 10, 100, step=10))
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        div_size = trial.suggest_float('div_size', 2, 10, step=1)

        # Add hyperparameter for loss weighting
        loss_weight = trial.suggest_float('loss_weight', 0.1, 10.0)

        a = len(training_data['data_train'][0])
        b = int(a / div_size)
        c = int(b / div_size)
        d = int(c / div_size)
        e = int(d / div_size)

        # Define the model and move to GPU
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

        # Data is already on GPU from get_training_data_mlp
        data_train = training_data['data_train']
        data_val = training_data['data_val']
        data_test = training_data['data_test']
        labels_train = training_data['labels_train']
        labels_val = training_data['labels_val']
        labels_test = training_data['labels_test']

        batch_start = torch.arange(0, len(data_train), batch_size)

        # Define loss functions
        bce_loss = nn.BCEWithLogitsLoss()  # For presence prediction (classification)
        mse_loss = nn.MSELoss(reduction='none')  # For concentration prediction (regression)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        def compute_loss(predictions, targets):
            # Split predictions and targets
            presence_logits = predictions[:, 0]  # Raw logits for presence
            concentration_pred = predictions[:, 1]  # Concentration predictions

            presence_true = targets[:, 0]  # True presence (0 or 1)
            concentration_true = targets[:, 1]  # True concentration

            # Classification loss for presence
            classification_loss = bce_loss(presence_logits, presence_true)

            # Regression loss for concentration (only where substance is present)
            concentration_loss = mse_loss(concentration_pred, concentration_true)
            # Weight the concentration loss by the true presence
            weighted_concentration_loss = torch.mean(concentration_loss * presence_true)

            # Combine losses
            total_loss = classification_loss + loss_weight * weighted_concentration_loss

            return total_loss, classification_loss, weighted_concentration_loss

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
                    # take a batch
                    data_batch = data_train[start : start + batch_size]
                    labels_batch = labels_train[start : start + batch_size]
                    # forward pass
                    predictions = model(data_batch)
                    total_loss, class_loss, conc_loss = compute_loss(predictions, labels_batch)
                    # backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(
                        total_loss=float(total_loss),
                        class_loss=float(class_loss),
                        conc_loss=float(conc_loss)
                    )

            # evaluate on validation set at end of each epoch
            model.eval()
            with torch.no_grad():
                predictions = model(data_val)
                val_loss, _, _ = compute_loss(predictions, labels_val)
                val_loss = float(val_loss)
                history.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = copy.deepcopy(model.state_dict())

        # Load best weights and evaluate on test set for final metrics
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            # Validation metrics
            val_pred = model(data_val)
            val_presence_logits = val_pred[:, 0]
            val_concentration_pred = val_pred[:, 1]
            val_presence_pred = torch.sigmoid(val_presence_logits)  # Convert to probabilities

            val_presence_true = labels_val[:, 0]
            val_concentration_true = labels_val[:, 1]

            # Classification metrics (presence)
            val_presence_binary = (val_presence_pred > 0.5).float()
            val_accuracy = torch.mean((val_presence_binary == val_presence_true).float())

            # Regression metrics (concentration, only for present substances)
            present_mask = val_presence_true == 1
            if present_mask.sum() > 0:
                val_conc_mae = torch.mean(torch.abs(
                    val_concentration_pred[present_mask] - val_concentration_true[present_mask]
                ))
                val_conc_rmse = torch.sqrt(torch.mean(
                    (val_concentration_pred[present_mask] - val_concentration_true[present_mask]) ** 2
                ))
                # R² for concentration
                ss_res_val = torch.sum((val_concentration_true[present_mask] - val_concentration_pred[present_mask]) ** 2)
                ss_tot_val = torch.sum((val_concentration_true[present_mask] - torch.mean(val_concentration_true[present_mask])) ** 2)
                val_conc_r2 = 1 - (ss_res_val / ss_tot_val)
            else:
                val_conc_mae = torch.tensor(0.0)
                val_conc_rmse = torch.tensor(0.0)
                val_conc_r2 = torch.tensor(0.0)

            # Test metrics
            test_pred = model(data_test)
            test_presence_logits = test_pred[:, 0]
            test_concentration_pred = test_pred[:, 1]
            test_presence_pred = torch.sigmoid(test_presence_logits)

            test_presence_true = labels_test[:, 0]
            test_concentration_true = labels_test[:, 1]

            # Classification metrics (presence)
            test_presence_binary = (test_presence_pred > 0.5).float()
            test_accuracy = torch.mean((test_presence_binary == test_presence_true).float())

            # Regression metrics (concentration, only for present substances)
            test_present_mask = test_presence_true == 1
            if test_present_mask.sum() > 0:
                test_conc_mae = torch.mean(torch.abs(
                    test_concentration_pred[test_present_mask] - test_concentration_true[test_present_mask]
                ))
                test_conc_rmse = torch.sqrt(torch.mean(
                    (test_concentration_pred[test_present_mask] - test_concentration_true[test_present_mask]) ** 2
                ))
                # R² for concentration
                ss_res_test = torch.sum((test_concentration_true[test_present_mask] - test_concentration_pred[test_present_mask]) ** 2)
                ss_tot_test = torch.sum((test_concentration_true[test_present_mask] - torch.mean(test_concentration_true[test_present_mask])) ** 2)
                test_conc_r2 = 1 - (ss_res_test / ss_tot_test)
            else:
                test_conc_mae = torch.tensor(0.0)
                test_conc_rmse = torch.tensor(0.0)
                test_conc_r2 = torch.tensor(0.0)

        # Return validation metrics for optimization and test metrics for final evaluation
        return (
            float(val_accuracy),  # Primary metric for optimization
            float(val_conc_r2),
            float(val_conc_mae),
            float(val_conc_rmse),
            float(test_accuracy),
            float(test_conc_r2),
            float(test_conc_mae),
            float(test_conc_rmse),
        )

    # Get device info from training data
    device_info = training_data['data_train'].device
    gpu_name = ''
    if torch.cuda.is_available():
        gpu_name = f' ({torch.cuda.get_device_name(0)})'
    return device_info, gpu_name, tqdm, train_mlp_model


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
def _(tqdm, train_mlp_model, training_data):
    # Hyperparameter optimisation loop

    import optuna
    from functools import partial

    trials = 1000

    def objective(training_data, trial):
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

        # Store all metrics
        trial.set_user_attr('val_accuracy', val_accuracy)
        trial.set_user_attr('val_conc_r2', val_conc_r2)
        trial.set_user_attr('val_conc_mae', val_conc_mae)
        trial.set_user_attr('val_conc_rmse', val_conc_rmse)
        trial.set_user_attr('test_accuracy', test_accuracy)
        trial.set_user_attr('test_conc_r2', test_conc_r2)
        trial.set_user_attr('test_conc_mae', test_conc_mae)
        trial.set_user_attr('test_conc_rmse', test_conc_rmse)

        # Optimize for a combination of both tasks
        # You can adjust these weights based on which task is more important
        combined_score = 0.5 * val_accuracy + 0.5 * val_conc_r2
        return combined_score

    study = optuna.create_study(
        direction='maximize',
        study_name='metabolite_randomisation',
        storage='sqlite:///model_database/metabolite_randomisation.db',
        load_if_exists=True,
    )

    completed_trials = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    )

    if trials - completed_trials > 0:
        with tqdm.tqdm(total=trials - completed_trials, desc="Optimizing") as pbar:
            def callback(study, trial):
                pbar.update(1)
            
            study.optimize(
                partial(objective, training_data),
                callbacks=[
                    optuna.study.MaxTrialsCallback(
                        trials, states=(optuna.trial.TrialState.COMPLETE,)
                    ),
                    callback
                ],
            )

    return optuna, study


if __name__ == "__main__":
    app.run()
