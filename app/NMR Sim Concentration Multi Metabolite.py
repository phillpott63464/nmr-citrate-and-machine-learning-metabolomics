import marimo

__generated_with = "0.14.13"
app = marimo.App(
    width="medium",
    layout_file="layouts/NMR Sim Concentration Multi Metabolite.slides.json",
)


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
    # Data generation

    from morgan.createTrainingData import createTrainingData
    import numpy as np

    substanceDict = {
        'Citric acid': ['SP:3368'],
        'Succinic acid': ['SP:3211'],
        'Maleic acid': ['SP:3110'],
        'Lactic acid': ['SP:3675'],
    }

    substanceSpectrumIds = [substanceDict[substance][-1] for substance in substanceDict]
    count = 100

    # Use the built-in loop capability of createTrainingData
    batch_data = createTrainingData(
        substanceSpectrumIds=substanceSpectrumIds,
        sampleNumber=count,  # This creates 1000 samples in one call
        rondomlyScaleSubstances=True,
        # referenceSubstanceSpectrumId='dss',
    )

    # Extract data into your current format
    labels = []
    spectra = []

    for i in range(count):
        # Extract scales for this sample
        sample_scales = {key: [values[i]] for key, values in batch_data['scales'].items()}

        # Create spectrum dict in your current format
        spectrum = {
            'scales': sample_scales,
            'intensities': batch_data['intensities'][i:i+1],  # Keep 2D shape
            'positions': batch_data['positions'],
            'components': batch_data['components']  # This is shared across all samples
        }
        spectra.append(spectrum)

    print(''.join(f"{x['scales']}\n'" for x in spectra[:5])) # Show first 5 only
    print(spectra[0]['intensities'].shape) #Y axis
    print(spectra[0]['positions'].shape) #X axis
    print(spectra[0]['components'].shape) #Peaks of all separate components

    # Get sample information for markdown display
    sample_scales_preview = '\n'.join([f"Sample {i+1}: {spectrum['scales']}" for i, spectrum in enumerate(spectra[:5])])
    intensities_shape = spectra[0]['intensities'].shape
    positions_shape = spectra[0]['positions'].shape
    components_shape = spectra[0]['components'].shape

    return (
        components_shape,
        count,
        intensities_shape,
        labels,
        np,
        positions_shape,
        sample_scales_preview,
        spectra,
        substanceDict,
    )


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

    plt.figure(figsize=(graph_count*4, graph_count*4))

    for graphcounter in range(1, graph_count**2+1):
        plt.subplot(graph_count, graph_count, graphcounter)
        # plt.title(f'{round(labels[graphcounter], 3)}M')
        plt.plot(spectra[graphcounter]['positions'], spectra[graphcounter]['intensities'][0])

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
def _(mo, preprocessedfigure):
    mo.md(
        rf"""
    ## Spectra Preprocessing
    - Extract only the relevant parts of the spectra
    - Add baseline distortion
    - Extract ratio of citrate to reference

    {mo.as_html(preprocessedfigure)}
    """
    )
    return


@app.cell
def _(np, spectra, substanceDict):
    def preprocess_peaks(intensities, positions, ranges, baseline_distortion=False):
        indices_range = []
        for range in ranges:
            indices_range.append(np.where(
                (positions >= range[0]) & (positions <= range[1])
            )[0])

        new_intensities = []
        new_positions = []

        for range in indices_range:
            temp_intensities = np.hstack(intensities[:, range])
            temp_positions = [positions[x] for x in range]

            # Add baseline distortion if enabled
            if baseline_distortion:
                # Normalize positions to [0, 1] for consistent baseline calculation
                x_normalized = (np.array(temp_positions) - np.min(temp_positions)) / (np.max(temp_positions) - np.min(temp_positions))
                baseline = 0.02 * np.sin(0.5 * np.pi * x_normalized)
                temp_intensities = temp_intensities + baseline

            new_intensities = np.concatenate([new_intensities, temp_intensities])
            new_positions = np.concatenate([new_positions, temp_positions])

        return new_positions, new_intensities

    def preprocess_ratio(scales, substanceDict):
        # referenceScale = scales['tsp'][0]

        # substanceIds = [substanceDict[id][0] for id in substanceDict]

        # substanceScales = [scales[id][0] for id in substanceIds]

        # ratio = [scale/referenceScale for scale in substanceScales]

        ratio = scales['SP:3368'][0]/scales['tsp'][0]

        return ratio

    def preprocess_spectra(spectra, ranges, substanceDict, baseline_distortion=False):
        new_positions, new_intensities = preprocess_peaks(spectra['intensities'], spectra['positions'], ranges, baseline_distortion)

        ratio = preprocess_ratio(spectra['scales'], substanceDict)

        return {
            'intensities': new_intensities,
            'positions': new_positions,
            'scales': spectra['scales'],
            'components': spectra['components'],
            'ratio': ratio,
        }

    # ranges = [
    #     [2.2, 2.8],
    #     [-0.1, 0.1],
    # ]

    ranges = [[-100, 100]]

    # Add baseline distortion flag
    baseline_distortion = True  # Set to False to disable baseline distortion



    def _(spectra):
        preprocessed_spectra = []
        for spectrum in spectra:
            preprocessed_spectra.append(preprocess_spectra(spectrum, ranges, substanceDict, baseline_distortion))

        return preprocessed_spectra

    preprocessed_spectra = _(spectra)

    print(len(preprocessed_spectra[0]['positions']))
    print(len(preprocessed_spectra[0]['intensities']))

    positions_count = len(preprocessed_spectra[0]['positions'])
    intensities_count = len(preprocessed_spectra[0]['intensities'])
    return (
        baseline_distortion,
        intensities_count,
        positions_count,
        preprocessed_spectra,
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

    plt.figure(figsize=(graph_count*4, graph_count*4))

    for graphcounter2 in range(1, graph_count**2+1):
        plt.subplot(graph_count, graph_count, graphcounter2)
        # plt.title(f'{round(labels[graphcounter], 3)}M')
        plt.plot(preprocessed_spectra[graphcounter2]['positions'], preprocessed_spectra[graphcounter2]['intensities'])

    preprocessedfigure = plt.gca()
    return (preprocessedfigure,)


@app.cell
def _(np, preprocessed_spectra, torch):
    from sklearn.model_selection import train_test_split

    # Add device detection and setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    def get_training_data_mlp(spectra, train_ratio=0.7, val_ratio=0.15, axes=0):
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
            data.append(np.concatenate([spectrum['intensities'], spectrum['positions']]))
            labels.append(spectrum['ratio'])

        print(spectra[0]['ratio'])

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
            data_temp, labels_temp, test_size=val_ratio_adjusted, shuffle=True, random_state=42
        )

        # Move tensors to GPU
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
            'labels_test': labels_test
        }

    training_data = get_training_data_mlp(spectra=preprocessed_spectra)

    print([training_data[x].shape for x in training_data])
    print(len(training_data['data_train'][0]))

    # Get sample ratio for display
    sample_ratio = preprocessed_spectra[0]['ratio']
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

    - **Sample Ratio (first spectrum):** {sample_ratio:.6f}

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

        n_epochs=int(trial.suggest_float('n_epochs', 10, 100, step=10))
        batch_size=int(trial.suggest_float('batch_size', 10, 100, step=10))
        lr=trial.suggest_float('lr', 1e-5, 1e-1)
        div_size=trial.suggest_float('div_size', 2, 10, step=1)

        a = len(training_data['data_train'][0])
        b = int(a/div_size)
        c = int(b/div_size)
        d = int(c/div_size)
        e = int(d/div_size)

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
            nn.Linear(e, 1),  
        ).to(device)

        # Data is already on GPU from get_training_data_mlp
        data_train = training_data['data_train']
        data_val = training_data['data_val']
        data_test = training_data['data_test']
        labels_train = training_data['labels_train']
        labels_val = training_data['labels_val']
        labels_test = training_data['labels_test']

        batch_start = torch.arange(0, len(data_train), batch_size)

        # loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

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
                    labels_pred = model(data_batch)
                    loss = loss_fn(labels_pred.squeeze(), labels_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(loss=float(loss))

            # evaluate on validation set at end of each epoch
            model.eval()
            with torch.no_grad():
                labels_pred = model(data_val)
                val_loss = loss_fn(labels_pred.squeeze(), labels_val)
                val_loss = float(val_loss)
                history.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = copy.deepcopy(model.state_dict())

        # Load best weights and evaluate on test set for final metrics
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            # Validation metrics (for optimization)
            val_pred = model(data_val).squeeze()
            val_mae = torch.mean(torch.abs(val_pred - labels_val))
            val_rmse = torch.sqrt(torch.mean((val_pred - labels_val)**2))
            ss_res_val = torch.sum((labels_val - val_pred) ** 2)
            ss_tot_val = torch.sum((labels_val - torch.mean(labels_val)) ** 2)
            val_r2_score = 1 - (ss_res_val / ss_tot_val)

            # Test metrics (for final evaluation)
            test_pred = model(data_test).squeeze()
            test_mae = torch.mean(torch.abs(test_pred - labels_test))
            test_rmse = torch.sqrt(torch.mean((test_pred - labels_test)**2))
            ss_res_test = torch.sum((labels_test - test_pred) ** 2)
            ss_tot_test = torch.sum((labels_test - torch.mean(labels_test)) ** 2)
            test_r2_score = 1 - (ss_res_test / ss_tot_test)

        # Return validation metrics for optimization and test metrics for final evaluation
        return val_mae, val_rmse, val_r2_score, test_mae, test_rmse, test_r2_score

    # Get device info from training data
    device_info = training_data['data_train'].device
    gpu_name = ""
    if torch.cuda.is_available():
        gpu_name = f" ({torch.cuda.get_device_name(0)})"
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

    - **R² Score: {study.best_trial.value:.4f}** (Coefficient of determination - measures how well the model explains variance in the data. Range: 0-1, higher is better)
    - **MAE: {study.best_trial.user_attrs['val_mae']:.6f}** (Mean Absolute Error - average absolute difference between predictions and true values. Lower is better)
    - **RMSE: {study.best_trial.user_attrs['val_rmse']:.6f}** (Root Mean Square Error - penalizes larger errors more heavily than MAE. Lower is better)

    **Final Test Set Performance:**

    - **R² Score: {study.best_trial.user_attrs['test_r2_score']:.4f}**
    - **MAE: {study.best_trial.user_attrs['test_mae']:.6f}**
    - **RMSE: {study.best_trial.user_attrs['test_rmse']:.6f}**

    **Best Hyperparameters:**

    - **Number of Epochs:** {study.best_trial.params['n_epochs']:.0f}
    - **Batch Size:** {study.best_trial.params['batch_size']:.0f}
    - **Learning Rate:** {study.best_trial.params['lr']:.2e}
    - **Division Size:** {study.best_trial.params['div_size']:.0f} (controls network width - smaller values = wider layers)

    **Model Architecture:**

    Input size → {int(study.best_trial.params['div_size'])} divisions → ... → 1 output

    **Data Split:**

    - Training: 70% 
    - Validation: 15% (used for hyperparameter optimization)
    - Test: 15% (held out for final evaluation)

    **Total Trials Completed:** {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}
    """
    )
    return


@app.cell
def _(train_mlp_model, training_data, val_r22):
    # Hyperparameter optimisation loop

    import optuna
    from functools import partial

    trials = 10

    def objective(training_data, trial):
        val_mae, val_rmse, val_r2, test_mae, test_rmse, test_r2 = train_mlp_model(training_data, trial)

        # Store all metrics in the trial for later analysis
        trial.set_user_attr('val_mae', float(val_mae))
        trial.set_user_attr('val_rmse', float(val_rmse))
        trial.set_user_attr('val_r2_score', float(val_r2))
        trial.set_user_attr('test_mae', float(test_mae))
        trial.set_user_attr('test_rmse', float(test_rmse))
        trial.set_user_attr('test_r2_score', float(test_r2))

        # Optimize for validation R² score (higher is better)
        return val_r22

    study = optuna.create_study(
        direction='maximize',
        study_name='reference_concentration_study',
        storage='sqlite:///reference_concentration_study.db',
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
        study.optimize(
            partial(objective, training_data),
            callbacks=[
                optuna.study.MaxTrialsCallback(
                    trials, states=(optuna.trial.TrialState.COMPLETE,)
                )
            ],
        )

    return optuna, study


if __name__ == "__main__":
    app.run()
