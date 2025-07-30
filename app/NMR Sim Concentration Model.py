import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## NMR Sim Concentration Model
    - Taken using knowledge learn from NMR Sim Test.py, but cleaned up, and with certain aspects removed
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define a spectrum creation and normalisation function""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, create multiple of these spectra with varying concentrations""")
    return


@app.cell
def _():
    from morgan.createTrainingData import createTrainingData
    import numpy as np

    substanceDict = {
        'Citric acid': ['SP:3368'],
    }

    substanceSpectrumIds = [substanceDict[substance][-1] for substance in substanceDict]
    count = 1000

    labels = []
    spectra = []

    for spectracounter in range (0, count):
        print(f'{spectracounter}/{count}')
        spectra.append(createTrainingData(
            substanceSpectrumIds=substanceSpectrumIds,
            rondomlyScaleSubstances=True,
            # referenceSubstanceSpectrumId='dss',
        ))

    print(''.join(f"{x['scales']}\n'" for x in spectra)) #Dict including scaling references
    print(spectra[0]['intensities'].shape) #Y axis
    print(spectra[0]['positions'].shape) #X axis
    print(spectra[0]['components'].shape) #Peaks of all separate componenets, not import

    return labels, np, spectra, substanceDict


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

    plt.show()
    return (plt,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Now preprocess the full spectra
    - Extract only the relevant parts of the spectra
    """
    )
    return


@app.cell
def _(np, plt, spectra, substanceDict):
    def preprocess_peaks(intensities, positions, ranges):
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

            new_intensities = np.concatenate([new_intensities, temp_intensities])
            new_positions = np.concatenate([new_positions, temp_positions])

        return new_positions, new_intensities

    def preprocess_ratio(scales, substanceDict):
        referenceScale = scales['tsp'][0]

        substanceIds = [substanceDict[id][0] for id in substanceDict]

        substanceScales = [scales[id][0] for id in substanceIds]

        ratios = [scale/referenceScale for scale in substanceScales]

        return ratios

    def preprocess_spectra(spectra, ranges, substanceDict):
        new_positions, new_intensities = preprocess_peaks(spectra['intensities'], spectra['positions'], ranges)

        ratios = preprocess_ratio(spectra['scales'], substanceDict)

        return {
            'intensities': new_intensities,
            'positions': new_positions,
            'scales': spectra['scales'],
            'components': spectra['components'],
            'ratios': ratios,
        }

    ranges = [
        [2.2, 2.8],
        [-0.1, 0.1],
    ]

    preprocessed_spectra = []

    for spectrum in spectra:
        preprocessed_spectra.append(preprocess_spectra(spectrum, ranges, substanceDict))

    print(len(preprocessed_spectra[0]['positions']))
    print(len(preprocessed_spectra[0]['intensities']))

    plt.plot(preprocessed_spectra[0]['positions'], preprocessed_spectra[0]['intensities'])



    return (preprocessed_spectra,)


@app.cell(hide_code=True)
def _(mo, np, spectra, training_data):
    mo.md(
        rf"""
    Shape of spectra: {np.array(spectra).shape}

    ## Define a function to convert the spectra x/y into training tensors

    Shape of tensors: {[training_data[x].shape for x in training_data]}
    """
    )
    return


@app.cell
def _(np, preprocessed_spectra):
    from sklearn.model_selection import train_test_split
    import torch



    def get_training_data_mlp(spectra, train_ratio=0.7, axes=0):
        """
        Input:
            data = [
                data instances, (float32)
                axes, (x/y)
                points along each axis (float32)
            ]

            labels = array of labels (float32)

            train_ratio = ratio of data to be trained. Defaults 0.7, 70% data used for training

            axes = number of axes to remove. Defaults to 0 (no transformation). Use negative numbers

        Output:
            data_train = [
                data instances, (float32)
                points along both axes in single vector, (float32)
            ]
            data_test = identical
            labels_train = array of labels (float32)
            labels_test = identical
        """
        data = []
        labels = []

        for spectrum in spectra:
            data.append(np.concatenate([spectrum['intensities'], spectrum['positions']]))
            labels.append(spectrum['ratios'])

        data=np.array(data)

        data_train, data_test, labels_train, labels_test = train_test_split(
            data, labels, train_size=train_ratio, shuffle=True
        )

        data_train = torch.tensor(data_train, dtype=torch.float32)
        labels_train = torch.tensor(labels_train, dtype=torch.float32).squeeze(1)
        data_test = torch.tensor(data_test, dtype=torch.float32)
        labels_test = torch.tensor(labels_test, dtype=torch.float32).squeeze(1)


        return {
            'data_train': data_train,
            'data_test': data_test,
            'labels_train': labels_train,
            'labels_test': labels_test
        }

    training_data = get_training_data_mlp(spectra=preprocessed_spectra)

    print([training_data[x].shape for x in training_data])
    print(len(training_data['data_train'][0]))

    return torch, training_data


@app.cell
def _(mo):
    mo.md(r"""## Define a pytorch MLP model (fairly simple one)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define a training loop for an MLP model""")
    return


@app.cell
def _(np, r2, torch):
    import tqdm
    import copy
    import torch.optim as optim
    import torch.nn as nn

    def train_mlp_model(
        training_data, trial
    ):
        """Generic training function for regression"""

        n_epochs=int(trial.suggest_float('n_epochs', 10, 100, step=10))
        batch_size=int(trial.suggest_float('batch_size', 10, 100, step=10))
        lr=trial.suggest_float('lr', 1e-5, 1e-1)
        div_size=trial.suggest_float('div_size', 2, 10, step=1)
    
    
        a = len(training_data['data_train'][0])
        b = int(a/div_size)
        c = int(b/div_size)
        d = int(c/div_size)
        e = int(d/div_size)

        # Define the model
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
        )

        data_train = training_data['data_train']
        data_test = training_data['data_test']
        labels_train = training_data['labels_train']
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

            # evaluate accuracy at end of each epoch
            model.eval()
            with torch.no_grad():
                labels_pred = model(data_test)
                test_loss = loss_fn(labels_pred.squeeze(), labels_test)
                test_loss = float(test_loss)
                history.append(test_loss)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_weights = copy.deepcopy(model.state_dict())

        # Final evaluation for regression
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            labels_pred = model(data_test).squeeze()

            # Calculate regression metrics instead of classification accuracy
            mae = torch.mean(torch.abs(labels_pred - labels_test))
            rmse = torch.sqrt(torch.mean((labels_pred - labels_test)**2))

            # Calculate R² score
            ss_res = torch.sum((labels_test - labels_pred) ** 2)
            ss_tot = torch.sum((labels_test - torch.mean(labels_test)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)

            print(f'Final MAE: {mae:.6f}')
            print(f'Final RMSE: {rmse:.6f}')
            print(f'R² Score: {r2_score:.4f}')
            print(f'Best Loss: {best_loss:.6f}')

            # Show some example predictions vs true values
            print(f'First 10 predictions: {labels_pred[:10]}')
            print(f'First 10 true labels: {labels_test[:10]}')
            print(f'Prediction errors: {torch.abs(labels_pred[:10] - labels_test[:10])}')

        return mae, rmse, r2
    return (train_mlp_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Train the model""")
    return


@app.cell
def _():
    # best_loss, best_weights, history, metrics = train_mlp_model(training_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now hyperparameter optimisation""")
    return


@app.cell
def _(train_mlp_model, training_data):
    import optuna
    from functools import partial

    trials = 10

    def objective(training_data, trial):
        mae, rmse, r2 = train_mlp_model(training_data, trial)
    
        score = mae + rmse - r2
        return score

    study = optuna.create_study(
        direction='minimize',
        study_name='mlp_study',
        storage='sqlite:///mlp_study.db',
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



    return


if __name__ == "__main__":
    app.run()
