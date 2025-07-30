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
    count = 10

    labels = []
    spectra = []

    for spectracounter in range (0, count):
        print(f'{spectracounter}/{count}')
        spectra.append(createTrainingData(
            substanceSpectrumIds=substanceSpectrumIds,
            rondomlyScaleSubstances=True,
        ))

    print(spectra[0]['scales'])
    print(spectra[0]['intensities'].shape)

    return labels, np, spectra


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
    return


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
def _(labels, np, spectra):
    from sklearn.model_selection import train_test_split
    import torch

    def get_training_data_mlp(data, labels, train_ratio=0.7, axes=0):
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
        data = np.array(data)

        data_train, data_test, labels_train, labels_test = train_test_split(
            data, labels, train_size=train_ratio, shuffle=True
        )

        data_train = torch.tensor(data_train, dtype=torch.float32).view(len(data_train), axes)
        labels_train = torch.tensor(labels_train, dtype=torch.float32)
        data_test = torch.tensor(data_test, dtype=torch.float32).view(len(data_test), axes)
        labels_test = torch.tensor(labels_test, dtype=torch.float32)


        return {
            'data_train': data_train,
            'data_test': data_test,
            'labels_train': labels_train,
            'labels_test': labels_test
        }

    training_data = get_training_data_mlp(data=spectra, labels=labels, axes=-1)
    return torch, training_data


@app.cell
def _(mo):
    mo.md(r"""## Define a pytorch MLP model (fairly simple one)""")
    return


@app.cell
def _():
    import torch.nn as nn

    # Define the model
    no_transform_model = nn.Sequential(
        nn.Linear(1600, 512),  # 1600*800
        nn.ReLU(),
        nn.Linear(512, 128), # 800*24
        nn.ReLU(),
        nn.Linear(128, 32),  # 24*12
        nn.ReLU(),
        nn.Linear(32, 8),  # 12*6
        nn.ReLU(),
        nn.Linear(8, 1),  # 6*1
    )
    return nn, no_transform_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define a training loop for an MLP model""")
    return


@app.cell
def _(nn, np, torch):
    import tqdm
    import copy
    import torch.optim as optim

    def train_mlp_model(
        model,
        training_data,
        n_epochs=100,
        batch_size=10,
        lr=1e-3,
    ):
        """Generic training function for regression"""

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

        return best_loss, best_weights, history, {'mae': mae, 'rmse': rmse, 'r2': r2_score}
    return (train_mlp_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Train the model""")
    return


@app.cell
def _(no_transform_model, train_mlp_model, training_data):
    best_loss, best_weights, history, metrics = train_mlp_model(no_transform_model, training_data)


    return


@app.cell
def _(labels, np):
    print(np.mean(labels))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Problem: the model is spitting out a single prediction no matter what.

    Potential solutions:

    - Increase the data quantity (didn't work)
    - Reduce concentration of DSS (didn't work)
    - Alter model:
        - Add extra layer for a more gentle reduction
        - Add dropout regularization
    - Alter trianing parameters
    - Normalize the data when preprocessing
    """
    )
    return


if __name__ == "__main__":
    app.run()
